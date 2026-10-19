"""Detached, single-owner baseline runner. No automatic retry after failure.

Usage: python scripts/run.py --work-dir ./work [--smoke | --resume]
The ordinary invocation detaches; inspect runs/<name>/status.json and logs/.
Use --foreground when a process supervisor should own the worker directly.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
import uuid
import urllib.request

from baseline_runtime import (REPO, add_source_paths, atomic_json, compose_config,
    latest_complete, owner, read_json, safe_environment, sha256, verify_checkpoint, selected_gpu)


def source_fingerprint():
    files = {}
    for root in [REPO/'scripts', REPO/'svg_opsd', REPO/'configs',
                 REPO/'vendor/VAGEN/vagen', REPO/'vendor/verl/verl']:
        for path in sorted(root.rglob('*')):
            if path.is_file() and path.suffix in {'.py', '.yaml', '.yml', '.json', '.txt'} and '__pycache__' not in path.parts:
                files[path.relative_to(REPO).as_posix()] = sha256(path)
    files['sources.lock.json'] = sha256(REPO/'sources.lock.json')
    return files


def contract(config, work, versions):
    from omegaconf import OmegaConf
    value = OmegaConf.to_container(config, resolve=True)
    value['trainer']['resume_mode'] = 'disable'  # Mode changes control flow, not the recipe.
    value['trainer']['resume_from_path'] = False
    record = {'config': value, 'code_sha256': source_fingerprint(), 'versions': versions,
              'assets_manifest_sha256': sha256(work/'assets_manifest.json'),
              'dataset_manifest_sha256': sha256(work/'data/dataset_manifest.json')}
    encoded = json.dumps(record, sort_keys=True, separators=(',', ':')).encode()
    return record, hashlib.sha256(encoded).hexdigest()


def create_smoke_subset(work):
    """Use one frozen training row; never relabel or sample a test example."""
    from datasets import Dataset
    source = work/'data/train500.parquet'
    selected = Dataset.from_parquet(str(source)).select([0])
    destination = work/'data/smoke1.parquet'
    if destination.exists():
        if Dataset.from_parquet(str(destination)).to_list() != selected.to_list():
            raise RuntimeError('Existing smoke row differs from frozen training task')
    else:
        temporary = destination.with_suffix('.parquet.tmp')
        selected.to_parquet(str(temporary))
        os.replace(temporary, destination)
    return {'source_sha256': sha256(source), 'subset_sha256': sha256(destination),
            'task_id': selected[0]['extra_info']['task_id'], 'trajectories': 8}


def port_available(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(('127.0.0.1', port))


def stop_owned_group(process):
    """Only touch the newly created session, never pkill/ray stop/global workers."""
    if process is None or process.poll() is not None:
        return
    try:
        if os.getpgid(process.pid) != process.pid:
            raise RuntimeError('Child process-group ownership changed')
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            if os.getpgid(process.pid) == process.pid:
                os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=15)
    except ProcessLookupError:
        pass


def wait_service(process, port, token, timeout=120):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError('Environment service exited; inspect its owned log')
        try:
            with urllib.request.urlopen(f'http://127.0.0.1:{port}/baseline_identity', timeout=2) as response:
                data = json.load(response)
            if data.get('owner_token') != token:
                raise RuntimeError('Another service owns the expected port')
            return
        except (OSError, TimeoutError):
            time.sleep(.5)
    raise RuntimeError('Environment service readiness timed out')


def train_child(args):
    work = Path(args.work_dir).resolve()
    os.environ.update(safe_environment(work))
    add_source_paths()
    config = compose_config(work, args.smoke, args.resume)
    run = Path(config.trainer.default_local_dir)
    identity = owner(run)
    # Parent already checked versions, source/data hashes and owned checkpoint.
    from omegaconf import OmegaConf
    frozen = read_json(run/'run_contract.json')
    current = OmegaConf.to_container(config, resolve=True)
    current['trainer']['resume_mode'] = 'disable'
    current['trainer']['resume_from_path'] = False
    if current != frozen['config'] or source_fingerprint() != frozen['code_sha256']:
        raise RuntimeError('Training config or source changed after preflight')
    import ray
    env = safe_environment(work)
    ray_keys = ['VLLM_USE_V1', 'HF_HUB_OFFLINE', 'TRANSFORMERS_OFFLINE', 'WANDB_MODE',
                'HF_HUB_DISABLE_TELEMETRY', 'VLLM_NO_USAGE_STATS', 'DO_NOT_TRACK',
                'TOKENIZERS_PARALLELISM', 'NCCL_DEBUG', 'PYTHONHASHSEED', 'PYTHONUNBUFFERED',
                'PYTHONPATH', 'HF_HOME', 'TORCH_HOME', 'PYTHONDONTWRITEBYTECODE']
    # Do not connect to an existing unrelated Ray cluster. Its environment and
    # model ownership cannot satisfy this run's contract.
    ray.init(address='local', num_cpus=min(16, max(2, os.cpu_count() or 2)),
             object_store_memory=4*1024**3,
             _temp_dir=str(work/'tmp'/('ray_' + identity['run_id'])),
             runtime_env={'env_vars': {key: env[key] for key in ray_keys}},
             include_dashboard=False)
    try:
        from vagen.trainer.main_ppo import run_ppo
        from vagen.utils.compute_score import compute_score
        run_ppo(config, compute_score=compute_score)
    finally:
        ray.shutdown()


def supervisor(args):
    if os.name != 'posix':
        raise RuntimeError('Training launcher requires Linux; CPU source tests work on other systems')
    import fcntl
    work = Path(args.work_dir).expanduser().resolve()
    work.mkdir(parents=True, exist_ok=True)
    lock = (work/'.baseline_training.lock').open('a+')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise RuntimeError('Another owned baseline/smoke supervisor is active')
    os.environ.update(safe_environment(work))
    add_source_paths()
    config = compose_config(work, args.smoke, args.resume)
    run = Path(config.trainer.default_local_dir)
    run.mkdir(parents=True, exist_ok=True)
    status_path = run/'status.json'
    status = {'supervisor_pid': os.getpid(), 'run_dir': str(run), 'started_unix': time.time(),
              'phase': 'preflight', 'training_complete': False, 'smoke': args.smoke}
    atomic_json(status_path, status)
    server = trainer = None
    handles = []
    interrupted = False

    def signal_handler(number, frame):
        nonlocal interrupted
        interrupted = True
        raise KeyboardInterrupt

    for number in [signal.SIGTERM, signal.SIGINT]:
        signal.signal(number, signal_handler)
    # SIGHUP is ignored by this owned detached supervisor; SSH may disconnect.
    signal.signal(signal.SIGHUP, signal.SIG_IGN)
    try:
        from preflight import inspect
        result = inspect(work, args.smoke)
        atomic_json(run/'preflight.json', result)
        if not result['ready']:
            raise RuntimeError('Preflight failed: ' + ', '.join(result['blockers']))
        if args.smoke:
            atomic_json(run/'smoke_subset.json', create_smoke_subset(work))
        frozen, digest = contract(config, work, result['versions'])
        identity_path = run/'RUN_OWNER.json'
        if identity_path.exists():
            existing = owner(run)
            if not args.resume:
                raise RuntimeError('Run already exists; use --resume, never overwrite it')
            if existing['contract_sha256'] != digest:
                raise RuntimeError('Resume requires identical code, data, models, environment and recipe')
            latest = latest_complete(run)
            if latest is None:
                raise RuntimeError('No complete owned checkpoint to resume; inspect before restarting')
            last_step = int(Path(latest).name.removeprefix('global_step_'))
            if last_step >= config.trainer.total_training_steps:
                status.update(phase='already_complete', training_complete=True, completed_step=last_step)
                return 0
        else:
            if args.resume:
                raise RuntimeError('Resume requested but owned run identity does not exist')
            if any(run.glob('global_step_*')):
                raise RuntimeError('Unowned checkpoint directory exists; refusing to start')
            atomic_json(run/'run_contract.json', frozen)
            atomic_json(identity_path, {'schema': 'svg_baseline_owned_run_v1',
                'run_dir': str(run.resolve()), 'run_id': uuid.uuid4().hex,
                'contract_sha256': digest, 'created_unix': time.time(),
                'checkpoint_retention_authorization': 'latest_two_complete_owned_states_only'})
        port = int(config.rollout_manager.base_url.rsplit(':', 1)[1])
        port_available(port)
        token = uuid.uuid4().hex  # Non-secret local process ownership nonce.
        logs = run/'logs'
        logs.mkdir(exist_ok=True)
        stamp = time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())
        service_log = (logs/(stamp + '_environment.log')).open('ab')
        training_log = (logs/(stamp + '_trainer.log')).open('ab')
        handles += [service_log, training_log]
        env = safe_environment(work)
        server = subprocess.Popen([sys.executable, str(REPO/'scripts/env_service.py'),
            '--work-dir', str(work), '--port', str(port), '--owner-token', token],
            cwd=REPO, env=env, stdin=subprocess.DEVNULL, stdout=service_log,
            stderr=subprocess.STDOUT, start_new_session=True)
        status.update(phase='environment_starting', environment_pid=server.pid)
        atomic_json(status_path, status)
        wait_service(server, port, token)
        command = [sys.executable, str(Path(__file__).resolve()), '--work-dir', str(work), '--train-child']
        if args.smoke: command.append('--smoke')
        if args.resume: command.append('--resume')
        trainer = subprocess.Popen(command, cwd=REPO, env=env, stdin=subprocess.DEVNULL,
            stdout=training_log, stderr=subprocess.STDOUT, start_new_session=True)
        status.update(phase='training', trainer_pid=trainer.pid, log=str(training_log.name))
        atomic_json(status_path, status)
        while trainer.poll() is None:
            if server.poll() is not None:
                raise RuntimeError('Owned environment service exited during training')
            status['updated_unix'] = time.time()
            atomic_json(status_path, status)
            time.sleep(5)
        if trainer.returncode:
            raise RuntimeError('Training child exited with code ' + str(trainer.returncode))
        final_step = int(config.trainer.total_training_steps)
        verify_checkpoint(run, final_step)
        if not args.smoke:
            for step in [0, 10, 20, 30, 40, 50]:
                report = read_json(run/f'validation/dev64_step_{step:06d}.json')
                if report['step'] != step or len(set(report['task_ids'])) != 64:
                    raise RuntimeError('Final fixed-dev validation evidence incomplete')
        status.update(phase='complete', completed_step=final_step, training_complete=True)
        return 0
    except KeyboardInterrupt:
        status.update(phase='interrupted', error='Owned supervisor interrupted; complete checkpoint may be resumed')
        return 130
    except Exception as exc:
        status.update(phase='failed', error_type=type(exc).__name__, error=str(exc))
        print('Run stopped: ' + str(exc), flush=True)
        return 1
    finally:
        stop_owned_group(trainer)
        stop_owned_group(server)
        for handle in handles: handle.close()
        status.update(updated_unix=time.time(), supervisor_finished=True, interrupted=interrupted)
        # Inventory only. Never kill remaining processes with broad Ray/GPU commands.
        try:
            gpu_uuid = subprocess.check_output(['nvidia-smi', '-i', selected_gpu(), '--query-gpu=uuid',
                        '--format=csv,noheader'], text=True, timeout=15).strip()
            raw = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid,used_gpu_memory',
                       '--format=csv,noheader,nounits'], text=True, timeout=15).strip()
            remaining = [line for line in raw.splitlines() if line.split(',')[0].strip() == gpu_uuid]
            status['remaining_gpu_processes'] = remaining
            status['gpu_released'] = not bool(remaining)
        except Exception:
            status['gpu_released'] = None
        atomic_json(status_path, status)
        fcntl.flock(lock, fcntl.LOCK_UN)
        lock.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--work-dir', required=True)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--foreground', action='store_true')
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--train-child', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.train_child:
        train_child(args)
        return
    if args.worker or args.foreground:
        raise SystemExit(supervisor(args))
    if os.name != 'posix':
        raise SystemExit('Training is supported on Linux only')
    work = Path(args.work_dir).expanduser().resolve()
    name = 'smoke1' if args.smoke else 'baseline50'
    logdir = work/'runs'/name/'logs'
    logdir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())
    log = logdir/(stamp + '_supervisor.log')
    command = [sys.executable, str(Path(__file__).resolve()), '--work-dir', str(work), '--worker']
    if args.smoke: command.append('--smoke')
    if args.resume: command.append('--resume')
    with log.open('ab') as stream:
        process = subprocess.Popen(command, cwd=REPO, env=safe_environment(work), stdin=subprocess.DEVNULL,
            stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
    print(json.dumps({'supervisor_pid': process.pid, 'log': str(log),
                      'status': str(work/'runs'/name/'status.json'),
                      'note': 'Detached launch submitted; status, not this message, confirms training started.'}, indent=2))


if __name__ == '__main__':
    main()
