"""Owned run/checkpoint bookkeeping. No cluster-wide cleanup or external APIs."""
from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
import shutil
import time

REPO = Path(__file__).resolve().parents[1]
OWNER_NAME = 'RUN_OWNER.json'
COMPLETE_NAME = 'CHECKPOINT_COMPLETE.json'
REGISTRY_NAME = 'checkpoint_registry.json'


def sha256(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('w', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def under(path, parent):
    path, parent = Path(path), Path(parent).resolve()
    resolved = path.resolve()
    if not resolved.is_relative_to(parent) or resolved == parent:
        raise RuntimeError('Path is outside its owned directory')
    return resolved


def owner(run):
    run = Path(run).resolve()
    data = read_json(run/OWNER_NAME)
    if data.get('schema') != 'svg_baseline_owned_run_v1' or data.get('run_dir') != str(run):
        raise RuntimeError('Missing or mismatched owned-run identity')
    if not data.get('run_id') or not data.get('contract_sha256'):
        raise RuntimeError('Incomplete owned-run identity')
    return data


def checkpoint_path(run, step):
    run = Path(run).resolve()
    step = int(step)
    if step < 1:
        raise RuntimeError('Invalid checkpoint step')
    path = run/f'global_step_{step}'
    if path.is_symlink():
        raise RuntimeError('Checkpoint must not be a symlink')
    return under(path, run)


def verify_checkpoint(run, step, *, full_hash=True):
    run = Path(run).resolve()
    identity = owner(run)
    path = checkpoint_path(run, step)
    record = read_json(path/COMPLETE_NAME)
    if any(record.get(k) != identity[k] for k in ['run_id', 'contract_sha256']):
        raise RuntimeError('Checkpoint belongs to another run or recipe')
    if record.get('step') != int(step) or not record.get('complete'):
        raise RuntimeError('Checkpoint is not complete')
    required = {'data.pt', 'actor/model_world_size_1_rank_0.pt',
                'actor/optim_world_size_1_rank_0.pt', 'actor/extra_state_world_size_1_rank_0.pt',
                'actor/huggingface/config.json'}
    files = record.get('files', {})
    if not required.issubset(files):
        raise RuntimeError('Missing single-rank model/optimizer/RNG/dataloader/config state')
    for relative, metadata in files.items():
        target = under(path/relative, path)
        if (path/relative).is_symlink() or not target.is_file() or target.stat().st_size != metadata['bytes']:
            raise RuntimeError('Checkpoint file missing, linked or truncated')
        if full_hash and sha256(target) != metadata['sha256']:
            raise RuntimeError('Checkpoint checksum mismatch')
    return record


def registry(run):
    run = Path(run)
    identity = owner(run)
    path = run/REGISTRY_NAME
    value = read_json(path) if path.exists() else {
        'schema': 'svg_baseline_checkpoint_registry_v1', 'run_id': identity['run_id'],
        'retention': 'latest_two_complete', 'checkpoints': [], 'removed': []}
    if value.get('run_id') != identity['run_id']:
        raise RuntimeError('Checkpoint registry belongs to another run')
    steps = [row['step'] for row in value['checkpoints']]
    if len(steps) != len(set(steps)):
        raise RuntimeError('Duplicate checkpoint registry entries')
    return value


def latest_complete(run, *, full_hash=True):
    """Only registered, source-bound checkpoints are eligible for pickle loading."""
    state = registry(run)
    if not state['checkpoints']:
        return None
    latest = max(state['checkpoints'], key=lambda row: row['step'])
    step = latest['step']
    if sha256(checkpoint_path(run, step)/COMPLETE_NAME) != latest['manifest_sha256']:
        raise RuntimeError('Registered checkpoint manifest changed')
    verify_checkpoint(run, step, full_hash=full_hash)
    return str(checkpoint_path(run, step))


def before_checkpoint_save(run, step):
    """Limit peak to two complete states, keeping the newest one throughout save."""
    run = Path(run).resolve()
    owner(run)
    state = registry(run)
    destination = checkpoint_path(run, step)
    if destination.exists():
        # Never overwrite an incomplete or previously completed save silently.
        raise RuntimeError('Checkpoint target exists; inspect/archive it before retrying')
    complete = sorted(state['checkpoints'], key=lambda row: row['step'])
    if complete and int(step) <= complete[-1]['step']:
        raise RuntimeError('Checkpoint step must advance monotonically')
    for row in complete:
        verify_checkpoint(run, row['step'], full_hash=True)
    while len(complete) >= 2:
        oldest = complete.pop(0)
        path = checkpoint_path(run, oldest['step'])
        # Delete only a fully hashed path listed in this run's registry, never a glob.
        if any(p.is_symlink() for p in path.rglob('*')):
            raise RuntimeError('Refuse checkpoint cleanup through a symlink')
        tombstone = dict(oldest, removed_unix=time.time(), reason='latest_two_complete_pre_save')
        state['removed'].append(tombstone)
        state['checkpoints'] = complete
        atomic_json(run/REGISTRY_NAME, state)
        shutil.rmtree(path)
        print(f'Checkpoint retention removed owned complete step {oldest["step"]}; newer state retained', flush=True)


def after_checkpoint_save(run, step):
    run = Path(run).resolve()
    identity = owner(run)
    path = checkpoint_path(run, step)
    files = {}
    for file in sorted(path.rglob('*')):
        if file.is_symlink():
            raise RuntimeError('Checkpoint symlink is not supported')
        if file.is_file() and file.name != COMPLETE_NAME:
            if file.name.endswith('.tmp') or file.stat().st_size == 0:
                raise RuntimeError('Incomplete checkpoint file')
            files[file.relative_to(path).as_posix()] = {'bytes': file.stat().st_size, 'sha256': sha256(file)}
    required = {'data.pt', 'actor/model_world_size_1_rank_0.pt',
                'actor/optim_world_size_1_rank_0.pt', 'actor/extra_state_world_size_1_rank_0.pt',
                'actor/huggingface/config.json'}
    if not required.issubset(files):
        raise RuntimeError('Do not publish COMPLETE: missing resumable state files')
    record = {'schema': 'svg_baseline_complete_checkpoint_v1', 'complete': True,
              'run_id': identity['run_id'], 'contract_sha256': identity['contract_sha256'],
              'step': int(step), 'created_unix': time.time(), 'files': files}
    atomic_json(path/COMPLETE_NAME, record)
    verify_checkpoint(run, step, full_hash=False)
    state = registry(run)
    if any(row['step'] == int(step) for row in state['checkpoints']):
        raise RuntimeError('Checkpoint already registered')
    state['checkpoints'].append({'step': int(step), 'manifest_sha256': sha256(path/COMPLETE_NAME),
                                'bytes': sum(row['bytes'] for row in files.values())})
    atomic_json(run/REGISTRY_NAME, state)
    tracker = run/'latest_checkpointed_iteration.txt'
    temporary = tracker.with_suffix('.txt.tmp')
    with temporary.open('w', encoding='utf-8') as stream:
        stream.write(str(int(step)))
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, tracker)


def trusted_torch_load(path):
    """PyTorch>=2.6 compatibility ONLY for this run's hashed checkpoint files.

    A checksum provides integrity, not a sandbox: the work directory must be
    trusted and must never be populated with someone else's pickle checkpoint.
    """
    path = Path(path).resolve()
    folder = next((p for p in path.parents if p.name.startswith('global_step_')), None)
    if folder is None:
        raise RuntimeError('Refuse unowned pickle checkpoint')
    step = int(folder.name.removeprefix('global_step_'))
    manifest = verify_checkpoint(folder.parent, step, full_hash=False)
    relative = path.relative_to(folder).as_posix()
    metadata = manifest['files'].get(relative)
    if not metadata or sha256(path) != metadata['sha256']:
        raise RuntimeError('Refuse unverified pickle checkpoint file')
    import torch
    return torch.load(str(path), map_location='cpu', weights_only=False)


def add_source_paths():
    import sys
    for path in [REPO, REPO/'scripts', REPO/'vendor/VAGEN', REPO/'vendor/verl']:
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))


def compose_config(work_dir, smoke=False, resume=False):
    from omegaconf import OmegaConf
    work = Path(work_dir).expanduser().resolve()
    # Upstream base config has no external defaults. Merge is explicit, so a
    # caller does not accidentally inherit a private launcher or path.
    base = OmegaConf.load(REPO/'vendor/VAGEN/vagen/trainer/config/ppo_trainer.yaml')
    config = OmegaConf.merge(base, OmegaConf.load(REPO/'configs/common.yaml'),
                             OmegaConf.load(REPO/'configs/baseline50.yaml'))
    OmegaConf.set_struct(config, False)
    config.runtime.work_dir = str(work)
    config.runtime.repo_dir = str(REPO)
    config.trainer.resume_mode = 'auto' if resume else 'disable'
    config.trainer.del_local_ckpt_after_load = False
    config.trainer.remove_previous_ckpt_in_save = False
    if smoke:
        config.data.train_files = str(work/'data/smoke1.parquet')
        config.data.train_batch_size = 1
        config.actor_rollout_ref.actor.ppo_mini_batch_size = 8
        config.actor_rollout_ref.rollout.max_num_seqs = 8
        config.rollout_manager.mini_batch_size = 8
        config.trainer.total_training_steps = 1
        config.trainer.total_epochs = 1
        config.trainer.experiment_name = 'smoke1'
        config.trainer.val_before_train = False
        config.trainer.test_freq = -1
        config.trainer.save_freq = 1
    OmegaConf.resolve(config)
    missing = OmegaConf.missing_keys(config)
    if missing:
        raise RuntimeError('Unresolved config fields: ' + ', '.join(sorted(missing)))
    return config


def safe_environment(work_dir):
    """Propagate required V0/offline flags, never print or transfer credentials."""
    work = Path(work_dir).resolve()
    env = dict(os.environ)
    selected = selected_gpu(env)
    for name in list(env):
        if any(marker in name.upper() for marker in ['API_KEY', 'ACCESS_TOKEN', 'AUTH_TOKEN', 'SECRET']):
            env.pop(name, None)
    env.update(VLLM_USE_V1='0', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
               HF_HUB_DISABLE_TELEMETRY='1', VLLM_NO_USAGE_STATS='1', DO_NOT_TRACK='1',
               WANDB_MODE='disabled', TOKENIZERS_PARALLELISM='false',
               NCCL_DEBUG='WARN', PYTHONHASHSEED='2026', PYTHONUNBUFFERED='1',
               PYTHONDONTWRITEBYTECODE='1', CUDA_VISIBLE_DEVICES=selected,
               PYTHONPATH=os.pathsep.join(map(str, [REPO, REPO/'scripts',
                                                   REPO/'vendor/VAGEN', REPO/'vendor/verl'])),
               HF_HOME=str(work/'cache/huggingface'), TORCH_HOME=str(work/'cache/torch'))
    return env


def selected_gpu(env=None):
    value = (os.environ if env is None else env).get('CUDA_VISIBLE_DEVICES', '0').strip()
    if not re.fullmatch(r'(?:[0-9]+|GPU-[A-Za-z0-9-]+)', value):
        raise RuntimeError('Select exactly one full GPU by CUDA_VISIBLE_DEVICES=index or GPU-UUID')
    return value
