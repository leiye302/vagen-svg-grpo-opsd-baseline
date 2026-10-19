"""Read-only source, resource, asset and scientific-contract checks; never train."""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
from pathlib import Path
import shutil
import struct
import subprocess

from baseline_runtime import REPO, compose_config, read_json, sha256, under, selected_gpu

GIB = 1024**3


def safetensor_inventory(model):
    """Count real serialized parameters/dtypes without loading tensor payloads."""
    model = Path(model)
    index = read_json(model/'model.safetensors.index.json')
    count, tensor_bytes, dtypes = 0, 0, {}
    names = set()
    for filename in sorted(set(index['weight_map'].values())):
        path = under(model/filename, model)
        with path.open('rb') as stream:
            length = struct.unpack('<Q', stream.read(8))[0]
            if length > 64*1024**2:
                raise RuntimeError('Invalid safetensors header length')
            header = json.loads(stream.read(length))
        for name, spec in header.items():
            if name == '__metadata__': continue
            if name in names or index['weight_map'].get(name) != filename:
                raise RuntimeError('Safetensors shard/index identity mismatch')
            names.add(name)
            number = math.prod(spec['shape'])
            count += number
            tensor_bytes += spec['data_offsets'][1] - spec['data_offsets'][0]
            dtypes[spec['dtype']] = dtypes.get(spec['dtype'], 0) + number
    if names != set(index['weight_map']):
        raise RuntimeError('Missing safetensors index tensors')
    return {'parameter_elements': count, 'tensor_bytes': tensor_bytes, 'dtypes': dtypes}


def checkpoint_budget(elements, free_bytes, existing_checkpoint_bytes=0, smoke=False):
    # The pinned FSDP actor is FP32, AdamW moments are FP32. Mixed precision
    # affects compute, not those saved tensors: (4+4+4)*P bytes, not 7*BF16 W.
    one = 12*elements + GIB
    copies = 1 if smoke else 2
    remaining = max(0, copies*one - existing_checkpoint_bytes)
    return {'dtype_basis': 'FP32 actor + FP32 Adam first/second moments; 1 GiB serialization/RNG allowance',
            'planned_bytes_per_complete_checkpoint': one, 'peak_checkpoint_copies': copies,
            'additional_checkpoint_bytes': remaining, 'reserve_bytes': 5*GIB,
            'required_free_bytes': remaining + 5*GIB, 'free_bytes': free_bytes,
            'fits': free_bytes >= remaining + 5*GIB,
            'retention': 'before writing third state remove only verified oldest, keep newest complete'}


def verify_assets(work):
    manifest = read_json(work/'assets_manifest.json')
    if manifest.get('complete') is not True or not {'policy', 'dataset', 'dino', 'dreamsim'}.issubset(manifest.get('assets', {})):
        raise RuntimeError('Prepare all public policy/dataset/reward assets, not --only-data')
    if manifest.get('sources_lock_sha256') != sha256(REPO/'sources.lock.json'):
        raise RuntimeError('Public asset revision lock differs from this release')
    if manifest['assets']['dreamsim'].get('offline_cpu_reload_passed') is not True:
        raise RuntimeError('DreamSim cache has not passed disconnected reload')
    files = manifest['files']
    if isinstance(files, list):
        pairs = [(row['path'], row) for row in files]
    else:
        pairs = list(files.items())
    if not pairs:
        raise RuntimeError('Empty public asset manifest')
    for relative, metadata in pairs:
        path = under(work/relative, work)
        if not path.is_file() or sha256(path) != metadata['sha256']:
            raise RuntimeError('Public asset hash mismatch: ' + relative)
        if 'bytes' in metadata and path.stat().st_size != metadata['bytes']:
            raise RuntimeError('Public asset size mismatch: ' + relative)
    return {'manifest_sha256': sha256(work/'assets_manifest.json'), 'verified_files': len(pairs)}


def verify_data(work):
    manifest = read_json(work/'data/dataset_manifest.json')
    if manifest.get('test_used') is not False:
        raise RuntimeError('Official test split must not be used in this training run')
    ids = {}
    for split, filename, count in [('train', 'train500.parquet', 500), ('dev', 'dev64.parquet', 64)]:
        record = manifest['splits'][split]
        if record['count'] != count or len(record['task_ids']) != count or len(set(record['task_ids'])) != count:
            raise RuntimeError('Frozen task count/uniqueness mismatch')
        if sha256(work/'data'/filename) != record['sha256']:
            raise RuntimeError('Frozen parquet hash mismatch')
        ids[split] = set(record['task_ids'])
    if ids['train'] & ids['dev']:
        raise RuntimeError('Train/dev task leakage')
    return {'manifest_sha256': sha256(work/'data/dataset_manifest.json'), 'train': 500, 'dev': 64}


def memory_available():
    if not Path('/proc/meminfo').exists(): return None
    available = next(int(line.split()[1])*1024 for line in Path('/proc/meminfo').read_text().splitlines()
                     if line.startswith('MemAvailable:'))
    # Containers may have a much lower cgroup limit than host free -h reports.
    for maximum, current in [('/sys/fs/cgroup/memory.max', '/sys/fs/cgroup/memory.current'),
                              ('/sys/fs/cgroup/memory/memory.limit_in_bytes',
                               '/sys/fs/cgroup/memory/memory.usage_in_bytes')]:
        if Path(maximum).exists() and Path(current).exists():
            value = Path(maximum).read_text().strip()
            if value.isdigit():
                available = min(available, max(0, int(value)-int(Path(current).read_text())))
    return available


def inspect(work_dir, smoke=False, *, check_gpu=True, check_assets=True):
    work = Path(work_dir).expanduser().resolve()
    report = {'training_started': False, 'gpu_compute_test_performed': False, 'blockers': [], 'warnings': []}
    try:
        config = compose_config(work, smoke=smoke)
        assert config.algorithm.adv_estimator == 'grpo'
        assert config.algorithm.svg_opsd.arm == 'baseline'
        assert config.actor_rollout_ref.actor.sdar_gate_beta == 5
        assert config.actor_rollout_ref.actor.sdar_coef == .01
        assert config.actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu == 1
        assert config.rollout_manager.n_trajectory == 8 and config.rollout_manager.max_turns == 2
        assert config.data.max_response_length == 750 and config.data.truncation == 'error'
        assert config.trainer.total_training_steps == (1 if smoke else 50)
        assert config.preparation.route_sft_manifest is None
        assert config.algorithm.svg_opsd.teacher_score_batch_size == 1
        report['planned_updates'] = config.trainer.total_training_steps
        report['planned_validation_steps'] = [] if smoke else [0, 10, 20, 30, 40, 50]
        report['planned_checkpoint_steps'] = [1] if smoke else [20, 40, 50]
        report['route_sft_required'] = False
    except Exception as exc:
        report['blockers'].append('effective_config:' + type(exc).__name__ + ':' + str(exc))
        report['ready'] = False
        return report
    try:
        subprocess.run([__import__('sys').executable, str(REPO/'scripts/bootstrap_sources.py'), '--verify-only'],
                       check=True, capture_output=True, text=True)
        from apply_runtime_fixes import apply
        report['runtime_patches'] = apply(verify_only=True)
    except Exception as exc:
        report['blockers'].append('pinned_sources_or_runtime_fixes:' + type(exc).__name__)
    expected = {'torch': '2.7.0', 'vllm': '0.9.2', 'transformers': '4.51.3',
                'ray': '2.43.0', 'flash-attn': '2.8.3', 'tensordict': '0.6.2',
                'CairoSVG': '2.9.1', 'Pillow': '11.3.0', 'dreamsim': '0.2.1'}
    report['versions'] = {}
    for package, required in expected.items():
        try:
            actual = importlib.metadata.version(package)
            report['versions'][package] = actual
            if actual.split('+')[0] != required:
                report['blockers'].append('version_mismatch:' + package)
        except importlib.metadata.PackageNotFoundError:
            report['blockers'].append('package_missing:' + package)
    if report['versions'].get('torch', '').endswith('+cu128') is False:
        report['blockers'].append('Blackwell_runtime_requires_torch_cu128')
    try:
        report['data'] = verify_data(work)
        if check_assets: report['assets'] = verify_assets(work)
        inventory = safetensor_inventory(work/'models/Qwen2.5-VL-3B-Instruct')
        report['model'] = inventory
        run = Path(config.trainer.default_local_dir)
        existing = 0
        if (run/'RUN_OWNER.json').exists():
            from baseline_runtime import registry, verify_checkpoint
            for record in registry(run)['checkpoints']:
                verify_checkpoint(run, record['step'])
                existing += record['bytes']
        report['disk'] = checkpoint_budget(inventory['parameter_elements'], shutil.disk_usage(work).free, existing, smoke)
        if not report['disk']['fits']:
            report['blockers'].append('disk_insufficient_for_two_full_checkpoint_peak_and_5GiB_reserve')
        report['cpu_available_bytes'] = memory_available()
        # Actor/master gradients/moments, reference and transient CPU state buffers.
        report['cpu_memory_planning_bytes'] = 24*inventory['parameter_elements'] + 8*GIB
        if report['cpu_available_bytes'] is not None and report['cpu_available_bytes'] < report['cpu_memory_planning_bytes']:
            report['blockers'].append('effective_host_cgroup_memory_below_offload_planning_budget')
    except Exception as exc:
        report['blockers'].append('assets_data_or_storage:' + type(exc).__name__ + ':' + str(exc))
    if check_gpu:
        try:
            selection = selected_gpu()
            gpu = subprocess.check_output(['nvidia-smi', '-i', selection, '--query-gpu=index,uuid,name,memory.total,memory.used',
                                           '--format=csv,noheader,nounits'], text=True, timeout=15).strip().splitlines()
            processes = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid,used_gpu_memory',
                                                  '--format=csv,noheader,nounits'], text=True, timeout=15).strip()
            gpu_uuid = gpu[0].split(',')[1].strip() if len(gpu) == 1 else None
            selected_processes = [line for line in processes.splitlines()
                                  if line.split(',')[0].strip() == gpu_uuid]
            report['gpu_inventory'] = gpu
            report['selected_gpu'] = selection
            report['gpu_compute_processes'] = selected_processes
            if len(gpu) != 1 or float(gpu[0].split(',')[-2]) < 79*1024:
                report['blockers'].append('expected_selected_full_GPU_with_nominal_80GB_or_more')
            if selected_processes or (gpu and float(gpu[0].split(',')[-1]) > 1024):
                report['blockers'].append('GPU_not_free_do_not_disturb_existing_work')
        except Exception as exc:
            report['blockers'].append('gpu_inventory_unavailable:' + type(exc).__name__)
    report['ready'] = not report['blockers']
    report['warnings'].append('Readiness is not a GPU numerical/backward/weight-sync test; run --smoke first.')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--work-dir', required=True)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--cpu-only', action='store_true', help='Skip nvidia-smi; does not certify GPU readiness')
    args = parser.parse_args()
    result = inspect(args.work_dir, args.smoke, check_gpu=not args.cpu_only)
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result['ready'] else 2)
