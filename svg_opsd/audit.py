"""Fixed-dev validation identity check and atomic, image-free result evidence."""
import hashlib
import json
import math
from pathlib import Path
import statistics


def validation_report(records, manifest, step, model):
    expected = manifest['splits']['dev']['task_ids']
    actual = [row.get('task_id') for row in records]
    if len(actual) != 64 or len(set(actual)) != 64 or set(actual) != set(expected):
        raise RuntimeError('Validation must contain exactly the frozen dev64, no resampling')
    metrics = {}
    cleaned = []
    for row in records:
        numeric = {}
        for key, value in row['metrics'].items():
            try:
                value = float(value)
            except (ValueError, TypeError):
                continue
            if not math.isfinite(value):
                raise RuntimeError('Nonfinite validation metric')
            numeric[key] = value
            metrics.setdefault(key, []).append(value)
        cleaned.append({'task_id': row['task_id'], 'config_id': row['config_id'],
                        'output_str': row['output_str'], 'metrics': numeric})
    means = {'val/dev64/' + key: statistics.mean(values) for key, values in metrics.items()}
    return {'step': int(step), 'model_start_path': model,
            'dev_parquet_sha256': manifest['splits']['dev']['sha256'],
            'test_used': False, 'task_ids': sorted(actual), 'means': means,
            'records': sorted(cleaned, key=lambda row: row['task_id'])}


def save_fixed_validation(records, config, step):
    manifest_path = Path(config.data.val_files).parent/'dataset_manifest.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    report = validation_report(records, manifest, step, config.actor_rollout_ref.model.path)
    report['manifest_sha256'] = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    root = Path(config.trainer.default_local_dir)/'validation'
    root.mkdir(parents=True, exist_ok=True)
    target = root/f'dev64_step_{int(step):06d}.json'
    data = json.dumps(report, ensure_ascii=False, indent=2)
    if target.exists() and target.read_text(encoding='utf-8') != data:
        raise RuntimeError('Refuse to overwrite different validation evidence at same step')
    if not target.exists():
        temporary = target.with_suffix('.json.tmp')
        temporary.write_text(data, encoding='utf-8')
        temporary.replace(target)
    return report['means']
