"""Exact-preimage runtime fixes, with originals, diffs and idempotent SHA records."""
import argparse
import difflib
import hashlib
import json
from pathlib import Path
import os

ROOT = Path(__file__).resolve().parents[1]


def replace_exact(text, old, new, count=1):
    if text.count(old) != count:
        raise RuntimeError(f'Expected {count} exact source blocks; found {text.count(old)}')
    return text.replace(old, new)


def patch_trainer(text):
    text = replace_exact(text, '    def _save_checkpoint(self):\n',
        '    def _save_checkpoint(self):\n'
        '        from baseline_runtime import before_checkpoint_save, after_checkpoint_save\n'
        '        before_checkpoint_save(self.config.trainer.default_local_dir, self.global_steps)\n')
    old = "        # latest checkpointed iteration tracker (for atomic usage)\n" \
          "        local_latest_checkpointed_iteration = os.path.join(self.config.trainer.default_local_dir,\n" \
          "                                                           'latest_checkpointed_iteration.txt')\n" \
          "        with open(local_latest_checkpointed_iteration, 'w') as f:\n" \
          "            f.write(str(self.global_steps))"
    text = replace_exact(text, old,
        '        after_checkpoint_save(self.config.trainer.default_local_dir, self.global_steps)')
    text = replace_exact(text,
        '            global_step_folder = find_latest_ckpt_path(checkpoint_folder)  # None if no latest',
        '            from baseline_runtime import latest_complete\n'
        '            global_step_folder = latest_complete(checkpoint_folder)')
    text = replace_exact(text,
        '            dataloader_state_dict = torch.load(dataloader_local_path)',
        '            from baseline_runtime import trusted_torch_load\n'
        '            dataloader_state_dict = trusted_torch_load(dataloader_local_path)')
    text = replace_exact(text,
        '        self._load_checkpoint()\n\n        # perform validation before training',
        '        self._load_checkpoint()\n'
        '        if self.global_steps >= self.total_training_steps:\n'
        '            return  # An already complete resume must not perform update N+1.\n\n'
        '        # perform validation before training')
    text = replace_exact(text,
        "        if self.val_reward_fn is not None and self.config.trainer.get('val_before_train', True):",
        "        if self.global_steps == 0 and self.val_reward_fn is not None and self.config.trainer.get('val_before_train', True):")
    text = replace_exact(text,
        '                    if self.val_reward_fn is not None and (self.config.trainer.test_freq <= 0 or\n'
        '                            self.global_steps % self.config.trainer.test_freq != 0):',
        '                    if self.val_reward_fn is not None and self.config.trainer.test_freq > 0 and (\n'
        '                            self.global_steps % self.config.trainer.test_freq != 0):')
    return text


def patch_vllm(text):
    return replace_exact(text,
        '                max_num_batched_tokens=max_num_batched_tokens,\n',
        '                max_num_batched_tokens=max_num_batched_tokens,\n'
        '                max_num_seqs=config.max_num_seqs,\n', 2)


def patch_checkpoint(text):
    for variable in ['local_model_path', 'local_optim_path', 'local_extra_state_path']:
        text = replace_exact(text, f'torch.load({variable})', f'trusted_torch_load({variable})')
    text = replace_exact(text, '        model_state_dict = trusted_torch_load(local_model_path)',
        '        from baseline_runtime import trusted_torch_load\n'
        '        model_state_dict = trusted_torch_load(local_model_path)')
    return text


PATCHES = {
    'VAGEN': {'vagen/trainer/ppo/ray_trainer.py': patch_trainer},
    'verl': {'verl/workers/rollout/vllm_rollout/vllm_rollout_spmd.py': patch_vllm,
             'verl/utils/checkpoint/fsdp_checkpoint_manager.py': patch_checkpoint}}


def digest(data):
    return hashlib.sha256(data).hexdigest()


def apply(root=ROOT, verify_only=False):
    summary = {}
    for name, fixes in PATCHES.items():
        directory = root/'vendor'/name
        ledger = directory/'.baseline_runtime_fixes.json'
        records = json.loads(ledger.read_text()) if ledger.exists() else {}
        prepared = []
        for relative, fix in fixes.items():
            target = directory/relative
            current = target.read_bytes()
            if relative in records:
                if digest(current) != records[relative]['after']:
                    raise RuntimeError(f'Recorded patched source changed: {name}/{relative}')
                original = directory/records[relative]['original']
                if digest(original.read_bytes()) != records[relative]['before']:
                    raise RuntimeError('Runtime patch original was modified')
                # A changed patch implementation is not silently accepted.
                recomputed = fix(original.read_text(encoding='utf-8')).encode()
                if digest(recomputed) != records[relative]['after']:
                    raise RuntimeError('Runtime patch implementation differs from recorded result')
                continue
            if verify_only:
                raise RuntimeError(f'Runtime patch not applied: {name}/{relative}')
            text = current.decode().replace('\r\n', '\n')
            updated = fix(text)
            compile(updated, relative, 'exec')
            prepared.append((relative, current, text, updated))
        # Check all replacements before mutating this repository.
        for relative, original, text, updated in prepared:
            backup_rel = '.baseline_runtime_originals/' + relative
            backup = directory/backup_rel
            backup.parent.mkdir(parents=True, exist_ok=True)
            if backup.exists() and backup.read_bytes() != original:
                raise RuntimeError('Refuse to overwrite a different original source')
            backup.write_bytes(original)
            diff = ''.join(difflib.unified_diff(text.splitlines(True), updated.splitlines(True),
                         fromfile=relative + ':before', tofile=relative + ':after'))
            backup.with_suffix(backup.suffix + '.diff').write_text(diff, encoding='utf-8')
            target = directory/relative
            temporary = target.with_suffix(target.suffix + '.runtime_tmp')
            temporary.write_bytes(updated.encode())
            os.replace(temporary, target)
            records[relative] = {'before': digest(original), 'after': digest(updated.encode()),
                                 'original': backup_rel}
        if not verify_only:
            temporary = ledger.with_suffix('.json.tmp')
            temporary.write_text(json.dumps(records, indent=2, sort_keys=True), encoding='utf-8')
            os.replace(temporary, ledger)
        summary[name] = len(records)
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--verify-only', action='store_true')
    args = parser.parse_args()
    print(json.dumps(apply(verify_only=args.verify_only)))
