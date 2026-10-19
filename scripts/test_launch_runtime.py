"""CPU-only tests of exact patches and owned-checkpoint safety; never train."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import apply_runtime_fixes as fixes
import baseline_runtime as runtime
import preflight


class OwnedCheckpointTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)/'run'
        self.root.mkdir()
        runtime.atomic_json(self.root/'RUN_OWNER.json', {'schema': 'svg_baseline_owned_run_v1',
            'run_id': 'unit-test-run', 'contract_sha256': 'frozen-recipe', 'run_dir': str(self.root.resolve())})

    def tearDown(self): self.temp.cleanup()

    def save(self, step):
        runtime.before_checkpoint_save(self.root, step)
        checkpoint = self.root/f'global_step_{step}'
        for name in ['data.pt', 'actor/model_world_size_1_rank_0.pt', 'actor/optim_world_size_1_rank_0.pt',
                     'actor/extra_state_world_size_1_rank_0.pt', 'actor/huggingface/config.json']:
            target = checkpoint/name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes((name + ':' + str(step)).encode())
        runtime.after_checkpoint_save(self.root, step)
        return checkpoint

    def test_complete_identity_and_tracker(self):
        path = self.save(20)
        self.assertEqual(runtime.latest_complete(self.root), str(path.resolve()))
        self.assertEqual((self.root/'latest_checkpointed_iteration.txt').read_text(), '20')
        self.assertTrue(runtime.verify_checkpoint(self.root, 20)['complete'])

    def test_two_state_peak_and_only_own_complete_deletion(self):
        self.save(20); self.save(40)
        unrelated = self.root/'global_step_999'
        unrelated.mkdir(); (unrelated/'keep').write_text('not ours')
        runtime.before_checkpoint_save(self.root, 50)
        self.assertFalse((self.root/'global_step_20').exists())
        self.assertTrue((self.root/'global_step_40').exists())
        self.assertTrue(unrelated.exists())
        self.assertEqual([row['step'] for row in runtime.registry(self.root)['checkpoints']], [40])

    def test_corrupt_newest_blocks_oldest_deletion(self):
        self.save(20); latest = self.save(40)
        (latest/'data.pt').write_text('corruption')
        with self.assertRaises(RuntimeError): runtime.before_checkpoint_save(self.root, 50)
        self.assertTrue((self.root/'global_step_20').exists())

    def test_incomplete_target_never_overwritten(self):
        (self.root/'global_step_20').mkdir()
        with self.assertRaises(RuntimeError): runtime.before_checkpoint_save(self.root, 20)

    def test_no_external_pickle(self):
        external = Path(self.temp.name)/'foreign.pt'
        external.write_bytes(b'untrusted')
        with self.assertRaises(RuntimeError): runtime.trusted_torch_load(external)

    def test_mismatched_owner_refused(self):
        checkpoint = self.save(20)
        record = runtime.read_json(checkpoint/runtime.COMPLETE_NAME)
        record['run_id'] = 'other'
        runtime.atomic_json(checkpoint/runtime.COMPLETE_NAME, record)
        with self.assertRaises(RuntimeError): runtime.verify_checkpoint(self.root, 20)

    def test_missing_optimizer_not_complete(self):
        checkpoint = self.root/'global_step_20'
        checkpoint.mkdir(); (checkpoint/'data.pt').write_bytes(b'test')
        with self.assertRaises(RuntimeError): runtime.after_checkpoint_save(self.root, 20)
        self.assertIsNone(runtime.latest_complete(self.root))


class RuntimePatchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = runtime.REPO/'vendor'

    def fixtures(self, root):
        for name, patches in fixes.PATCHES.items():
            for relative in patches:
                source = self.source/name/relative
                ledger = self.source/name/'.baseline_runtime_fixes.json'
                if ledger.exists() and relative in runtime.read_json(ledger):
                    source = self.source/name/runtime.read_json(ledger)[relative]['original']
                target = root/'vendor'/name/relative
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(source.read_bytes())

    def test_real_source_exact_patch_idempotent(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); self.fixtures(root)
            self.assertEqual(fixes.apply(root), {'VAGEN': 1, 'verl': 2})
            self.assertEqual(fixes.apply(root, True), {'VAGEN': 1, 'verl': 2})
            self.assertEqual(fixes.apply(root), {'VAGEN': 1, 'verl': 2})
            trainer = (root/'vendor/VAGEN/vagen/trainer/ppo/ray_trainer.py').read_text()
            self.assertIn('if self.global_steps >= self.total_training_steps:', trainer)
            self.assertIn('self.global_steps == 0 and self.val_reward_fn', trainer)
            self.assertNotIn('self.config.trainer.test_freq <= 0 or', trainer)
            self.assertIn('before_checkpoint_save', trainer)
            self.assertIn('after_checkpoint_save', trainer)

    def test_source_drift_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); self.fixtures(root); fixes.apply(root)
            source = root/'vendor/VAGEN/vagen/trainer/ppo/ray_trainer.py'
            source.write_text(source.read_text() + '\n# drift\n')
            with self.assertRaises(RuntimeError): fixes.apply(root)
            self.assertTrue(source.read_text().endswith('# drift\n'))

    def test_wrong_preimage_fails_before_write(self):
        with self.assertRaises(RuntimeError): fixes.patch_trainer('different program')
        with self.assertRaises(RuntimeError): fixes.patch_vllm('different program')

    def test_disabled_validation_semantics(self):
        for frequency, step, expected in [(-1, 1, False), (10, 50, False), (10, 49, True)]:
            self.assertEqual(frequency > 0 and step % frequency != 0, expected)

    def test_memory_budget_uses_actual_fp32_adam_not_seven_bf16(self):
        count = 3_700_000_000
        report = preflight.checkpoint_budget(count, 100*preflight.GIB)
        self.assertEqual(report['planned_bytes_per_complete_checkpoint'], 12*count + preflight.GIB)
        self.assertEqual(report['peak_checkpoint_copies'], 2)
        smoke = preflight.checkpoint_budget(count, 100*preflight.GIB, smoke=True)
        self.assertEqual(smoke['peak_checkpoint_copies'], 1)

    def test_existing_gpu_selection_preserved(self):
        with patch.dict('os.environ', {'CUDA_VISIBLE_DEVICES': '3'}):
            self.assertEqual(runtime.safe_environment('/work')['CUDA_VISIBLE_DEVICES'], '3')
        with patch.dict('os.environ', {'CUDA_VISIBLE_DEVICES': 'GPU-abcd-1234'}):
            self.assertEqual(runtime.safe_environment('/work')['CUDA_VISIBLE_DEVICES'], 'GPU-abcd-1234')
        with patch.dict('os.environ', {'CUDA_VISIBLE_DEVICES': '0,1'}):
            with self.assertRaises(RuntimeError): runtime.safe_environment('/work')

    def test_teacher_static_skill_is_in_frozen_source_contract(self):
        import run
        original = run.source_fingerprint()
        key = 'configs/svg_static_skill.txt'
        self.assertIn(key, original)
        real_hash = run.sha256
        def changed_skill(path):
            return 'changed-static-skill' if Path(path).name == 'svg_static_skill.txt' else real_hash(path)
        with patch.object(run, 'sha256', side_effect=changed_skill):
            changed = run.source_fingerprint()
        self.assertNotEqual(original, changed)
        self.assertEqual(changed[key], 'changed-static-skill')

    @unittest.skipUnless(importlib.util.find_spec('omegaconf'), 'OmegaConf not installed')
    def test_effective_config_formal_and_smoke(self):
        with tempfile.TemporaryDirectory() as temporary:
            cfg = runtime.compose_config(temporary)
            self.assertEqual(cfg.trainer.total_training_steps, 50)
            self.assertEqual(cfg.trainer.test_freq, 10)
            self.assertEqual(cfg.trainer.save_freq, 20)
            self.assertIsNone(cfg.preparation.route_sft_manifest)
            smoke = runtime.compose_config(temporary, True)
            self.assertEqual(smoke.trainer.total_training_steps, 1)
            self.assertEqual(smoke.rollout_manager.n_trajectory, 8)
            self.assertEqual(smoke.rollout_manager.max_turns, 2)
            self.assertEqual(smoke.data.max_response_length, 750)
            self.assertEqual(smoke.data.train_batch_size, 1)
            self.assertEqual(smoke.actor_rollout_ref.actor.ppo_mini_batch_size, 8)
            self.assertEqual(smoke.trainer.test_freq, -1)


if __name__ == '__main__':
    unittest.main(verbosity=2)
