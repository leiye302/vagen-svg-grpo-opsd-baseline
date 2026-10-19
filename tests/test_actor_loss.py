"""CPU-only tests of the real actor update_policy body and public SDAR math.

The fake DataProto/device shims isolate actor loss plumbing. These tests do not
claim to test distributed FSDP, VLM forward, vLLM or environment execution.
"""
import ast
import copy
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
ACTOR = ROOT / 'overlays/verl/verl/workers/actor/dp_actor.py'
LOSS = ROOT / 'overlays/verl/verl/trainer/ppo/sdar_utils.py'


def functions(path):
    return [n for n in ast.parse(path.read_text(encoding='utf-8')).body
            if isinstance(n, ast.FunctionDef)]


def executable_body(node):
    node = copy.deepcopy(node)
    if isinstance(node.body[0], ast.Expr) and isinstance(node.body[0].value, ast.Constant):
        node.body.pop(0)
    return ast.dump(node, include_attributes=False)


class SourceTests(unittest.TestCase):
    def test_real_hook_selects_consumes_then_backprops(self):
        source = ACTOR.read_text(encoding='utf-8')
        tree = ast.parse(source)
        klass = next(n for n in tree.body if isinstance(n, ast.ClassDef))
        method = next(n for n in klass.body if isinstance(n, ast.FunctionDef) and n.name == 'update_policy')
        text = ast.get_source_segment(source, method)
        self.assertIn("select_keys.append('teacher_log_probs')", text)
        self.assertIn("teacher_log_probs = data['teacher_log_probs']", text)
        self.assertLess(text.index('compute_sdar_loss('), text.index('loss.backward()'))
        self.assertIn('response_mask=response_mask', text)
        self.assertNotIn('ref_log_prob=data', text)

    def test_public_function_ast_equivalence(self):
        # Local audited source is optional when this test is copied to the server.
        original_root = ROOT / 'vendor/SDAR/verl/trainer/ppo'
        if not original_root.exists():
            self.skipTest('Audited reference snapshot not copied on this host')
        ours = {n.name: n for n in functions(LOSS)}
        for name, file in [('compute_sdar_loss', 'sdar_utils.py'), ('agg_loss', 'core_algos.py'),
                           ('compute_policy_loss', 'core_algos.py')]:
            ref = next(n for n in functions(original_root / file) if n.name == name)
            self.assertEqual(executable_body(ours[name]), executable_body(ref), name)


try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, 'torch unavailable; run on existing remote torch environment for CPU tests')
class GradientTests(unittest.TestCase):
    def setUp(self):
        self.mean = lambda values, mask: (values * mask).sum() / mask.sum()
        ns = {'torch': torch, 'verl_F': SimpleNamespace(masked_mean=self.mean)}
        exec(compile(ast.Module(body=functions(LOSS), type_ignores=[]), str(LOSS), 'exec'), ns)
        self.loss = ns['compute_sdar_loss']
        self.pg_loss = ns['compute_policy_loss']

    def test_public_dual_clip_is_preserved(self):
        logp = torch.tensor([[5.]], requires_grad=True).log()
        loss, _, _, lower = self.pg_loss(torch.zeros_like(logp), logp,
                                      -torch.ones_like(logp), torch.ones_like(logp), cliprange=.2)
        self.assertEqual(loss.item(), 3.)
        self.assertEqual(lower.item(), 1.)

    def test_detached_teacher_detached_gate_and_mask(self):
        student = torch.tensor([[-1., -2., -3.]], requires_grad=True)
        teacher = torch.tensor([[-.5, -3., -2.]], requires_grad=True)
        mask = torch.tensor([[1., 1., 0.]])
        loss, _ = self.loss(student, teacher, mask, 5.)
        loss.backward()
        expected = -torch.sigmoid(5. * (teacher.detach() - student.detach())) * mask / mask.sum()
        torch.testing.assert_close(student.grad, expected)
        self.assertIsNone(teacher.grad)
        self.assertEqual(student.grad[0, 2].item(), 0.)

    def test_zero_beta_half_gate_control(self):
        student = torch.tensor([[-1., -2.]], requires_grad=True)
        teacher = torch.tensor([[-.1, -9.]], requires_grad=True)
        loss, metrics = self.loss(student, teacher, torch.ones_like(student), 0.)
        loss.backward()
        torch.testing.assert_close(student.grad, torch.full_like(student, -.25))
        self.assertEqual(metrics['sdar/gate_mean'], .5)

    def run_real_update(self, teacher_mode='valid', coef=.01):
        class Batch(dict):
            @property
            def batch_size(self):
                return next(iter(self.values())).shape[:1]

            def to(self, device):
                return Batch({k: v.to(device) for k, v in self.items()})

        class Proto:
            def __init__(self, batch, non_tensor=None):
                self.batch = Batch(batch)
                self.non_tensor_batch = non_tensor or {}
                self.meta_info = {'temperature': 1.}

            def select(self, batch_keys, non_tensor_batch_keys=()):
                return Proto({k: self.batch[k] for k in batch_keys},
                             {k: self.non_tensor_batch[k] for k in non_tensor_batch_keys})

            def chunk(self, chunks):
                count = self.batch.batch_size[0]
                assert count % chunks == 0
                size = count // chunks
                return [Proto({k: v[i:i + size] for k, v in self.batch.items()},
                              {k: v[i:i + size] for k, v in self.non_tensor_batch.items()})
                        for i in range(0, count, size)]

        class Config(dict):
            __getattr__ = dict.__getitem__

        student = torch.nn.Parameter(torch.tensor([[-1., -2., -3.]]))
        teacher = torch.tensor([[-.5, -2.5, -.1], [-.3, -2., -.1]], requires_grad=True)
        tensor_batch = {
            'responses': torch.ones((2, 3), dtype=torch.long),
            'input_ids': torch.ones((2, 5), dtype=torch.long),
            'attention_mask': torch.ones((2, 5)),
            'position_ids': torch.zeros((2, 3, 5), dtype=torch.long),
            'old_log_probs': torch.zeros((2, 3)),
            'advantages': torch.zeros((2, 3)),
            'loss_mask': torch.tensor([[0., 0., 1., 1., 0.], [0., 0., 1., 1., 0.]]),
            'teacher_log_probs': teacher,
        }
        if teacher_mode == 'missing':
            tensor_batch.pop('teacher_log_probs')
        if teacher_mode == 'shape':
            tensor_batch['teacher_log_probs'] = teacher[:, :2]
        if teacher_mode == 'active_nan':
            tensor_batch['teacher_log_probs'] = teacher.detach().clone()
            tensor_batch['teacher_log_probs'][0, 0] = float('nan')
        if teacher_mode == 'masked_nan':
            tensor_batch['teacher_log_probs'] = teacher.detach().clone()
            tensor_batch['teacher_log_probs'][:, 2] = float('nan')
        proto = Proto(tensor_batch, {'multi_modal_inputs': [{'pixel_values': 'sentinel'}] * 2})
        cfg = Config(use_kl_loss=False, sdar_coef=coef, sdar_gate_beta=5.,
                     loss_agg_mode='token-mean', ppo_mini_batch_size=2,
                     ppo_micro_batch_size_per_gpu=1, ppo_epochs=1,
                     use_dynamic_bsz=False, clip_ratio=.2, entropy_coeff=0.)
        seen = []

        def forward(micro_batch, temperature):
            self.assertEqual(micro_batch['multi_modal_inputs'][0]['pixel_values'], 'sentinel')
            seen.append(True)
            return torch.zeros_like(student), student

        optimizer = torch.optim.SGD([student], lr=.1)

        def step():
            seen.append(student.grad.detach().clone())
            optimizer.step()
            return torch.tensor(0.)

        fake = SimpleNamespace(actor_module=torch.nn.Identity(), config=cfg,
                               actor_optimizer=optimizer, _forward_micro_batch=forward,
                               _optimizer_step=step)
        method = next(n for n in next(n for n in ast.parse(ACTOR.read_text(encoding='utf-8')).body
                                      if isinstance(n, ast.ClassDef)).body
                      if isinstance(n, ast.FunctionDef) and n.name == 'update_policy')

        def append(metrics, entries):
            for key, value in entries.items():
                metrics.setdefault(key, []).append(value)

        ns = {'torch': torch, 'DataProto': Proto, 'compute_sdar_loss': self.loss,
              'compute_sdar_policy_loss': lambda **kw:
                  (kw['log_prob'].sum() * 0., torch.tensor(0.), torch.tensor(0.), torch.tensor(0.)),
              'append_to_dict': append, 'verl_F': SimpleNamespace(masked_mean=self.mean),
              'core_algos': SimpleNamespace(compute_policy_loss=lambda **kw:
                  (kw['log_prob'].sum() * 0., torch.tensor(0.), torch.tensor(0.)))}
        exec(compile(ast.Module(body=[method], type_ignores=[]), str(ACTOR), 'exec'), ns)
        with patch.object(torch.cuda, 'current_device', return_value='cpu'):
            metrics = ns['update_policy'](fake, proto)
        return student, teacher, metrics, seen

    def test_real_update_backpropagates_sdar_with_multimodal_chunk(self):
        student, teacher, metrics, seen = self.run_real_update()
        self.assertGreater(student[0, 0].item(), -1.)
        self.assertEqual(student[0, 2].item(), -3.)
        self.assertIsNone(teacher.grad)
        self.assertEqual(len(metrics['sdar/loss']), 2)
        self.assertEqual(len(seen), 3)

    def test_real_update_preserves_legacy_without_sdar(self):
        student, _, metrics, _ = self.run_real_update(teacher_mode='missing', coef=0.)
        self.assertEqual(student[0, 0].item(), -1.)
        self.assertNotIn('sdar/loss', metrics)

    def test_missing_teacher_fails_closed(self):
        with self.assertRaisesRegex(ValueError, 'teacher scores are missing'):
            self.run_real_update('missing')

    def test_misaligned_teacher_fails_closed(self):
        with self.assertRaisesRegex(ValueError, 'align'):
            self.run_real_update('shape')

    def test_nonfinite_active_teacher_fails_closed(self):
        with self.assertRaisesRegex(ValueError, 'Non-finite'):
            self.run_real_update('active_nan')

    def test_environment_placeholders_do_not_contaminate_loss(self):
        student, _, _, _ = self.run_real_update('masked_nan')
        self.assertTrue(torch.isfinite(student).all())


if __name__ == '__main__':
    unittest.main(verbosity=2)
