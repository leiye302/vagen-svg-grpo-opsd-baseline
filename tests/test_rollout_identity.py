"""Execute the real derived reset method with transport-only dummy clients."""
import ast
from collections import defaultdict
from pathlib import Path
import unittest
import types


class Config:
    def __init__(self, frozen_task_id):
        self.frozen_task_id = frozen_task_id
    def config_id(self):
        return self.frozen_task_id


class Client:
    def close_batch(self, ids):
        pass
    def create_environments_batch(self, configs):
        pass
    def reset_batch(self, ids):
        # Response completion order deliberately differs from request order.
        return {key: ({'obs_str': 'target', 'multi_modal_data': {}}, {}) for key in reversed(list(ids))}
    def get_system_prompts_batch(self, ids):
        return {key: 'SVG' for key in ids}
    def compute_reward_batch(self, ids):
        return {key: 0.0 for key in ids}


class IdentityTest(unittest.TestCase):
    def setUp(self):
        root = Path(__file__).resolve().parents[1]
        source = root/'overlays/VAGEN/vagen/rollout/qwen_rollout/rollout_manager_service.py'
        node = ast.parse(source.read_text(encoding='utf-8'))
        cls = next(item for item in node.body if isinstance(item, ast.ClassDef))
        reset = next(item for item in cls.body if isinstance(item, ast.FunctionDef) and item.name == 'reset')
        generate = next(item for item in cls.body if isinstance(item, ast.FunctionDef) and item.name == 'generate_batch_for_update')
        reset.decorator_list = []
        generate.decorator_list = []
        future = ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0)
        module = ast.fix_missing_locations(ast.Module(body=[future, reset, generate], type_ignores=[]))
        ns = {'defaultdict': defaultdict, 'REGISTERED_ENV': {'svg': {'config_cls': Config}},
              'DataProto': types.SimpleNamespace(from_single_dict=lambda value: value),
              'collate_fn': lambda rows: rows}
        exec(compile(module, str(source), 'exec'), ns)
        self.manager = type('Manager', (), {'reset': ns['reset'], 'generate_batch_for_update': ns['generate_batch_for_update'],
            'record': lambda self, env_id, **kw: self.recorder[env_id].append(kw)})()
        self.manager.envs = None
        self.manager.recorder = None
        self.manager.split = 'train'
        self.manager.env_client = Client()
        self.manager.config = types.SimpleNamespace(use_multi_turn_reward=False)

    def reset(self, names):
        self.manager.reset([{'env_name': 'svg', 'env_config': {'frozen_task_id': name},
                             'seed': i} for i, name in enumerate(names)])
        self.assertEqual([cfg.frozen_task_id for cfg in self.manager.envs.values()], names)
        self.assertEqual(len(self.manager.envs), len(names))

    def test_new_pool(self):
        self.reset(['A', 'A', 'B', 'B'])

    def test_reused_pool_reversed_task_groups(self):
        self.reset(['A']*8 + ['B']*8)
        self.reset(['B']*8 + ['A']*8)

    def test_mixed_reuse_and_new_tasks(self):
        self.reset(['A', 'A', 'B', 'B'])
        self.reset(['B', 'A', 'C', 'A'])

    def test_smaller_next_batch(self):
        self.reset(['A', 'A', 'B', 'B'])
        self.reset(['B', 'A'])

    def test_real_generated_rows_match_preassigned_uid_groups(self):
        # First input has A/B/D groups. Next input reorders retained groups and
        # adds C. A dictionary-order-only check is insufficient: drive the real
        # generation-batch serializer and join with the trainer's preassigned uid.
        self.reset(['A']*8 + ['B']*8 + ['D']*8)
        names = ['B']*8 + ['C']*8 + ['A']*8
        self.reset(names)
        for env_id, cfg in self.manager.envs.items():
            self.manager.recorder[env_id] = [{'task_id': cfg.frozen_task_id}]
        self.manager._generate_input_for_uptate = lambda recording, step, window_size: {
            'observed_task_id': recording[0]['task_id'], 'step_reward_sum': 0.0}
        outputs = self.manager.generate_batch_for_update()
        uids = ['uid_B']*8 + ['uid_C']*8 + ['uid_A']*8
        grouped = defaultdict(list)
        for uid, output in zip(uids, outputs, strict=True):
            grouped[uid].append(output['observed_task_id'])
        self.assertEqual(grouped, {'uid_B': ['B']*8, 'uid_C': ['C']*8, 'uid_A': ['A']*8})


if __name__ == '__main__':
    unittest.main()
