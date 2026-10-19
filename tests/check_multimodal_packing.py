"""CPU test of native VAGEN packing with the real Qwen processor, no model load.

Dependencies that import GPU training machinery are isolated by AST loading
their original pure functions. This tests token/image/mask/RoPE construction,
not FSDP/vLLM/LLM log-probability correctness.
"""
import argparse
import ast
from pathlib import Path
import sys
import types


def definitions(path, names, namespace):
    nodes = [n for n in ast.parse(Path(path).read_text(encoding='utf-8')).body
             if isinstance(n, ast.FunctionDef) and n.name in names]
    if {n.name for n in nodes} != set(names):
        raise ValueError('Missing pinned function')
    future = ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0)
    module = ast.fix_missing_locations(ast.Module(body=[future] + nodes, type_ignores=[]))
    exec(compile(module, str(path), 'exec'), namespace)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--vagen', required=True)
    ap.add_argument('--verl', required=True)
    ap.add_argument('--model', required=True)
    args = ap.parse_args()
    import torch
    from PIL import Image
    from transformers import AutoProcessor
    from svg_opsd.teacher import teacher_context, actor_segments, assert_same_actor_tokens
    processor = AutoProcessor.from_pretrained(args.model, local_files_only=True, use_fast=False)
    processor.tokenizer.add_special_tokens({'additional_special_tokens': ['<RETHINK>']})
    # No model is loaded or resized; this is a tokenizer/processor CPU test only.
    ns = {'torch': torch, 'F': torch.nn.functional}
    definitions(Path(args.verl)/'verl/utils/torch_functional.py',
                ['pad_sequence_to_length', 'tokenize_and_postprocess_data'], ns)
    verl_f = types.SimpleNamespace(**{k: ns[k] for k in
          ['pad_sequence_to_length', 'tokenize_and_postprocess_data']})
    definitions(Path(args.verl)/'verl/utils/model.py', ['compute_position_id_with_mask'], ns)
    definitions(Path(args.verl)/'verl/models/transformers/qwen2_vl.py', ['get_rope_index'], ns)
    module = types.ModuleType('verl.models.transformers.qwen2_vl')
    module.get_rope_index = ns['get_rope_index']
    sys.modules[module.__name__] = module
    ns.update(verl_F=verl_f)
    source = Path(args.vagen)/'vagen/rollout/qwen_rollout/rollout_manager_service.py'
    klass = next(n for n in ast.parse(source.read_text(encoding='utf-8')).body if isinstance(n, ast.ClassDef))
    methods = {'_handle_special_tokens', '_handle_multi_modal_data', '_compute_loss_mask',
               '_single_recording_to_prompt', '_generate_input_for_uptate'}
    klass.body = [n for n in klass.body if isinstance(n, ast.FunctionDef) and n.name in methods]
    future = ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[future, klass], type_ignores=[])),
                 str(source), 'exec'), ns)
    manager = ns[klass.name]()
    manager.tokenizer, manager.processor = processor.tokenizer, processor
    manager.system_prompts = {'e': 'Reproduce the target image as SVG.'}
    manager.config = types.SimpleNamespace(special_token_for_loss_mask=['<|box_start|>', '<|box_end|>'],
        max_trajectory_length=4096, truncation='error', use_multi_turn_reward=False,
        use_loss_mask=True, use_gae_mask=True)
    first = '<think>A red circle is centered.</think><RETHINK><answer><svg><circle cx="50" cy="50" r="30"/></svg></answer>'
    second = '<think>Keep the circle and match the color.</think><answer><svg><circle fill="red" cx="50" cy="50" r="30"/></svg></answer>'
    records = [
        {'env_id': 'e', 'obs_str': 'Target: <image>', 'image_data': [Image.new('RGB', (112, 112), 'red')], 'info': {}, 'reward': 0.},
        {'env_id': 'e', 'obs_str': 'Rendered result: <image>', 'image_data': [Image.new('RGB', (112, 112), 'blue')], 'info': {'llm_raw_response': first}, 'reward': 1.},
        {'env_id': 'e', 'obs_str': 'Final result: <image>', 'image_data': [Image.new('RGB', (112, 112), 'green')], 'info': {'llm_raw_response': second}, 'reward': 2.},
    ]
    student = manager._generate_input_for_uptate(records, 2, None)
    slen = len(student['responses'])
    spans = actor_segments(student['loss_mask'][-slen:])
    assert len(spans) == 2
    assert student['position_ids'].shape == (3, 4096)
    assert student['multi_modal_inputs']['image_grid_thw'].shape[0] == 2
    for target, hint in [(0, 'Use a red fill.'), (1, None)]:
        with teacher_context(manager, 'Shared geometry skill.', target, hint):
            teacher = manager._generate_input_for_uptate(records, target + 1, None)
        tlen = len(teacher['responses'])
        tspans = actor_segments(teacher['loss_mask'][-tlen:])
        assert len(tspans) == target + 1
        assert_same_actor_tokens(student['responses'], spans[target], teacher['responses'], tspans[-1])
        text = processor.tokenizer.decode(teacher['input_ids'][teacher['attention_mask'].bool()], skip_special_tokens=False)
        assert ('Use a red fill.' in text) == (target == 0)
        assert teacher['multi_modal_inputs']['image_grid_thw'].shape[0] == target + 1
        assert teacher['position_ids'].shape == (3, 4096)
        assert not torch.equal(teacher['position_ids'], student['position_ids'])
    plain = manager._generate_input_for_uptate(records, 2, None)
    torch.testing.assert_close(plain['input_ids'], student['input_ids'])
    torch.testing.assert_close(plain['multi_modal_inputs']['pixel_values'], student['multi_modal_inputs']['pixel_values'])
    print('PASS native multimodal Teacher packing: 2 decisions; same Actor token IDs; image order/grid; fresh 3-axis RoPE; no Student mutation; no future-hint leak')

    # Drive the REAL attach_teacher_scores function; only RPC/DataProto transport
    # is replaced. The pinned native collator and native multimodal packing stay.
    from collections import defaultdict
    import numpy as np
    ns.update(defaultdict=defaultdict, np=np)
    definitions(Path(args.verl)/'verl/utils/dataset/rl_dataset.py', ['collate_fn'], ns)
    class Batch:
        def __init__(self, values):
            self.batch = {k: v for k, v in values.items() if isinstance(v, torch.Tensor)}
            self.non_tensor_batch = {k: v for k, v in values.items() if not isinstance(v, torch.Tensor)}
            self.meta_info = {}
        @classmethod
        def from_single_dict(cls, values):
            return cls(values)
        def __len__(self):
            return self.batch['responses'].shape[0]
    class CurrentActor:
        world_size = 2
        def __init__(self):
            self.calls = 0
        def compute_log_prob(self, batch):
            assert not torch.is_grad_enabled()
            assert len(batch) % self.world_size == 0
            assert 'multi_modal_inputs' in batch.non_tensor_batch
            self.calls += 1
            # Nonconstant scores expose shifted/misplaced scatter indices.
            values = -(batch.batch['responses'].float() % 97) / 97 - 0.5
            return types.SimpleNamespace(batch={'old_log_probs': values})
    parent_module = types.ModuleType('verl')
    parent_module.DataProto = Batch
    dataset_module = types.ModuleType('verl.utils.dataset.rl_dataset')
    dataset_module.collate_fn = ns['collate_fn']
    sys.modules['verl'] = parent_module
    sys.modules[dataset_module.__name__] = dataset_module
    manager.envs, manager.recorder = {'e': None}, {'e': records}
    manager.env_states = {'e': {'step': 2}}
    manager.actor_rollout_wg = CurrentActor()
    class Config(dict):
        __getattr__ = dict.__getitem__
    cfg = Config(arm='baseline', teacher_score_batch_size=1,
                 static_skill_file=str(Path(__file__).resolve().parents[1]/'configs/svg_static_skill.txt'))
    packed = Batch.from_single_dict(ns['collate_fn']([student]))
    from svg_opsd.teacher import attach_teacher_scores
    output = attach_teacher_scores(manager, packed, cfg)
    active = output.batch['loss_mask'][:, -slen:].bool()
    expected = -(output.batch['responses'].float() % 97) / 97 - 0.5
    torch.testing.assert_close(output.batch['teacher_log_probs'][active], expected[active])
    assert not output.batch['teacher_log_probs'][~active].any()
    assert manager.actor_rollout_wg.calls == 2
    torch.testing.assert_close(output.batch['input_ids'][0], student['input_ids'])
    print('PASS real attach_teacher_scores: current-Actor no-grad RPC, world-size padding, native image collator, per-decision exact-token scatter, environment zero mask')


if __name__ == '__main__':
    main()

