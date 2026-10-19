import copy
import ast
from pathlib import Path
import sys
import types
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from svg_opsd.protocol import parse_route, native_environment_text, decode_policy_responses
from svg_opsd.teacher import teacher_prompt, teacher_context, actor_segments, assert_same_actor_tokens
from svg_opsd.fixed_tasks import fixed_sample
from svg_opsd.safety import svg_safety

VALID = '<think>Draw the visible circle.</think><answer><svg><circle/></svg></answer>'


class Tokenizer:
    eos_token = '<|im_end|>'
    eos_token_id = 9
    pad_token_id = 0

    def decode(self, ids, skip_special_tokens=False):
        return ''.join({1:'a', 2:'<special>', 9:self.eos_token, 0:'<pad>'}[i] for i in ids)

    def apply_chat_template(self, chat, **kwargs):
        return ''.join(f'<role:{m["role"]}>{m["content"]}{self.eos_token}\n' for m in chat)


class ContractTests(unittest.TestCase):
    def test_protocol_and_invalid_responses(self):
        self.assertTrue(parse_route(VALID).protocol_valid)
        for bad in [VALID.replace('Draw the visible circle.', ''), VALID+VALID,
                    VALID.replace('</think>', ''), '<answer><svg></svg></answer><think>x</think>']:
            self.assertFalse(parse_route(bad).protocol_valid)
            self.assertEqual(native_environment_text(bad), '')

    def test_only_trailing_transport_tokens_are_removed(self):
        self.assertEqual(decode_policy_responses(Tokenizer(), [[1,2,9,0],[1,9,2,9]]),
                         ['a<special>', 'a<|im_end|><special>'])

    def test_teacher_is_causal_and_student_is_unchanged(self):
        m = types.SimpleNamespace(tokenizer=Tokenizer(), system_prompts={'e':'SVG system'},
            config=types.SimpleNamespace(special_token_for_loss_mask=['<begin>','<end>']))
        m._handle_special_tokens = lambda text, **kwargs: '<begin>'+text+'<end>'
        data = [
            {'env_id':'e','obs_str':'target image','info':{},'reward':0,'image_data':[]},
            {'env_id':'e','obs_str':'first render','info':{'llm_raw_response':VALID},'reward':1,'image_data':[]},
            {'env_id':'e','obs_str':'FUTURE_SENTINEL','info':{'llm_raw_response':VALID},'reward':2,'image_data':[]},
        ]
        before = copy.deepcopy(data)
        with teacher_context(m, 'STATIC_SKILL_SENTINEL', 0, None):
            result = teacher_prompt(m, data, 1, None, True, True)
        self.assertEqual(data,before)
        self.assertIsNone(m._svg_teacher_context)
        self.assertIn('STATIC_SKILL_SENTINEL',result['prompt'])
        self.assertNotIn('FUTURE_SENTINEL',result['prompt'])
        self.assertNotIn('first render',result['prompt'])

    def test_token_alignment(self):
        self.assertEqual(actor_segments([0,1,1,0,1,0]), [[1,2],[4]])
        assert_same_actor_tokens([0,4,5],[1,2],[4,5,0],[0,1])
        with self.assertRaises(RuntimeError):
            assert_same_actor_tokens([0,4,5],[1,2],[4,7,0],[0,1])

    def test_fixed_identity(self):
        cfg = types.SimpleNamespace(fixed_source_index=1,fixed_source_filename='b')
        self.assertEqual(fixed_sample([{'Filename':'a'},{'Filename':'b'}],cfg),{'Filename':'b'})
        cfg.fixed_source_filename='wrong'
        with self.assertRaises(ValueError): fixed_sample([{'Filename':'a'},{'Filename':'b'}],cfg)

    def test_svg_blocks_remote_resources_and_scripts(self):
        self.assertFalse(svg_safety('<svg><script>alert(1)</script></svg>')[0])
        self.assertFalse(svg_safety('<svg><image href="https://example.com/image.png"/></svg>')[0])
        self.assertFalse(svg_safety('<svg><style>.x {fill: URL(https://example.com/x)}</style></svg>')[0])

    def test_answer_must_be_exactly_one_svg(self):
        for body in ['<svg></svg><svg></svg>', 'prose <svg></svg>',
                     '<svg></svg> prose', '```xml\n<svg></svg>\n```']:
            text='<think>x</think><answer>'+body+'</answer>'
            self.assertFalse(parse_route(text).protocol_valid)
            self.assertEqual(native_environment_text(text),'')

    def test_execution_preserves_nested_svg(self):
        source = ROOT/'overlays/VAGEN/vagen/env/svg/env.py'
        tree = ast.parse(source.read_text(encoding='utf-8'))
        method = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == '_extract_svg_code')
        namespace = {}
        exec(compile(ast.Module(body=[method], type_ignores=[]), str(source), 'exec'), namespace)
        extract = namespace['_extract_svg_code']
        nested = '<svg><svg><circle/></svg></svg>'
        self.assertEqual(extract(None, nested), nested)
        self.assertEqual(extract(None, '<think>x</think><answer>'+nested+'</answer>'), nested)
        self.assertEqual(extract(None, '<svg></svg><svg></svg>'), '')

    def test_verify_only_rejects_unmodified_upstream(self):
        sys.path.insert(0, str(ROOT/'scripts'))
        from bootstrap_sources import accepted_hashes
        hashes = {'upstream_sha256':'original', 'overlay_sha256':'opsd'}
        self.assertNotIn('original', accepted_hashes(hashes, verify_only=True))
        self.assertIn('original', accepted_hashes(hashes, verify_only=False))
        self.assertIn('opsd', accepted_hashes(hashes, verify_only=True))
        self.assertIn('runtime', accepted_hashes(hashes, 'runtime', verify_only=True))


if __name__ == '__main__': unittest.main()
