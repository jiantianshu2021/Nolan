"""Synthetic request boundaries: no real model, browser, or paid transport."""
import copy
import io
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch
from urllib.error import URLError

from norlan import core, pipeline
from norlan.brain import BrainError


class BrainSubmissionBoundaryTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        core.create_project(self.root, '合成上下文预检测试。', 'live')
        core.install_plan(self.root, pipeline.demo_plan())
        core.configure(self.root, {'caps': {'llm': 2}}, limit=20)
        self.config = {'provider': 'openai', 'model': 'synthetic-model', 'billing': 'api'}
        self.enterContext(patch('norlan.connection.production_brain_config', side_effect=lambda _config: dict(self.config)))
        self.enterContext(patch.dict(os.environ, {'OPENAI_API_KEY': 'SYNTHETIC-LOCAL-TEST-KEY'}))
        self.shot = copy.deepcopy(core.load_project(self.root)['episodes'][0]['shots'][0])

    def api_response(self, output):
        return io.BytesIO(json.dumps({'status': 'completed', 'output': [{'type': 'message', 'content': [
            {'type': 'output_text', 'text': json.dumps(output)}]}]}).encode())

    def test_repeated_oversize_input_never_reserves_or_counts_a_model_call(self):
        self.config['max_input_chars'] = 20
        before = core.load_project(self.root)
        with patch('norlan.brain._api_open', side_effect=AssertionError('No model POST')):
            for _ in range(2):
                with self.assertRaisesRegex(BrainError, '请求尚未提交'):
                    pipeline._brain(self.root, 'plan', '完整原文不得截断' * 20)
        after = core.load_project(self.root)
        for key in ('jobs', 'llm_calls', 'budget', 'events'):
            self.assertEqual(after[key], before[key])

    def test_api_reservation_is_durable_before_transport_and_completed_after_response(self):
        output = {'prompt': '合成提示词', 'dialogue': '', 'continuity_notes': '保留切点。'}
        def submit(_request, **_kwargs):
            state = core.load_project(self.root)
            self.assertEqual(state['llm_calls'], 1)
            self.assertEqual(state['budget']['reserved'], 2)
            job = list(state['jobs'].values())[-1]
            self.assertEqual((job['state'], job['accounting']), ('submitting', 'reserved'))
            return self.api_response(output)
        with patch('norlan.brain._api_open', side_effect=submit) as transport:
            self.assertEqual(pipeline._brain(self.root, 'refine_shot', self.shot, {}, []), output)
        self.assertEqual(transport.call_count, 1)
        job = list(core.load_project(self.root)['jobs'].values())[-1]
        self.assertEqual((job['state'], job['accounting']), ('completed', 'spent'))

    def test_uncertain_submitted_api_call_blocks_a_second_model_request(self):
        with patch('norlan.brain._api_open', side_effect=URLError('SYNTHETIC-TIMEOUT')) as transport:
            with self.assertRaises(BrainError):
                pipeline._brain(self.root, 'refine_shot', self.shot, {}, [])
            first = core.load_project(self.root)
            job = list(first['jobs'].values())[-1]
            self.assertEqual((job['state'], job['accounting']), ('uncertain', 'unknown'))
            self.assertEqual(first['budget']['unknown'], 2)
            with self.assertRaisesRegex(core.GateError, '结果不明'):
                pipeline._brain(self.root, 'refine_shot', self.shot, {}, [])
            self.assertEqual(transport.call_count, 1)
            self.assertEqual(core.load_project(self.root)['jobs'], first['jobs'])

    def test_submitted_invalid_response_stays_conservatively_spent(self):
        with patch('norlan.brain._api_open', return_value=self.api_response({'wrong': 'schema'})):
            with self.assertRaises(BrainError):
                pipeline._brain(self.root, 'refine_shot', self.shot, {}, [])
        job = list(core.load_project(self.root)['jobs'].values())[-1]
        self.assertEqual((job['state'], job['accounting']), ('failed', 'spent'))

    def test_budget_refusal_stops_before_any_model_post(self):
        core.configure(self.root, limit=1)
        before = core.load_project(self.root)
        with patch('norlan.brain._api_open', side_effect=AssertionError('No model POST')):
            with self.assertRaisesRegex(core.GateError, '预算不足'):
                pipeline._brain(self.root, 'refine_shot', self.shot, {}, [])
        after = core.load_project(self.root)
        self.assertEqual(after['jobs'], before['jobs'])
        self.assertEqual(after['llm_calls'], before['llm_calls'])

    def test_local_no_frame_review_does_not_consume_a_model_call(self):
        with patch('norlan.brain._api_open', side_effect=AssertionError('No model POST')):
            result = pipeline._brain(self.root, 'review', {}, [])
        self.assertEqual(result['verdict'], 'uncertain')
        self.assertEqual(core.load_project(self.root)['jobs'], {})
        self.assertEqual(core.load_project(self.root)['llm_calls'], 0)

    def test_codex_reservation_precedes_actual_cli_request(self):
        self.config = {'provider': 'codex', 'billing': 'subscription', 'model': 'synthetic-model'}
        output = {'prompt': '合成提示词', 'dialogue': '', 'continuity_notes': ''}
        def invoke(args, **_kwargs):
            state = core.load_project(self.root)
            self.assertEqual(state['llm_calls'], 1)
            self.assertEqual(list(state['jobs'].values())[-1]['state'], 'submitting')
            Path(args[args.index('--output-last-message') + 1]).write_text(json.dumps(output))
            return subprocess.CompletedProcess(args, 0, '', '')
        with patch('norlan.brain.resolve_codex', return_value='/synthetic/codex'), \
                patch('norlan.brain.subprocess.run', side_effect=invoke):
            self.assertEqual(pipeline._brain(self.root, 'refine_shot', self.shot, {}, []), output)
        self.assertEqual(core.load_project(self.root)['llm_calls'], 1)

    def test_codex_stream_loss_cannot_repeat_an_uncertain_model_request(self):
        self.config = {'provider': 'codex', 'billing': 'subscription', 'model': 'synthetic-model'}
        failed = subprocess.CompletedProcess(['/synthetic/codex'], 1, '', 'stream disconnected before completion')
        with patch('norlan.brain.resolve_codex', return_value='/synthetic/codex'), \
                patch('norlan.brain.subprocess.run', return_value=failed) as transport:
            with self.assertRaises(BrainError):
                pipeline._brain(self.root, 'refine_shot', self.shot, {}, [])
            state = core.load_project(self.root)
            job = list(state['jobs'].values())[-1]
            self.assertEqual((job['state'], job['accounting']), ('uncertain', 'unknown'))
            with self.assertRaisesRegex(core.GateError, '结果不明'):
                pipeline._brain(self.root, 'refine_shot', self.shot, {}, [])
            self.assertEqual(transport.call_count, 1)


if __name__ == '__main__':
    unittest.main()
