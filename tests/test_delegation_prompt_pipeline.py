"""Synthetic regression for saved delegation -> one final candidate, never media."""
import copy
import hashlib
import json
import unittest
from unittest.mock import patch

from norlan import core, pipeline
import test_prompt_gate as fixtures


class DelegationPromptPipelineTests(unittest.TestCase):
    setUp = fixtures.PromptGateTests.setUp
    shot = fixtures.PromptGateTests.shot
    confirm = fixtures.PromptGateTests.confirm

    def answer(self, root, method, *args, **kwargs):
        if method == 'plan_revision':
            self.assertIn('交给gpt来准备提示词', args[-1].lower())
            return {'needs_assets': False, 'additions': [], 'shot_updates': [],
                    'summary': '在原镜头结构内先准备草稿，不要求亲自填写秒点。', 'blockers': []}
        if method == 'refine_shot':
            current = self.shot()
            return {'prompt': '合成新草稿：主角先问，女声从1.5秒起跨5秒切镜连续播读。',
                    'dialogue': current['dialogue'], 'continuity_notes': '拟定声音/UI安排，尚未生成。',
                    'timing_proposal': {'summary': '保留15秒及5秒摄影切点，播报1.5秒开始。',
                        'subshots': [{'start': 0, 'end': 5, 'prompt': '原动作和主角台词；1.5秒出现规则界面，女声开始。'},
                                     {'start': 5, 'end': 15, 'prompt': '原第二镜动作；界面保持，女声跨切点读完。'}],
                        'audio_windows': [{'start': 0, 'end': 1.3, 'speaker': '主角', 'text': '要我喂你？'},
                                          {'start': 1.5, 'end': 14.8, 'speaker': '系统', 'text': self.text}]}}
        return {'verdict': 'uncertain', 'issues': [], 'observed_end_state': '合成', 'repair_prompt': ''}

    def configure_block(self):
        core.configure(self.root, {'require_voice_reference': False})  # Synthetic test isolates timing consent.
        with core.transaction(self.root) as state:
            shot = state['episodes'][0]['shots'][0]
            shot.update(duration=15, dialogue='主角：要我喂你？\n系统：' + self.text,
                        prompt='OLD_CANONICAL_UI_STARTS_AT_FIVE', feedback='无意见保持原来的就行', revision=0,
                        subshots=[{'start': 0, 'end': 5, 'prompt': '旧第一镜动作和台词。'},
                                  {'start': 5, 'end': 15, 'prompt': 'OLD_CANONICAL_UI_STARTS_AT_FIVE'}])
            signature = hashlib.sha256(json.dumps([0, shot['feedback']], ensure_ascii=False).encode()).hexdigest()
            shot['revision_analysis'] = {'signature': signature, 'blockers': ['旧规则要求亲自填写秒点。'], 'summary': '合成旧阻塞。'}
            state.update(status='blocked', last_error='旧规则要求亲自填写秒点。')
        current = self.shot()
        core.submit_planning_feedback(self.root, current['id'], '交给gpt来准备提示词', current.get('version', 0), signature)
        return copy.deepcopy(self.shot()['subshots'])

    def test_saved_delegation_reaches_final_prompt_once_and_continue_cannot_reopen_the_question(self):
        original = self.configure_block()
        state = pipeline.run(self.root)
        self.assertEqual(state['status'], 'awaiting_prompt_approval', state.get('last_error'))
        candidate = copy.deepcopy(self.shot()['prompt_candidate'])
        self.assertIsNotNone(candidate['timing_proposal'])
        self.assertNotIn('OLD_CANONICAL_UI_STARTS_AT_FIVE', candidate['prompt'])
        self.assertIn('1.5–14.8', candidate['prompt'])
        self.assertEqual(self.shot()['subshots'], original)
        self.assertNotIn('prompt_approval', self.shot())
        calls = self.brain.call_count
        for _ in range(3):
            self.assertEqual(pipeline.run(self.root)['status'], 'awaiting_prompt_approval')
        self.assertEqual(self.brain.call_count, calls)
        self.assertEqual(self.shot()['prompt_candidate'], candidate)
        self.api.submit_video.assert_not_called()

    def test_only_confirmed_exact_draft_can_reach_the_media_request_boundary(self):
        original = self.configure_block()
        pipeline.run(self.root)
        candidate = copy.deepcopy(self.shot()['prompt_candidate'])
        self.confirm()
        with patch.object(pipeline, '_video', side_effect=core.GateError('SYNTHETIC: stopped before any media POST')) as media:
            pipeline.run(self.root)
        media.assert_called_once()
        self.assertEqual(media.call_args.args[5], candidate['prompt'])
        self.assertEqual(self.shot()['subshots'], original)
        self.api.submit_video.assert_not_called()

    def test_review_uses_the_confirmed_candidate_schedule_without_mutating_the_plan(self):
        original = self.configure_block()
        pipeline.run(self.root)
        current = self.shot()
        current['refined_prompt'] = copy.deepcopy(current['prompt_candidate']['refined'])
        review = pipeline._review_shot_summary(current)
        self.assertEqual(review['subshots'], current['prompt_candidate']['timing_proposal']['subshots'])
        self.assertEqual(review['prompt'], current['refined_prompt']['prompt'])
        self.assertNotIn('OLD_CANONICAL_UI_STARTS_AT_FIVE', review['prompt'])
        self.assertNotEqual(review['subshots'], original)
        self.assertEqual(current['subshots'], original)


if __name__ == '__main__':
    unittest.main()
