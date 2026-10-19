"""Legacy in-flight task recovery with new asset inputs; synthetic, no API."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from norlan import asset_inputs, core, pipeline


def write(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return path


class LegacyAssetRecoveryTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        core.create_project(self.root, 'synthetic legacy recovery fixture', mode='live')
        core.install_plan(self.root, pipeline.demo_plan())
        core.approve(self.root, 'plan', 'plan', 1)
        image = write(self.root / 'media/hero.png', b'SYNTHETIC APPROVED IMAGE')
        with core.transaction(self.root) as state:
            hero = core._find(state, 'asset', 'hero')
            hero.update(status='candidate', version=1, file='media/hero.png', sha256=core.digest(image))
            state['assets'].append({'id':'voice', 'kind':'voice', 'name':'legacy voice',
                'prompt':'stable voice', 'character_id':'hero', 'status':'generating',
                'version':1, 'history':[], 'feedback':'', 'job_id':'old-voice'})
            state['assets'].append({'id':'confirmed-new', 'kind':'prop', 'name':'new prop',
                'prompt':'synthetic prop', 'status':'planned', 'version':0, 'history':[], 'feedback':'',
                'creation_input':{**asset_inputs.empty_input(), 'mode':'gpt', 'confirmed':True, 'revision':1}})
            # A prior human revision must also wait; the gate cannot create a new POST.
            state['assets'].append({'id':'revision', 'kind':'prop', 'name':'revised prop',
                'prompt':'synthetic prop', 'status':'rejected', 'version':1, 'history':[], 'feedback':'change color'})
        core.approve(self.root, 'asset', 'hero', 1)
        core.configure(self.root, {'caps':{'video_second':1}}, limit=50)
        core.reserve(self.root, 'old-voice', 'voice', 'voice', 5)
        core.update_job(self.root, 'old-voice', state='submitted', task_id='synthetic-task-id')
        self.provider = Mock()
        self.provider.poll_video.return_value = {'status':'completed', 'url':'https://example.test/synthetic.mp4'}
        self.provider.download.side_effect = lambda _url, target: write(Path(target), b'SYNTHETIC VIDEO')
        self.enterContext(patch.object(pipeline, '_ark', return_value=self.provider))
        self.brain = self.enterContext(patch.object(pipeline, '_brain', side_effect=AssertionError('No GPT calls')))
        self.enterContext(patch.object(pipeline.media, 'extract_evidence', side_effect=self.evidence))

    @staticmethod
    def evidence(video, directory):
        frame = write(Path(directory) / 'frame.png', b'SYNTHETIC FRAME')
        audio = write(Path(directory) / 'audio.wav', b'SYNTHETIC AUDIO')
        return {'frames':[str(frame)], 'tail_frame':str(frame), 'audio':str(audio),
                'metadata':{'duration':5, 'has_audio':True}, 'sample_times':[0]}

    def test_existing_voice_recovers_before_gate_and_frees_inputs_without_new_post(self):
        with self.assertRaisesRegex(core.GateError, '在途'):
            core.set_asset_input(self.root, 'station', 0, 0, 'gpt', '')
        with patch.object(core, 'reserve', wraps=core.reserve) as reserve:
            state = pipeline.run(self.root)
        self.assertEqual(state['status'], 'awaiting_asset_input')
        self.assertEqual(set(state['jobs']), {'old-voice'})
        self.assertEqual(state['jobs']['old-voice']['accounting'], 'spent')
        self.assertEqual(state['budget']['reserved'], 0)
        self.assertEqual(core._find(state, 'asset', 'voice')['status'], 'candidate')
        self.assertEqual(core._find(state, 'asset', 'station')['version'], 0)
        self.assertEqual(core._find(state, 'asset', 'confirmed-new')['version'], 0)
        self.assertEqual(core._find(state, 'asset', 'revision')['version'], 1)
        self.provider.poll_video.assert_called_once_with('synthetic-task-id')
        self.provider.download.assert_called_once()
        self.provider.submit_video.assert_not_called()
        self.provider.generate_image.assert_not_called()
        self.provider.preflight.assert_not_called()
        self.brain.assert_not_called()
        reserve.assert_not_called()
        state = core.set_asset_input(self.root, 'station', 0, 0, 'gpt', '')
        self.assertTrue(asset_inputs.confirmed(core._find(state, 'asset', 'station')))

    def test_unknown_original_submission_still_requires_reconciliation(self):
        core.update_job(self.root, 'old-voice', state='uncertain', accounting='unknown')
        state = pipeline.run(self.root)
        self.assertEqual(state['status'], 'blocked')
        self.assertIn('结果不明', state['last_error'])
        self.assertEqual(state['jobs']['old-voice']['accounting'], 'unknown')
        self.provider.poll_video.assert_not_called()
        self.provider.submit_video.assert_not_called()
        self.brain.assert_not_called()

    def test_recovering_voice_still_requires_approved_bound_character(self):
        with core.transaction(self.root) as state:
            core._find(state, 'asset', 'hero')['status'] = 'candidate'
        state = pipeline.run(self.root)
        self.assertEqual(state['status'], 'blocked')
        self.assertIn('已批准的可见角色', state['last_error'])
        self.provider.poll_video.assert_not_called()
        self.provider.submit_video.assert_not_called()
        core.approve(self.root, 'asset', 'hero', 1)
        self.assertEqual(pipeline.run(self.root)['status'], 'awaiting_asset_input')
        self.provider.poll_video.assert_called_once_with('synthetic-task-id')


if __name__ == '__main__':
    unittest.main()
