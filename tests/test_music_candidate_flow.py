"""Real reservation/candidate state transitions, synthetic music provider only."""
import copy
from pathlib import Path
import shutil
import unittest
from unittest.mock import Mock, patch

from norlan import core, episode_post, pipeline
from tests import test_episode_post as fixtures


@unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "requires local FFmpeg")
class MusicCandidateFlowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Reuse actual, explicitly synthetic media fixtures without inheriting
        # or re-running the separate lifecycle test methods.
        fixtures.EpisodePostTests.setUpClass()

    @classmethod
    def tearDownClass(cls):
        fixtures.EpisodePostTests.tearDownClass()

    def setUp(self):
        self.case = fixtures.EpisodePostTests("test_restart_during_music_wait_reuses_proofreading_and_source")
        self.case.setUp()
        self.addCleanup(self.case.doCleanups)
        self.root = self.case.root
        core.configure(self.root, {"music": {"mode": "volc"},
                                  "poll_seconds": 1})
        with core.transaction(self.root) as state:
            state["mode"] = "live"
        self.provider = Mock()
        self.provider.submit.return_value = {"id": "synthetic-song-task", "status": "pending"}
        self.provider.poll.return_value = {"status": "completed", "url": "https://example.invalid/synthetic-test-music"}
        self.provider.download.side_effect = lambda url, destination: shutil.copyfile(
            self.case.fixture / "synthetic-music.wav", destination)
        self.enterContext(patch("norlan.music.MusicProvider", return_value=self.provider))

    def finish(self):
        return episode_post.finish(self.root, self.case.episode(), brain_call=self.case.brain,
                                   gate=pipeline._gate, soundtrack=pipeline._soundtrack)

    def test_one_explicit_generation_waits_for_audition_without_repeat_on_resume(self):
        self.case.import_music(approve=True)
        original_shots = copy.deepcopy(self.case.shots())
        ep = self.case.episode()
        episode_post.request_generation(self.root, ep["id"], ep["version"])
        with self.assertRaisesRegex(core.GateError, "已有音乐生成请求"):
            episode_post.request_generation(self.root, ep["id"], ep["version"])
        self.assertFalse(self.finish())
        first = core.load_project(self.root)
        post = self.case.episode()["postproduction"]
        self.assertFalse(post["music_generation_requested"])
        self.assertEqual(post["status"], "awaiting_music_approval")
        candidates = [t for t in first["music_library"]["tracks"] if t["status"] == "awaiting_approval"]
        self.assertEqual(len(candidates), 1)
        candidate = candidates[0]
        self.assertTrue(core.safe_path(self.root, candidate["file"]).is_file())
        jobs = [job for job in first["jobs"].values() if job["kind"] == "music"]
        self.assertEqual(len(jobs), 1)
        self.assertEqual(jobs[0]["task_id"], "synthetic-song-task")
        self.assertEqual((jobs[0]["state"], jobs[0]["accounting"]), ("completed", "spent"))
        self.assertTrue(jobs[0]["cost_track"])
        self.assertEqual(first["budget"]["cost_pending_jobs"], 1)
        self.assertIsNone(first["budget"]["actual_cost"], "missing public price must not become a zero-cost bill")
        self.assertEqual(self.provider.submit.call_args.args[1], 30)

        # A healthy restart reuses local ASR/GPT proofreading and the completed
        # candidate. An existing approved theme cannot bypass this new audition.
        self.assertFalse(self.finish())
        self.assertFalse(self.finish())
        self.assertEqual(len(core.load_project(self.root)["music_library"]["tracks"]), 2)
        self.assertEqual(len(self.case.calls("proofread_subtitles")), 1)
        self.assertEqual(len(self.case.calls("arrange_music")), 0)
        self.provider.submit.assert_called_once()
        self.provider.poll.assert_called_once()
        self.provider.download.assert_called_once()
        self.assertEqual(self.case.shots(), original_shots)

        # Only the exact human-approved candidate can enter the next mix.
        episode_post.decide_music(self.root, candidate["id"], candidate["version"], candidate["sha256"])
        self.assertTrue(self.finish())
        final = self.case.episode()
        self.assertEqual(final["status"], "awaiting_review")
        self.assertEqual(final["postproduction"]["status"], "awaiting_final_approval")
        self.assertTrue(final["subtitles_burned"])
        self.assertEqual(len(self.case.calls("proofread_subtitles")), 1)
        self.assertEqual(len(self.case.calls("arrange_music")), 1)
        self.provider.submit.assert_called_once()
        self.assertEqual(core.load_project(self.root)["budget"]["cost_pending_jobs"], 1)
        self.assertEqual(self.case.shots(), original_shots)

    def test_130_second_episode_requests_one_60_second_candidate_without_a_budget(self):
        ep = self.case.episode()
        with patch.object(pipeline.media, "probe", return_value={"duration": 65}):
            output = pipeline._soundtrack(self.root, {**ep, "post_music_candidate": True})
        self.assertTrue(Path(output).is_file())
        self.provider.submit.assert_called_once()
        self.assertEqual(self.provider.submit.call_args.args[1], 60)
        state = core.load_project(self.root)
        jobs = [job for job in state["jobs"].values() if job["kind"] == "music"]
        self.assertEqual(len(jobs), 1)
        self.assertEqual(jobs[0]["amount"], 0)
        self.assertTrue(jobs[0]["cost_track"])
        self.assertEqual(state["budget"]["cost_pending_jobs"], 1)
        self.assertIsNone(state["budget"]["actual_cost"], "zero estimate means unknown cost, not free")
        # Once the candidate has been downloaded, resuming the original task
        # reuses that file; a long episode is scored later with explicit cues.
        self.assertEqual(pipeline._soundtrack(self.root, {**self.case.episode(), "post_music_candidate": True}), output)
        self.provider.submit.assert_called_once()
        self.provider.poll.assert_called_once()
        self.provider.download.assert_called_once()


if __name__ == "__main__":
    unittest.main()
