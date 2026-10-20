"""Synthetic live video transitions; no actual media or GPT providers."""
import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from norlan import core, dashboard, pipeline
import test_prompt_gate as prompt_fixtures
from test_pipeline import choose_gpt_assets, fake_media, run_with_shot_approvals


class VideoReviewPipelineTests(unittest.TestCase):
    def setUp(self):
        # Reuse the existing fake provider/transport and real human prompt gate.
        self.fixture = prompt_fixtures.PromptGateTests("runTest")
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.root = self.fixture.root

    def generate(self):
        self.assertEqual(pipeline.run(self.root)["status"], "awaiting_prompt_approval")
        self.fixture.confirm()
        return pipeline.run(self.root)

    def review_calls(self):
        return [call for call in self.fixture.brain.call_args_list if call.args[1] == "review"]

    def test_default_off_publishes_candidate_and_local_evidence_without_gpt_review(self):
        state = self.generate()
        shot = self.fixture.shot()
        self.assertEqual(state["status"], "review_required")
        self.assertFalse(core.human_approved_shot(shot))
        self.assertTrue(shot["evidence"]["frames"])
        self.assertTrue(shot["evidence"]["audio"])
        self.assertNotIn("review", shot)
        self.assertEqual(shot["review_skipped"]["sha256"], shot["sha256"])
        self.assertEqual(shot["review_skipped"]["version"], shot["version"])
        self.assertEqual(self.review_calls(), [])
        jobs = copy.deepcopy(state["jobs"])
        for _ in range(2):
            resumed = pipeline.run(self.root)
            self.assertEqual(resumed["status"], "review_required")
            self.assertEqual(resumed["jobs"], jobs)
        self.fixture.api.submit_video.assert_called_once()

    def test_enabled_review_runs_then_switching_off_keeps_its_report(self):
        core.set_video_review_enabled(self.root, True)
        self.generate()
        report = copy.deepcopy(self.fixture.shot()["review"])
        self.assertEqual(len(self.review_calls()), 1)
        self.assertNotIn("review_skipped", self.fixture.shot())
        core.set_video_review_enabled(self.root, False)
        pipeline.run(self.root)
        self.assertEqual(self.fixture.shot()["review"], report)
        self.assertEqual(len(self.review_calls()), 1)

    def test_disabling_during_local_preview_stops_the_next_review(self):
        core.set_video_review_enabled(self.root, True)
        real_preview = pipeline._refresh_candidate_preview

        def preview(root, episode):
            result = real_preview(root, episode)
            core.set_video_review_enabled(root, False)
            return result

        with patch.object(pipeline, "_refresh_candidate_preview", side_effect=preview):
            self.generate()
        self.assertEqual(self.review_calls(), [])
        self.assertEqual(self.fixture.shot()["review_skipped"]["reason"], "disabled")

    def test_already_started_review_can_finish_after_disabling(self):
        core.set_video_review_enabled(self.root, True)

        def answer(root, method, *args, **kwargs):
            if method == "review":
                core.set_video_review_enabled(root, False)
            return self.fixture.answer(root, method, *args, **kwargs)

        self.fixture.brain.side_effect = answer
        self.generate()
        self.assertEqual(len(self.review_calls()), 1)
        self.assertEqual(self.fixture.shot()["review"]["verdict"], "pass")
        self.assertFalse(core.video_review_enabled(core.load_project(self.root)))

    def test_enabling_does_not_backfill_a_skipped_version(self):
        self.generate()
        original = copy.deepcopy(self.fixture.shot())
        core.set_video_review_enabled(self.root, True)
        self.assertEqual(pipeline.run(self.root)["status"], "review_required")
        self.assertEqual(self.fixture.shot(), original)
        self.assertEqual(self.review_calls(), [])
        self.fixture.api.submit_video.assert_called_once()

    def test_restart_before_skip_returns_to_human_without_failed_report(self):
        self.generate()
        with core.transaction(self.root) as state:
            shot = state["episodes"][0]["shots"][0]
            shot["status"] = "reviewing"
            shot.pop("review_skipped")
        resumed = pipeline.run(self.root)
        shot = self.fixture.shot()
        self.assertEqual(resumed["status"], "review_required")
        self.assertEqual(shot["review_skipped"]["reason"], "disabled")
        self.assertNotIn("review", shot)
        self.assertEqual(self.review_calls(), [])

    def test_public_skip_is_bound_to_current_video_version_and_hash(self):
        self.generate()
        state = core.load_project(self.root)
        public = dashboard.public_state(state)["episodes"][0]["shots"][0]
        self.assertEqual(public["review_skipped"]["sha256"], public["sha256"])
        for key, value in (("version", 999), ("sha256", "f"*64)):
            bad = copy.deepcopy(state)
            bad["episodes"][0]["shots"][0]["review_skipped"][key] = value
            self.assertNotIn("review_skipped", dashboard.public_state(bad)["episodes"][0]["shots"][0])


class VideoReviewCutTests(unittest.TestCase):
    def test_disabled_cut_review_keeps_local_stitches_and_human_episode_gate(self):
        with tempfile.TemporaryDirectory() as directory, fake_media(), patch.object(pipeline, "_brain", side_effect=AssertionError("no model calls")):
            root = Path(directory)
            core.create_project(root, "合成拼接测试。")
            core.install_plan(root, pipeline.demo_plan())
            core.approve(root, "plan", "plan", 1)
            choose_gpt_assets(root)
            state = pipeline.run(root)
            for asset in state["assets"]:
                core.approve(root, "asset", asset["id"], asset["version"])
            state = run_with_shot_approvals(root)
            self.assertEqual(state["status"], "awaiting_episode_review")
            episode = state["episodes"][0]
            self.assertEqual(episode["status"], "awaiting_review")
            self.assertTrue(episode["rough_cuts"])
            self.assertEqual(episode["reviews"], [])
            cut = episode["rough_cuts"][-1]
            self.assertEqual(cut["review_skipped"]["sha256"], cut["sha256"])
            self.assertTrue(all(core.human_approved_shot(shot) for shot in episode["shots"]))
            self.assertNotEqual(state["episodes"][1]["status"], "approved")
            core.set_video_review_enabled(root, True)
            again = pipeline.run(root)
            self.assertEqual(again["episodes"][0]["reviews"], [])
            self.assertEqual(again["status"], "awaiting_episode_review")


if __name__ == "__main__":
    unittest.main()
