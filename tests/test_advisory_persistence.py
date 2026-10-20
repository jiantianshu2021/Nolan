"""Synthetic GPT review persistence; no real model or media service calls."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from norlan import core, pipeline


class AdvisoryPersistenceTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        core.create_project(self.root, "合成缓存恢复测试。")
        core.set_video_review_enabled(self.root, True)
        core.install_plan(self.root, pipeline.demo_plan())
        core.configure(self.root, {"caps": {"llm": 2}}, limit=100)
        path = self.root / "media/E001/E001-S001-v1.mp4"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"SYNTHETIC COMPLETED VIDEO")
        core.reserve(self.root, "synthetic-video", "video", "E001-S001", 4)
        core.update_job(self.root, "synthetic-video", state="completed", accounting="spent", output=str(path.relative_to(self.root)))
        with core.transaction(self.root) as state:
            state["mode"] = "live"
            shot = state["episodes"][0]["shots"][0]
            shot.update(version=1, status="reviewing", job_id="synthetic-video", file=str(path.relative_to(self.root)), sha256=core.digest(path))
        self.shot = core.load_project(self.root)["episodes"][0]["shots"][0]
        self.target = pipeline._shot_review_target(self.shot)
        self.enterContext(patch("norlan.connection.production_brain_config", return_value={"provider": "codex", "billing": "subscription"}))
        self.brain = self.enterContext(patch("norlan.brain.Brain"))
        self.ark = self.enterContext(patch.object(pipeline, "_ark", side_effect=AssertionError("no media calls during recovery")))

    def test_target_is_durable_before_call_and_result_is_saved_with_completion(self):
        report = {"verdict": "fail", "issues": ["合成画面需人工判断。"], "observed_end_state": "合成尾态",
                  "repair_prompt": "仅供参考", "scope": "sampled_frames_only", "audio_assessed": False}

        def answer(*args):
            before = core.load_project(self.root)
            self.assertFalse(any(job["kind"] == "llm" for job in before["jobs"].values()))
            self.assertEqual(before["llm_calls"], 0)
            # Simulate the transport boundary after local request preparation.
            self.brain.return_value.before_submit()
            job = list(core.load_project(self.root)["jobs"].values())[-1]
            self.assertEqual(job["state"], "submitting")
            self.assertEqual(job["purpose"], "advisory_review")
            self.assertEqual(job["advisory_target"], self.target)
            self.assertNotIn("review_result", job)
            self.assertEqual(core.load_project(self.root)["llm_calls"], 1)
            return report

        self.brain.return_value.review.side_effect = answer
        self.assertEqual(pipeline._brain(self.root, "review", {"advisory_target": self.target}, []), report)
        job = list(core.load_project(self.root)["jobs"].values())[-1]
        self.assertEqual((job["state"], job["accounting"]), ("completed", "spent"))
        self.assertEqual(job["review_result"], report)
        self.assertEqual(job["review_result_hash"], core._json_digest(report))
        state = pipeline.run(self.root)
        self.assertEqual(state["status"], "review_required", state.get("last_error"))
        self.assertTrue(state["episodes"][0]["shots"][0]["review"]["recovered_from_cache"])
        self.brain.return_value.review.assert_called_once()
        self.ark.assert_not_called()

    def test_process_interruption_keeps_unknown_budget_and_does_not_repeat_review(self):
        def interrupted_after_submit(*args):
            self.brain.return_value.before_submit()
            raise KeyboardInterrupt("synthetic process interruption after request began")

        self.brain.return_value.review.side_effect = interrupted_after_submit
        with self.assertRaises(KeyboardInterrupt):
            pipeline._brain(self.root, "review", {"advisory_target": self.target}, [])
        original = list(core.load_project(self.root)["jobs"].values())[-1]
        self.assertEqual(original["state"], "submitting")
        state = pipeline.run(self.root)
        self.assertEqual(state["status"], "review_required", state.get("last_error"))
        self.assertEqual((state["jobs"][original["id"]]["accounting"], state["budget"]["unknown"]), ("unknown", 2))
        self.assertTrue(state["episodes"][0]["shots"][0]["review"]["report_unavailable"])
        self.brain.return_value.review.assert_called_once()
        self.ark.assert_not_called()


if __name__ == "__main__":
    unittest.main()
