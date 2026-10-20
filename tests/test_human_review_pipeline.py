"""Offline regression tests: GPT reports never authorize paid video work."""
import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from norlan import core, pipeline
from norlan.providers import ProviderError
from test_pipeline import run_confirming_prompts, choose_gpt_assets, fake_media, run_with_shot_approvals, write


class HumanReviewPipelineTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.enterContext(fake_media())
        core.create_project(self.root, "合成人审测试，不调用真实服务。")
        core.set_video_review_enabled(self.root, True)
        core.install_plan(self.root, pipeline.demo_plan())
        core.approve(self.root, "plan", "plan", 1)
        choose_gpt_assets(self.root)
        state = run_confirming_prompts(self.root)
        for asset in state["assets"]:
            core.approve(self.root, "asset", asset["id"], asset["version"])
        core.configure(self.root, {"caps": {"video_second": 1}}, limit=100)
        with core.transaction(self.root) as state:
            state["mode"] = "live"
        self.verdict = "fail"
        self.api = Mock()
        self.api.submit_video.side_effect = lambda *a, **kw: {"id": "synthetic-" + str(self.api.submit_video.call_count)}
        self.api.poll_video.return_value = {"status": "completed", "url": "https://example.test/synthetic.mp4",
                                            "last_frame_url": "https://example.test/synthetic-tail.jpg"}
        self.api.download.side_effect = lambda url, output: write(output,
            b"\xff\xd8\xffSYNTHETIC OFFICIAL TAIL" if url.endswith(".jpg") else b"SYNTHETIC " + str(output).encode())
        self.enterContext(patch.object(pipeline, "_ark", return_value=self.api))
        self.brain = self.enterContext(patch.object(pipeline, "_brain", side_effect=self.fake_brain))

    def fake_brain(self, root, method, *args):
        if method == "plan_revision":
            return {"needs_assets": False, "additions": [], "shot_updates": [],
                    "summary": "合成普通返修", "blockers": []}
        if method == "refine_shot":
            return {"prompt": "SYNTHETIC", "dialogue": "", "continuity_notes": "synthetic"}
        return {"verdict": self.verdict, "issues": ["[细节] 合成测试：纸角位置"] if self.verdict == "fail" else [],
                "observed_end_state": "合成测试：人物在左侧", "repair_prompt": "仅供人选择的修改建议"}

    def shot(self, index=0):
        return core.load_project(self.root)["episodes"][0]["shots"][index]

    def test_fail_waits_and_resuming_never_repeats_video_or_report(self):
        state = run_confirming_prompts(self.root)
        self.assertEqual(state["status"], "review_required")
        shot = self.shot()
        self.assertEqual(shot["version"], 1)
        self.assertEqual(shot["review"]["sha256"], shot["sha256"])
        self.assertEqual(shot["review"]["version"], 1)
        self.assertTrue(shot["review"]["advisory"])
        before = copy.deepcopy(state["jobs"])
        calls = self.brain.call_count
        for _ in range(3):
            self.assertEqual(run_confirming_prompts(self.root)["status"], "review_required")
        self.api.submit_video.assert_called_once()
        self.assertEqual(self.brain.call_count, calls)
        self.assertEqual(core.load_project(self.root)["jobs"], before)
        self.assertEqual(self.shot(1).get("version", 0), 0)

    def test_pass_also_requires_human_decision(self):
        self.verdict = "pass"
        state = run_confirming_prompts(self.root)
        self.assertEqual(state["status"], "review_required")
        self.assertEqual(self.shot()["status"], "review_required")
        self.assertEqual(self.shot(1).get("version", 0), 0)
        self.api.submit_video.assert_called_once()

    def test_uncertain_also_requires_human_decision(self):
        self.verdict = "uncertain"
        self.assertEqual(run_confirming_prompts(self.root)["status"], "review_required")
        self.api.submit_video.assert_called_once()

    def test_approve_failed_report_accepts_actual_state_without_regenerating_it(self):
        run_confirming_prompts(self.root)
        original = self.shot()
        core.approve(self.root, "shot", original["id"], 1)
        run_confirming_prompts(self.root)
        first, second = self.shot(), self.shot(1)
        self.assertEqual(first["sha256"], original["sha256"])
        self.assertEqual(first["version"], 1)
        self.assertEqual(second["status"], "review_required")
        previous = second["reference_manifest"]["context"]["previous_actual"]
        self.assertEqual(previous["human_decision"]["decision"], "approved")
        self.assertEqual(previous["accepted_deviations"], original["review"]["issues"])
        self.assertEqual(self.api.submit_video.call_count, 2)

    def test_reject_authorizes_exactly_one_version_and_preserves_history(self):
        run_confirming_prompts(self.root)
        original = self.shot()
        core.reject(self.root, "shot", original["id"], 1, "仅修改人物朝向，纸角保留")
        self.verdict = "pass"
        run_confirming_prompts(self.root)
        run_confirming_prompts(self.root)
        shot = self.shot()
        self.assertEqual(shot["version"], 2)
        self.assertEqual(shot["status"], "review_required")
        self.assertEqual(shot["history"][0]["sha256"], original["sha256"])
        self.assertEqual(self.api.submit_video.call_count, 2)
        self.assertEqual(self.shot(1).get("version", 0), 0)

    def test_report_failure_keeps_candidate_and_does_not_repeat_paid_work(self):
        def broken_report(root, method, *args):
            if method == "review":
                raise RuntimeError("synthetic report outage")
            return self.fake_brain(root, method, *args)
        self.brain.side_effect = broken_report
        state = run_confirming_prompts(self.root)
        self.assertEqual(state["status"], "review_required")
        self.assertTrue(self.shot()["review"]["report_unavailable"])
        self.assertTrue((self.root / self.shot()["file"]).is_file())
        run_confirming_prompts(self.root)
        self.api.submit_video.assert_called_once()
        self.assertEqual(self.brain.call_count, 2)
        core.approve(self.root, "shot", "E001-S001", 1)
        self.assertEqual(self.shot()["status"], "accepted")

    def _legacy_retry_fixture(self, *, stored_version=True):
        run_confirming_prompts(self.root)
        original = self.shot()
        with core.transaction(self.root) as state:
            shot = state["episodes"][0]["shots"][0]
            shot.setdefault("history", []).append({key: copy.deepcopy(shot[key]) for key in
                                    ("file", "sha256", "version", "evidence", "review")})
            if not stored_version:
                shot["history"][-1].pop("version")
            shot.update(status="planned", feedback="旧版自动修复意见")
            for key in ("file", "sha256", "evidence", "review", "refined_prompt", "job_id"):
                shot.pop(key, None)
            state["paused"] = True
        self.assertEqual(run_confirming_prompts(self.root)["status"], "paused")
        restored = self.shot()
        self.assertEqual(restored["status"], "review_required")
        self.assertEqual(restored["sha256"], original["sha256"])
        self.assertEqual(restored["feedback"], "")
        core.set_paused(self.root, False)
        self.assertEqual(run_confirming_prompts(self.root)["status"], "review_required")
        self.api.submit_video.assert_called_once()

    def test_legacy_automatic_retry_is_restored_for_review_even_while_paused(self):
        self._legacy_retry_fixture()

    def test_legacy_archive_without_version_uses_exact_versioned_filename(self):
        self._legacy_retry_fixture(stored_version=False)

    def test_registered_download_recovers_same_job_then_waits(self):
        self.api.download.side_effect = ProviderError("synthetic network failure")
        self.assertEqual(run_confirming_prompts(self.root)["status"], "blocked")
        original_job = self.shot()["job_id"]
        self.api.download.side_effect = lambda url, output: write(output,
            b"\xff\xd8\xffSYNTHETIC OFFICIAL TAIL" if url.endswith(".jpg") else b"SYNTHETIC RECOVERED")
        self.assertEqual(run_confirming_prompts(self.root)["status"], "review_required")
        self.assertEqual(self.shot()["job_id"], original_job)
        self.assertEqual(self.shot()["version"], 1)
        self.api.submit_video.assert_called_once()

    def test_legacy_inflight_version_does_not_expose_or_approve_older_file(self):
        run_confirming_prompts(self.root)
        original = self.shot()
        core.reject(self.root, "shot", original["id"], 1, "用户要求再做一版")
        self.api.download.side_effect = ProviderError("synthetic download failure")
        run_confirming_prompts(self.root)
        with core.transaction(self.root) as state:
            shot = state["episodes"][0]["shots"][0]
            for key in ("file", "sha256", "evidence", "review"):
                shot[key] = copy.deepcopy(original[key])
            state["paused"] = True
        self.assertEqual(run_confirming_prompts(self.root)["status"], "paused")
        self.assertNotIn("file", self.shot())
        self.assertEqual(self.shot()["version"], 2)
        self.assertEqual(self.shot()["history"][0]["sha256"], original["sha256"])
        self.api.download.side_effect = lambda url, output: write(output,
            b"\xff\xd8\xffSYNTHETIC OFFICIAL TAIL" if url.endswith(".jpg") else b"SYNTHETIC V2 RECOVERY")
        core.set_paused(self.root, False)
        state = run_confirming_prompts(self.root)
        self.assertEqual(state["status"], "review_required")
        self.assertEqual(self.shot()["version"], 2)
        self.assertEqual(self.shot()["review"]["version"], 2)
        self.assertEqual(self.api.submit_video.call_count, 2)

    def test_old_gpt_accepted_candidate_needs_human_but_no_new_generation(self):
        self.verdict = "pass"
        run_confirming_prompts(self.root)
        with core.transaction(self.root) as state:
            state["episodes"][0]["shots"][0]["status"] = "accepted"
        self.assertEqual(run_confirming_prompts(self.root)["status"], "review_required")
        self.api.submit_video.assert_called_once()

    def test_batch_report_cannot_revoke_human_decisions_or_replace_shot_reports(self):
        state = run_with_shot_approvals(self.root)
        self.assertEqual(state["status"], "awaiting_episode_review")
        episode = state["episodes"][0]
        self.assertEqual(episode["reviews"][0]["verdict"], "fail")
        for shot in episode["shots"]:
            self.assertEqual(shot["status"], "accepted")
            self.assertTrue(core.human_approved_shot(shot))
            self.assertEqual(shot["review"]["sha256"], shot["sha256"])
            self.assertNotIn("key", shot["review"])
        self.assertEqual(self.api.submit_video.call_count, 2)
