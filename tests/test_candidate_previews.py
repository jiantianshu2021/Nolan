"""Synthetic candidate-preview regressions; no network or paid providers."""
import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from norlan import core, pipeline
from test_pipeline import choose_gpt_assets, fake_media, write


class CandidatePreviewTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.enterContext(fake_media())
        core.create_project(self.root, "离线候选累计预览测试，不调用真实服务。")
        plan = pipeline.demo_plan()
        plan["episodes"] = [plan["episodes"][0]]
        template = plan["episodes"][0]["shots"][0]
        plan["episodes"][0]["shots"] = [
            {**copy.deepcopy(template), "id": f"E001-S{index:03}"} for index in range(1, 5)
        ]
        core.install_plan(self.root, plan)
        core.approve(self.root, "plan", "plan", 1)
        choose_gpt_assets(self.root)
        state = pipeline.run(self.root)
        for asset in state["assets"]:
            core.approve(self.root, "asset", asset["id"], asset["version"])
        core.configure(self.root, {"caps": {"video_second": 1}, "batch_size": 30}, limit=200)
        with core.transaction(self.root) as state:
            state["mode"] = "live"
        self.api = Mock()
        self.api.submit_video.side_effect = lambda *a, **kw: {
            "id": "synthetic-preview-" + str(self.api.submit_video.call_count)}
        self.api.poll_video.return_value = {"status": "completed", "url": "https://example.test/synthetic.mp4",
                                            "last_frame_url": "https://example.test/synthetic-tail.jpg"}
        self.api.download.side_effect = lambda url, output: write(output,
            b"\xff\xd8\xffSYNTHETIC OFFICIAL TAIL" if url.endswith(".jpg") else b"SYNTHETIC " + str(output).encode())
        self.enterContext(patch.object(pipeline, "_ark", return_value=self.api))
        self.brain = self.enterContext(patch.object(pipeline, "_brain", side_effect=self.fake_brain))
        pipeline.media.compose.reset_mock()

    def fake_brain(self, root, method, *args):
        if method == "refine_shot":
            return {"prompt": "SYNTHETIC PREVIEW", "dialogue": "", "continuity_notes": "synthetic"}
        return {"verdict": "pass", "issues": [], "observed_end_state": "合成测试：人物站左侧", "repair_prompt": ""}

    def episode(self):
        return core.load_project(self.root)["episodes"][0]

    def shot(self, index):
        return self.episode()["shots"][index]

    def approve(self, index):
        shot = self.shot(index)
        core.approve(self.root, "shot", shot["id"], shot["version"])

    def generate_through(self, index):
        """Generate the indexed candidate; preceding approvals are synthetic humans."""
        for current in range(index + 1):
            state = pipeline.run(self.root)
            self.assertEqual(state["status"], "review_required", state.get("last_error"))
            self.assertEqual(self.shot(current)["status"], "review_required")
            if current < index:
                self.approve(current)
        return state

    def preview(self):
        return self.episode().get("rough_cuts", [])[-1]

    def test_second_candidate_preview_is_published_before_gpt_and_does_not_approve(self):
        self.generate_through(0)
        self.assertEqual(self.episode().get("rough_cuts", []), [])
        self.approve(0)
        inspected = []

        def inspect_report(root, method, *args):
            if method == "review":
                current = self.shot(1)
                cut = self.preview()
                self.assertEqual(current["status"], "reviewing")
                self.assertNotIn("review", current)
                self.assertEqual(current["evidence"]["sha256"], current["sha256"])
                self.assertEqual(cut["shot_count"], 2)
                self.assertEqual(cut["sha256"], core.digest(self.root / cut["file"]))
                self.assertEqual(cut["shots"][-1], {
                    "id": current["id"], "version": 1, "sha256": current["sha256"]})
                inspected.append(cut["key"])
            return self.fake_brain(root, method, *args)

        self.brain.side_effect = inspect_report
        state = pipeline.run(self.root)
        self.assertEqual(len(inspected), 1)
        self.assertEqual(state["status"], "review_required", state.get("last_error"))
        self.assertEqual(self.shot(1)["status"], "review_required")
        self.assertFalse(core.human_approved_shot(self.shot(1)))
        self.assertEqual(self.shot(2).get("version", 0), 0)
        self.assertEqual(self.api.submit_video.call_count, 2)
        self.assertEqual(pipeline.media.compose.call_count, 1)
        before_calls, before_jobs = self.brain.call_count, copy.deepcopy(state["jobs"])
        pipeline.run(self.root)
        self.assertEqual(self.brain.call_count, before_calls)
        self.assertEqual(core.load_project(self.root)["jobs"], before_jobs)
        self.assertEqual(self.api.submit_video.call_count, 2)
        self.assertEqual(pipeline.media.compose.call_count, 1)
        with self.assertRaisesRegex(core.GateError, "前镜头未被接受"):
            pipeline._shot_context(self.root, self.episode(), self.shot(2), self.shot(1))

    def test_report_failure_keeps_already_published_candidate_preview(self):
        self.generate_through(0)
        self.approve(0)

        def broken_report(root, method, *args):
            if method == "review":
                self.assertEqual(self.preview()["shot_count"], 2)
                raise RuntimeError("Synthetic report failure")
            return self.fake_brain(root, method, *args)

        self.brain.side_effect = broken_report
        state = pipeline.run(self.root)
        self.assertEqual(state["status"], "review_required", state.get("last_error"))
        self.assertTrue(self.shot(1)["review"]["report_unavailable"])
        cut = copy.deepcopy(self.preview())
        before_calls = self.brain.call_count
        pipeline.run(self.root)
        self.assertEqual(self.preview(), cut)
        self.assertEqual(self.brain.call_count, before_calls)
        self.assertEqual(self.api.submit_video.call_count, 2)
        self.assertEqual(pipeline.media.compose.call_count, 1)

    def test_local_preview_failure_preserves_candidate_and_recovery_needs_no_paid_work(self):
        self.generate_through(0)
        self.approve(0)
        with patch.object(pipeline.media, "compose", side_effect=RuntimeError("SYNTHETIC PRIVATE DETAIL")):
            state = pipeline.run(self.root)
        self.assertEqual(state["status"], "review_required", state.get("last_error"))
        self.assertTrue(self.episode().get("preview_error"))
        self.assertNotIn("SYNTHETIC PRIVATE DETAIL", str(self.episode()["preview_error"]))
        self.assertEqual(self.shot(1)["status"], "review_required")
        self.assertTrue((self.root / self.shot(1)["file"]).is_file())
        before_jobs = copy.deepcopy(state["jobs"])
        before_review = copy.deepcopy(self.shot(1)["review"])
        before_calls = self.brain.call_count
        self.assertEqual(self.api.submit_video.call_count, 2)
        pipeline.run(self.root)
        self.assertEqual(self.preview()["shot_count"], 2)
        self.assertFalse(self.episode().get("preview_error"))
        self.assertEqual(self.shot(1)["review"], before_review)
        self.assertEqual(core.load_project(self.root)["jobs"], before_jobs)
        self.assertEqual(self.brain.call_count, before_calls)
        self.assertEqual(self.api.submit_video.call_count, 2)

    def test_legacy_pending_candidate_gets_local_preview_without_model_or_video_requests(self):
        self.generate_through(1)
        # Recreate an old project with completed candidates but no preview index
        # or files. Only synthetic fixture output is removed.
        cuts = copy.deepcopy(self.episode().get("rough_cuts", []))
        with core.transaction(self.root) as state:
            state["episodes"][0].pop("rough_cuts", None)
        for cut in cuts:
            (self.root / cut["file"]).unlink()
        before = core.load_project(self.root)
        pipeline.media.compose.reset_mock()
        self.brain.reset_mock()
        self.api.reset_mock()
        state = pipeline.run(self.root)
        self.assertEqual(state["status"], "review_required", state.get("last_error"))
        self.assertEqual(self.preview()["shot_count"], 2)
        self.assertEqual(state["jobs"], before["jobs"])
        self.assertEqual(state["episodes"][0]["shots"], before["episodes"][0]["shots"])
        self.assertEqual(pipeline.media.compose.call_count, 1)
        self.brain.assert_not_called()
        self.api.submit_video.assert_not_called()
        self.api.poll_video.assert_not_called()
        self.api.download.assert_not_called()

    def test_middle_revision_preview_uses_new_version_and_preserved_later_candidates(self):
        self.generate_through(3)
        before = copy.deepcopy(self.episode())
        old_cuts = before["rough_cuts"]
        old_bytes = {cut["file"]: (self.root / cut["file"]).read_bytes() for cut in old_cuts}
        core.reject(self.root, "shot", before["shots"][1]["id"], 1, "合成测试：只修改第二段动作")
        state = pipeline.run(self.root)
        self.assertEqual(state["status"], "review_required", state.get("last_error"))
        after = self.episode()
        cut = self.preview()
        self.assertEqual(cut["shot_count"], 4)
        self.assertEqual([shot["version"] for shot in cut["shots"]], [1, 2, 1, 1])
        self.assertEqual(cut["shots"], [
            {"id": shot["id"], "version": shot["version"], "sha256": shot["sha256"]}
            for shot in after["shots"]])
        self.assertEqual(after["rough_cuts"][:len(old_cuts)], old_cuts)
        for entry in old_cuts:
            self.assertEqual((self.root / entry["file"]).read_bytes(), old_bytes[entry["file"]])
        for index in (2, 3):
            self.assertEqual(after["shots"][index]["file"], before["shots"][index]["file"])
            self.assertEqual(after["shots"][index]["sha256"], before["shots"][index]["sha256"])
            self.assertEqual(after["shots"][index]["status"], "review_required")
        self.assertEqual(self.api.submit_video.call_count, 5)
        before_calls = self.brain.call_count
        pipeline.run(self.root)
        self.assertEqual(self.api.submit_video.call_count, 5)
        self.assertEqual(self.brain.call_count, before_calls)
        self.assertEqual(self.preview(), cut)

    def test_human_approval_reuses_candidate_preview_without_reencoding(self):
        with core.transaction(self.root) as state:
            state["episodes"][0]["shots"] = state["episodes"][0]["shots"][:2]
        self.generate_through(1)
        cut = copy.deepcopy(self.preview())
        self.approve(1)
        pipeline.media.compose.reset_mock()
        state = pipeline.run(self.root)
        self.assertEqual(state["status"], "awaiting_episode_review", state.get("last_error"))
        self.assertEqual(self.episode()["rough_cuts"], [cut])
        self.assertEqual(pipeline.media.compose.call_count, 1)  # final render only
        self.assertNotIn("-rough-", str(pipeline.media.compose.call_args.args[1]))
        self.assertEqual(self.api.submit_video.call_count, 2)

    def test_identical_bytes_new_version_gets_new_preview_manifest(self):
        self.generate_through(1)
        old_shot, old_cut = copy.deepcopy(self.shot(1)), copy.deepcopy(self.preview())
        old_video = (self.root / old_shot["file"]).read_bytes()
        old_preview = (self.root / old_cut["file"]).read_bytes()
        core.reject(self.root, "shot", old_shot["id"], 1, "合成测试：要求第二段再生成一个版本")
        self.api.download.side_effect = lambda url, output: write(output,
            b"\xff\xd8\xffSYNTHETIC OFFICIAL TAIL" if url.endswith(".jpg") else old_video)
        pipeline.media.compose.reset_mock()
        state = pipeline.run(self.root)
        self.assertEqual(state["status"], "review_required", state.get("last_error"))
        new_cut = self.preview()
        self.assertEqual(self.shot(1)["version"], 2)
        self.assertEqual(self.shot(1)["sha256"], old_shot["sha256"])
        self.assertEqual(new_cut["shots"][-1], {
            "id": old_shot["id"], "version": 2, "sha256": old_shot["sha256"]})
        self.assertNotEqual(new_cut["key"], old_cut["key"])
        self.assertNotEqual(new_cut["file"], old_cut["file"])
        self.assertEqual(self.episode()["rough_cuts"][0], old_cut)
        self.assertEqual((self.root / old_cut["file"]).read_bytes(), old_preview)
        self.assertEqual(pipeline.media.compose.call_count, 1)
        self.assertEqual(self.api.submit_video.call_count, 3)

    def test_preview_prefix_accepts_completed_candidate_states_without_approving_them(self):
        self.generate_through(2)
        with core.transaction(self.root) as state:
            state["episodes"][0]["shots"][1]["status"] = "reviewing"
        before = copy.deepcopy(self.episode()["shots"])
        self.brain.reset_mock()
        self.api.reset_mock()
        cut = pipeline._refresh_candidate_preview(self.root, "E001")
        self.assertEqual(cut["shot_count"], 3)
        self.assertEqual(self.episode()["shots"], before)
        self.assertEqual([shot["status"] for shot in before[:3]], ["accepted", "reviewing", "review_required"])
        self.brain.assert_not_called()
        self.api.submit_video.assert_not_called()

    def test_preview_prefix_stops_at_missing_current_file_or_generating_old_version(self):
        self.generate_through(2)
        original = copy.deepcopy(core.load_project(self.root))
        for missing_current in (True, False):
            with self.subTest(missing_current=missing_current):
                with core.transaction(self.root) as state:
                    state["episodes"][0] = copy.deepcopy(original["episodes"][0])
                    shot = state["episodes"][0]["shots"][1]
                    shot["history"] = [{key: shot[key] for key in ("file", "sha256", "version")}]
                    shot.update(status="review_required" if missing_current else "generating", version=2)
                    if missing_current:
                        shot.pop("file")
                        shot.pop("sha256")
                before_cuts = copy.deepcopy(self.episode()["rough_cuts"])
                pipeline.media.compose.reset_mock()
                self.brain.reset_mock()
                self.api.reset_mock()
                self.assertIsNone(pipeline._refresh_candidate_preview(self.root, "E001"))
                self.assertEqual(self.episode()["rough_cuts"], before_cuts)
                pipeline.media.compose.assert_not_called()
                self.brain.assert_not_called()
                self.api.submit_video.assert_not_called()

    def test_technical_evidence_failure_does_not_publish_unchecked_candidate(self):
        self.generate_through(0)
        self.approve(0)
        evidence = pipeline.media.extract_evidence.side_effect

        def broken_evidence(video, directory, **kwargs):
            if "E001-S002-v1" in str(video):
                raise ValueError("Synthetic invalid video evidence")
            return evidence(video, directory, **kwargs)

        self.brain.reset_mock()
        with patch.object(pipeline.media, "extract_evidence", side_effect=broken_evidence):
            state = pipeline.run(self.root)
        self.assertEqual(state["status"], "blocked")
        self.assertEqual(self.episode().get("rough_cuts", []), [])
        self.assertNotIn("file", self.shot(1))
        self.assertEqual([call.args[1] for call in self.brain.call_args_list], ["refine_shot"])
        self.assertEqual(self.api.submit_video.call_count, 2)
        pipeline.media.compose.assert_not_called()


if __name__ == "__main__":
    unittest.main()
