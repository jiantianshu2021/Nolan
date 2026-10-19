"""Offline restart regressions: an interrupted report never recreates media."""
import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from norlan import core, interactive, mcp_server, pipeline
from test_pipeline import choose_gpt_assets, fake_media, write


class AdvisoryRestartTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.enterContext(fake_media())
        core.create_project(self.root, "合成审查恢复测试小说。", "demo")
        core.install_plan(self.root, pipeline.demo_plan())
        core.approve(self.root, "plan", "plan", 1)
        choose_gpt_assets(self.root)
        state = pipeline.run(self.root)
        for asset in state["assets"]:
            core.approve(self.root, "asset", asset["id"], asset["version"])
        core.configure(self.root, {"caps": {"video_second": 1, "llm": 2}}, limit=100)
        with core.transaction(self.root) as state:
            state["mode"] = "live"
        # These mocks both prevent real service access and detect accidental
        # recovery paths that ask GPT or submit a replacement video.
        self.ark = self.enterContext(patch.object(pipeline, "_ark", side_effect=AssertionError("must not contact a media provider")))
        self.brain = self.enterContext(patch.object(pipeline, "_brain", side_effect=AssertionError("must not request a new GPT report")))

    def shot(self, index=0):
        return core.load_project(self.root)["episodes"][0]["shots"][index]

    def completed_shot(self, index=0, status="reviewing"):
        shot = self.shot(index)
        relative = f"media/E001/{shot['id']}-v1.mp4"
        path = write(self.root / relative, b"SYNTHETIC COMPLETED VIDEO " + shot["id"].encode())
        job_id = "synthetic-video-" + shot["id"]
        core.reserve(self.root, job_id, "video", shot["id"], 4)
        core.update_job(self.root, job_id, state="completed", accounting="spent", task_id="provider-" + shot["id"],
                        output=relative, version=1)
        with core.transaction(self.root) as state:
            saved = core._find(state, "shot", shot["id"])
            saved.update(status=status, version=1, file=relative, sha256=core.digest(path), job_id=job_id,
                         attempts=[{"job_id": job_id, "revision": 0, "version": 1, "time": core.now()}])
            state["status"] = "generating"
        current = self.shot(index)
        return {"kind": "shot", "id": current["id"], "version": current["version"], "sha256": current["sha256"],
                "file": relative, "video_job_id": job_id}

    def interrupted_review(self, target, amount=0, *, legacy=False, state="submitting", cache=None, valid_hash=True):
        job_id = "synthetic-gpt-review"
        core.reserve(self.root, job_id, "llm", "review", amount,
                     advisory_target=None if legacy else copy.deepcopy(target))
        if legacy:
            # Upgrade fixtures reproduce the old on-disk format; this is not
            # an API for changing the identity of a real registered call.
            with core.transaction(self.root) as current:
                current["jobs"][job_id].pop("purpose", None)
                current["jobs"][job_id].pop("advisory_target", None)
        updates = {"state": state}
        if state == "uncertain":
            updates["accounting"] = "unknown"
        elif state == "completed":
            updates["accounting"] = "spent"
        if cache is not None:
            updates.update(review_result=copy.deepcopy(cache),
                           review_result_hash=core._json_digest(cache) if valid_hash else "0" * 64)
        core.update_job(self.root, job_id, **updates)
        return job_id

    def assert_no_external_calls(self):
        self.ark.assert_not_called()
        self.brain.assert_not_called()

    def test_zero_and_nonzero_interrupted_reports_keep_unknown_accounting_but_allow_human_review(self):
        for amount in (0, 2):
            with self.subTest(amount=amount):
                if amount:
                    # Reuse the same fixture without manufacturing a second
                    # provider call or bypassing an existing unknown ledger.
                    with core.transaction(self.root) as state:
                        state["jobs"].pop("synthetic-gpt-review")
                target = self.completed_shot() if not amount else {"kind": "shot", **{
                    key: self.shot()[key] for key in ("id", "version", "sha256", "file")}, "video_job_id": self.shot()["job_id"]}
                with core.transaction(self.root) as state:
                    current = core._find(state, "shot", target["id"])
                    current["status"] = "reviewing"
                    current.pop("review", None)
                job_id = self.interrupted_review(target, amount)
                before = copy.deepcopy(core.load_project(self.root)["assets"])
                for _ in range(3):
                    result = pipeline.run(self.root)
                    self.assertEqual(result["status"], "review_required", result.get("last_error"))
                    current = self.shot()
                    self.assertEqual(current["status"], "review_required")
                    self.assertEqual((current["version"], current["sha256"], current["file"]),
                                     (target["version"], target["sha256"], target["file"]))
                    self.assertFalse(core.human_approved_shot(current))
                    report = current["review"]
                    self.assertEqual(report["verdict"], "uncertain")
                    self.assertTrue(report["report_unavailable"])
                    self.assertTrue(report["review_interrupted"])
                    self.assertEqual((report["version"], report["sha256"]), (target["version"], target["sha256"]))
                    ledger = result["jobs"][job_id]
                    self.assertEqual((ledger["state"], ledger["accounting"], ledger["amount"]), ("uncertain", "unknown", amount))
                    self.assertEqual(ledger["advisory_recovery"]["status"], "interrupted")
                    self.assertTrue(ledger["advisory_recovery"]["no_retry"])
                    self.assertEqual(result["budget"]["unknown"], amount)
                    self.assertEqual(result["assets"], before)
                    self.assertEqual(len(result["jobs"]), 4)  # 2 assets + video + the original review.
                self.assert_no_external_calls()

    def test_legacy_report_without_target_is_marked_unavailable_without_inventing_a_result(self):
        target = self.completed_shot()
        job_id = self.interrupted_review(target, legacy=True)
        result = pipeline.run(self.root)
        self.assertEqual(result["status"], "review_required", result.get("last_error"))
        self.assertTrue(self.shot()["review"]["report_unavailable"])
        self.assertEqual(self.shot()["review"]["verdict"], "uncertain")
        self.assertIsNone(result["jobs"][job_id].get("advisory_target"))
        self.assertEqual(result["jobs"][job_id]["accounting"], "unknown")
        self.assert_no_external_calls()

    def test_cached_completed_report_is_restored_without_calling_gpt_again(self):
        target = self.completed_shot()
        report = {"verdict": "fail", "issues": ["[待核实] 合成画面局部遮挡。"],
                  "observed_end_state": "合成尾态。", "repair_prompt": "供人工选择的建议。",
                  "scope": "sampled_frames_only", "audio_assessed": False}
        job_id = self.interrupted_review(target, amount=2, state="completed", cache=report)
        for _ in range(2):
            result = pipeline.run(self.root)
            self.assertEqual(result["status"], "review_required", result.get("last_error"))
            restored = self.shot()["review"]
            for key, value in report.items():
                self.assertEqual(restored[key], value)
            self.assertEqual((restored["version"], restored["sha256"]), (target["version"], target["sha256"]))
            self.assertFalse(core.human_approved_shot(self.shot()))
            self.assertEqual(result["jobs"][job_id]["accounting"], "spent")
            self.assertEqual(result["jobs"][job_id]["review_result"], report)
        self.assert_no_external_calls()

    def test_corrupt_report_cache_is_not_published_as_a_valid_result(self):
        target = self.completed_shot()
        report = {"verdict": "pass", "issues": [], "observed_end_state": "未经验证", "repair_prompt": ""}
        self.interrupted_review(target, state="completed", cache=report, valid_hash=False)
        result = pipeline.run(self.root)
        self.assertEqual(result["status"], "review_required", result.get("last_error"))
        self.assertEqual(self.shot()["review"]["verdict"], "uncertain")
        self.assertTrue(self.shot()["review"]["report_unavailable"])
        self.assert_no_external_calls()

    def test_unverified_media_hash_or_wrong_video_job_never_becomes_an_approvable_candidate(self):
        target = self.completed_shot()
        self.interrupted_review(target)
        for field, value in (("sha256", "0" * 64), ("job_id", "nonexistent-video-job")):
            with self.subTest(field=field):
                with core.transaction(self.root) as state:
                    current = core._find(state, "shot", target["id"])
                    current.update(status="reviewing", sha256=target["sha256"], job_id=target["video_job_id"])
                    current[field] = value
                result = pipeline.run(self.root)
                self.assertEqual(result["status"], "blocked")
                self.assertNotEqual(self.shot()["status"], "review_required")
                self.assertFalse(core.human_approved_shot(self.shot()))
                with self.assertRaises(core.GateError):
                    core.approve(self.root, "shot", target["id"], 1)
        self.assert_no_external_calls()

    def test_unknown_media_planning_and_refinement_calls_remain_blocking(self):
        self.completed_shot()
        for kind, purpose in (("video", "E001-S002"), ("llm", "plan"), ("llm", "refine_shot")):
            with self.subTest(kind=kind, purpose=purpose):
                with core.transaction(self.root) as state:
                    state["jobs"].pop("synthetic-unknown", None)
                core.reserve(self.root, "synthetic-unknown", kind, purpose, 1)
                core.update_job(self.root, "synthetic-unknown", state="uncertain", accounting="unknown")
                result = pipeline.run(self.root)
                self.assertEqual(result["status"], "blocked")
                self.assertEqual(result["jobs"]["synthetic-unknown"]["accounting"], "unknown")
                self.assertEqual(result["budget"]["unknown"], 1)
        self.assert_no_external_calls()

    def test_rough_cut_review_recovers_one_key_without_repeated_gpt_or_video(self):
        for index in (0, 1):
            target = self.completed_shot(index, status="review_required")
            core.approve(self.root, "shot", target["id"], target["version"])
        episode = core.load_project(self.root)["episodes"][0]
        cut = pipeline._rough_cut(self.root, episode, 2)
        target = {"kind": "rough_cut", "id": episode["id"], "key": cut["key"], "sha256": cut["sha256"], "file": cut["file"]}
        job_id = self.interrupted_review(target, amount=2)
        for attempt in range(3):
            result = pipeline.run(self.root)
            self.assertEqual(result["status"], "review_required" if attempt == 0 else "awaiting_episode_review", result.get("last_error"))
            reports = [report for report in result["episodes"][0]["reviews"] if report.get("key") == cut["key"]]
            self.assertEqual(len(reports), 1)
            self.assertTrue(reports[0]["report_unavailable"])
            self.assertEqual(reports[0]["verdict"], "uncertain")
            self.assertEqual(result["jobs"][job_id]["accounting"], "unknown")
        self.assert_no_external_calls()

    def single_shot_cut_restart(self, *, legacy):
        # A single generated segment is also a complete episode. The batch
        # report must not disappear just because its cut has only one source.
        with core.transaction(self.root) as state:
            state["episodes"][0]["shots"] = state["episodes"][0]["shots"][:1]
        target = self.completed_shot(status="review_required")
        core.approve(self.root, "shot", target["id"], target["version"])
        episode = core.load_project(self.root)["episodes"][0]
        cut = pipeline._rough_cut(self.root, episode, 1)
        if not legacy:
            registered = core.load_project(self.root)["episodes"][0].get("rough_cuts", [])
            self.assertEqual([item["key"] for item in registered], [cut["key"]])
        else:
            # Older releases wrote a one-source cut but omitted its registry
            # entry and review target. Only the saved batch-stage hint survives.
            with core.transaction(self.root) as state:
                current = state["episodes"][0]
                current["rough_cuts"] = []
                current["progress_stage"] = {"stage": "batch_review", "shot_id": target["id"],
                                             "started_at": core.now(), "updated_at": core.now()}
        review_target = {"kind": "rough_cut", "id": episode["id"], "key": cut["key"],
                         "sha256": cut["sha256"], "file": cut["file"]}
        job_id = self.interrupted_review(review_target, amount=2, legacy=legacy)
        original_bytes = (self.root / cut["file"]).read_bytes()
        for attempt in range(3):
            result = pipeline.run(self.root)
            self.assertEqual(result["status"], "review_required" if attempt == 0 else "awaiting_episode_review",
                             result.get("last_error"))
            current = result["episodes"][0]
            self.assertEqual(len(current.get("rough_cuts", [])), 1)
            recovered = current["rough_cuts"][0]
            self.assertEqual(recovered["shot_count"], 1)
            self.assertEqual(recovered["shots"], [{"id": target["id"], "version": 1, "sha256": target["sha256"]}])
            self.assertEqual(len(current["reviews"]), 1)
            self.assertEqual(current["reviews"][0]["key"], recovered["key"])
            self.assertEqual(current["reviews"][0]["verdict"], "uncertain")
            self.assertTrue(current["reviews"][0]["report_unavailable"])
            ledger = result["jobs"][job_id]
            self.assertEqual((ledger["state"], ledger["accounting"], ledger["amount"]), ("uncertain", "unknown", 2))
            if legacy:
                self.assertIsNone(ledger.get("advisory_target"))
            self.assertEqual((self.root / cut["file"]).read_bytes(), original_bytes)
        self.assert_no_external_calls()

    def test_registered_single_shot_cut_interruption_never_restarts_gpt_or_media(self):
        self.single_shot_cut_restart(legacy=False)

    def test_legacy_unregistered_single_shot_cut_is_rebuilt_locally_without_restarting_gpt_or_media(self):
        self.single_shot_cut_restart(legacy=True)

    def test_operator_guidance_separates_an_interrupted_report_from_media_failure_and_charges(self):
        for text in (interactive._DIRECTOR_INSTRUCTIONS, mcp_server.TOOL_MAP["run_production"]["description"]):
            self.assertIn("不表示视频", text)
            self.assertIn("待核实", text)
            self.assertIn("不自动重新审查", text)
            self.assertIn("未收费", text)
        diagnostic = mcp_server._diagnostic("GPT审查报告中断")
        self.assertEqual(diagnostic["category"], "advisory_review")


if __name__ == "__main__":
    unittest.main()
