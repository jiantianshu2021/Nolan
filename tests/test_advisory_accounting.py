"""Offline execution/billing separation for lost advisory reviews; synthetic IO."""
import copy
from pathlib import Path
import tempfile
import unittest

from norlan import core, pipeline


class AdvisoryAccountingTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        core.create_project(self.root, "合成审查恢复测试，不连接任何服务。")
        core.install_plan(self.root, pipeline.demo_plan())
        core.approve(self.root, "plan", "plan", 1)
        core.configure(self.root, limit=100)

    def artifact(self, relative, body=b"synthetic immutable media"):
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(body)
        return {"file": relative, "sha256": core.digest(path)}

    def unknown_review(self, amount=0, identifier="lost-review"):
        core.reserve(self.root, identifier, "llm", "review", amount,
                     pricing={"provider": "synthetic", "model": "synthetic-review", "currency": "CNY"})
        core.update_job(self.root, identifier, state="uncertain", accounting="unknown",
                        advisory_recovery={"status": "interrupted", "no_retry": True})
        return copy.deepcopy(core.load_project(self.root)["jobs"][identifier])

    def shot_target(self):
        record = self.artifact("media/E001/E001-S001-v1.mp4")
        core.reserve(self.root, "completed-video", "video", "E001-S001", 4)
        core.update_job(self.root, "completed-video", state="completed", accounting="spent",
                        task_id="synthetic-task", output=record["file"])
        with core.transaction(self.root) as state:
            core._find(state, "shot", "E001-S001").update(record, version=1, status="reviewing", job_id="completed-video")
        return {"kind": "shot", "id": "E001-S001", "version": 1, **record, "video_job_id": "completed-video"}

    def test_only_exact_interrupted_advisory_marker_exempts_an_unknown_job(self):
        valid = {"id": "lost-review", "kind": "llm", "item_id": "review", "task_id": None,
                 "state": "uncertain", "accounting": "unknown", "amount": 0,
                 "advisory_recovery": {"status": "interrupted", "no_retry": True}}
        self.assertTrue(core.is_nonblocking_advisory_unknown(valid))
        self.assertFalse(core.execution_unsettled(valid))
        alternatives = [{"kind": kind} for kind in ("video", "voice", "image", "music", "transcription")]
        alternatives += [{"item_id": method} for method in ("plan", "refine_shot", "plan_revision", "select_voice_audition")]
        alternatives += [{"task_id": "remote-id"}, {"state": "submitting"}, {"accounting": "reserved"},
                         {"advisory_recovery": {}}, {"advisory_recovery": {"status": "interrupted", "no_retry": 1}},
                         {"advisory_recovery": {"status": "completed", "no_retry": True}}, {"advisory_recovery": None}]
        for update in alternatives:
            with self.subTest(update=update):
                job = {**valid, **update}
                self.assertFalse(core.is_nonblocking_advisory_unknown(job))
                self.assertTrue(core.execution_unsettled(job))
        for invalid in (None, [], "review"):
            self.assertFalse(core.is_nonblocking_advisory_unknown(invalid))

    def test_blocking_unknown_jobs_returns_original_jobs_without_settling_anything(self):
        review = self.unknown_review()
        media = {"id": "unknown-video", "kind": "video", "state": "uncertain", "accounting": "unknown"}
        llm = {"id": "unknown-planning", "kind": "llm", "item_id": "plan", "state": "uncertain", "accounting": "unknown"}
        state = {"jobs": {review["id"]: review, media["id"]: media, llm["id"]: llm}}
        before = copy.deepcopy(state)
        blocked = core.blocking_unknown_jobs(state)
        self.assertEqual([job["id"] for job in blocked], ["unknown-video", "unknown-planning"])
        self.assertIs(blocked[0], media)
        self.assertEqual(state, before)
        self.assertTrue(core.execution_unsettled({"state": "submitted", "accounting": "spent"}))
        self.assertFalse(core.execution_unsettled({"state": "completed", "accounting": "spent"}))

    def test_zero_amount_is_not_a_free_bill_or_implicit_review_completion(self):
        original = self.unknown_review(0)
        state = core.load_project(self.root)
        self.assertEqual(state["jobs"][original["id"]], original)
        self.assertEqual(original["state"], "uncertain")
        self.assertEqual(original["accounting"], "unknown")
        self.assertNotIn("actual_cost", original["billing"])
        self.assertIsNone(state["budget"]["actual_cost"])
        core.reserve(self.root, "different-work", "image", "new-prop", 1)
        self.assertEqual(core.load_project(self.root)["jobs"][original["id"]], original)

    def test_nonzero_unknown_cost_stays_reserved_for_budget_and_is_not_released(self):
        core.configure(self.root, limit=10)
        original = self.unknown_review(7)
        core.reserve(self.root, "new-media", "video", "E001-S002", 3)
        state = core.load_project(self.root)
        self.assertEqual(state["budget"]["unknown"], 7)
        self.assertEqual(state["budget"]["reserved"], 3)
        self.assertEqual(state["budget"]["available"], 0)
        self.assertEqual(state["jobs"][original["id"]], original)
        with self.assertRaises(core.GateError):
            core.reserve(self.root, "over-budget", "image", "prop", 0.01)

    def test_media_and_nonreview_unknown_jobs_still_block_reservation(self):
        for kind, item in (("video", "E001-S001"), ("voice", "voice"), ("image", "hero"),
                           ("music", "E001"), ("transcription", "E001"), ("llm", "plan"), ("llm", "refine_shot")):
            with self.subTest(kind=kind, item=item):
                with core.transaction(self.root) as state:
                    state["jobs"] = {"unknown-job": {"id": "unknown-job", "kind": kind, "item_id": item,
                        "accounting": "unknown", "state": "uncertain", "amount": 0, "task_id": None,
                        "advisory_recovery": {"status": "interrupted", "no_retry": True}}}
                before = (self.root / "project.json").read_bytes()
                with self.assertRaises(core.GateError):
                    core.reserve(self.root, "blocked-new-work", "image", "prop", 1)
                self.assertEqual((self.root / "project.json").read_bytes(), before)

    def test_advisory_does_not_block_asset_inputs_bulk_choices_or_explicit_rework(self):
        original = self.unknown_review()
        core.set_asset_input(self.root, "hero", 0, 0, "custom", "用户确认的合成人物描述")
        core.set_remaining_asset_inputs_gpt(self.root, 1)
        record = self.artifact("media/hero-v1.png")
        with core.transaction(self.root) as state:
            core._find(state, "asset", "hero").update(record, status="candidate", version=1)
        core.reject(self.root, "asset", "hero", 1, "调整已说明的合成人物服装")
        state = core.load_project(self.root)
        self.assertEqual(core._find(state, "asset", "hero")["status"], "rejected")
        self.assertEqual(state["jobs"][original["id"]], original)

    def test_advisory_does_not_block_internal_role_migration_or_amendment_candidate(self):
        original = self.unknown_review()
        source = {"id": "announcer", "kind": "character", "voice_only": True, "name": "内部声源", "prompt": "画外逻辑声源",
                  "status": "candidate", "version": 1, "history": [], "feedback": ""}
        source["sha256"] = core.voice_only_digest(source)
        with core.transaction(self.root) as state:
            state["assets"].append(source)
        state = core.normalize_voice_sources(self.root)
        self.assertTrue(core.asset_ready(core._find(state, "asset", "announcer")))
        state = core.propose_asset_amendment(self.root, "E001-S001", 0, "新增合成道具", [{
            "id": "new-prop", "kind": "prop", "name": "合成道具", "prompt": "合成杯子"}],
            [{"id": "E001-S001", "add_asset_ids": ["new-prop"]}])
        proposal = state["asset_amendments"][-1]
        state = core.apply_asset_amendment(self.root, proposal["id"], "approve")
        self.assertEqual(state["asset_amendments"][-1]["status"], "awaiting_assets")
        self.assertEqual(state["jobs"][original["id"]], original)

    def test_unmarked_review_still_blocks_state_mutation_and_migration(self):
        original = self.unknown_review()
        with core.transaction(self.root) as state:
            state["jobs"][original["id"]].pop("advisory_recovery")
        with self.assertRaises(core.GateError):
            core.set_remaining_asset_inputs_gpt(self.root, 1)
        with self.assertRaises(core.GateError):
            core.propose_asset_amendment(self.root, "E001-S001", 0, "新增道具", [],
                                         [{"id": "E001-S001", "add_asset_ids": []}])

    def test_review_reservation_atomically_binds_current_video_target(self):
        target = self.shot_target()
        original_target = copy.deepcopy(target)
        job = core.reserve(self.root, "bound-review", "llm", "review", 0, advisory_target=target)
        self.assertEqual(job["purpose"], "advisory_review")
        self.assertEqual(job["advisory_target"], original_target)
        self.assertEqual((job["state"], job["accounting"]), ("submitting", "reserved"))
        target["id"] = "tampered-local-object"
        self.assertEqual(core.load_project(self.root)["jobs"]["bound-review"]["advisory_target"], original_target)

    def test_legacy_review_can_omit_target_without_fabricating_a_binding(self):
        job = core.reserve(self.root, "legacy-style-review", "llm", "review", 0)
        self.assertEqual(job["purpose"], "advisory_review")
        self.assertIsNone(job["advisory_target"])

    def test_rough_cut_target_binds_registered_batch_and_exact_bytes(self):
        cut = {"key": "synthetic-batch-key", **self.artifact("exports/E001-rough-2.mp4")}
        with core.transaction(self.root) as state:
            state["episodes"][0]["rough_cuts"] = [cut]
        target = {"kind": "rough_cut", "id": "E001", **cut}
        job = core.reserve(self.root, "batch-review", "llm", "review", 0, advisory_target=target)
        self.assertEqual(job["advisory_target"], target)

    def test_unregistered_one_shot_cut_can_still_bind_verified_bytes(self):
        target = {"kind": "rough_cut", "id": "E001", "key": "synthetic-one-shot", **self.artifact("exports/E001-rough-1.mp4")}
        job = core.reserve(self.root, "single-review", "llm", "review", 0, advisory_target=target)
        self.assertEqual(job["advisory_target"], target)

    def test_wrong_method_malformed_or_stale_target_never_reserves_or_increments_calls(self):
        target = self.shot_target()
        cases = [("video", "E001-S001", target), ("llm", "refine_shot", target), ("llm", "review", []),
                 ("llm", "review", {**target, "kind": "unknown"}), ("llm", "review", {**target, "version": True}),
                 ("llm", "review", {**target, "kind": []}), ("llm", "review", {**target, "kind": {}}),
                 ("llm", "review", {**target, "version": 2}), ("llm", "review", {**target, "sha256": "0" * 64}),
                 ("llm", "review", {**target, "file": "../escape.mp4"}),
                 ("llm", "review", {**target, "video_job_id": "other-job"}),
                 ("llm", "review", {**target, "approved": True})]
        for kind, item, value in cases:
            with self.subTest(kind=kind, item=item, value=value):
                before = (self.root / "project.json").read_bytes()
                with self.assertRaises((ValueError, core.GateError)):
                    core.reserve(self.root, "must-not-reserve", kind, item, 0, advisory_target=value)
                self.assertEqual((self.root / "project.json").read_bytes(), before)

    def test_target_bytes_or_completed_media_job_cannot_be_substituted(self):
        target = self.shot_target()
        (self.root / target["file"]).write_bytes(b"changed synthetic bytes")
        with self.assertRaises(core.GateError):
            core.reserve(self.root, "bad-bytes-review", "llm", "review", 0, advisory_target=target)
        (self.root / target["file"]).write_bytes(b"synthetic immutable media")
        with core.transaction(self.root) as state:
            state["jobs"][target["video_job_id"]]["state"] = "failed"
        with self.assertRaises(core.GateError):
            core.reserve(self.root, "bad-job-review", "llm", "review", 0, advisory_target=target)

    def test_update_job_cannot_rewrite_review_identity_or_target(self):
        target = self.shot_target()
        core.reserve(self.root, "frozen-review", "llm", "review", 0, advisory_target=target)
        for update in ({"purpose": "anything"}, {"advisory_target": None}, {"advisory_target": target}):
            before = (self.root / "project.json").read_bytes()
            with self.assertRaises(ValueError):
                core.update_job(self.root, "frozen-review", **update)
            self.assertEqual((self.root / "project.json").read_bytes(), before)

    def test_pause_and_llm_limits_are_not_bypassed_by_review_recovery(self):
        original = self.unknown_review()
        core.set_paused(self.root, True)
        with self.assertRaises(core.GateError):
            core.reserve(self.root, "paused-new-work", "video", "E001-S001", 1)
        core.set_paused(self.root, False)
        core.configure(self.root, {"max_llm_calls": 1})
        with self.assertRaises(core.GateError):
            core.reserve(self.root, "over-limit-review", "llm", "review", 0)
        self.assertEqual(core.load_project(self.root)["jobs"][original["id"]], original)

    def test_unavailable_report_metadata_is_not_saved_as_gpt_observations_in_human_decision(self):
        target = self.shot_target()
        fallback = "自动审查未完成，请人工观看本版本视频。"
        with core.transaction(self.root) as state:
            core._find(state, "shot", target["id"]).update(status="review_required", review={
                "verdict": "uncertain", "issues": [fallback], "version": target["version"], "sha256": target["sha256"],
                "report_unavailable": True, "review_interrupted": True, "scope": "sampled_frames_only"})
        state = core.approve(self.root, "shot", target["id"], target["version"])
        shot = core._find(state, "shot", target["id"])
        decision = shot["human_decision"]
        self.assertTrue(core.human_approved_shot(shot))
        self.assertTrue(decision["report_bound"])
        self.assertFalse(decision["report_available"])
        self.assertTrue(decision["report"]["report_unavailable"])
        self.assertTrue(decision["report"]["review_interrupted"])
        self.assertEqual(decision["report"]["sha256"], target["sha256"])
        self.assertFalse({"verdict", "issues"}.intersection(decision["report"]))
        self.assertIn("报告未完成", decision["report_note"])
        self.assertIn("实际观看", decision["report_note"])
        self.assertNotIn("GPT 的原始检查意见", decision["report_note"])
        self.assertEqual(shot["review"]["issues"], [fallback])

    def test_normal_bound_report_keeps_original_snapshot_and_available_status(self):
        target = self.shot_target()
        review = {"verdict": "fail", "issues": ["合成可供人工判断的原始检查意见"],
                  "version": target["version"], "sha256": target["sha256"]}
        with core.transaction(self.root) as state:
            core._find(state, "shot", target["id"]).update(status="review_required", review=review)
        state = core.approve(self.root, "shot", target["id"], target["version"])
        decision = core._find(state, "shot", target["id"])["human_decision"]
        self.assertTrue(decision["report_available"])
        self.assertTrue(decision["report_bound"])
        self.assertEqual(decision["report"], {"verdict": review["verdict"], "issues": review["issues"]})
        self.assertIn("GPT 的原始检查意见", decision["report_note"])


if __name__ == "__main__":
    unittest.main()
