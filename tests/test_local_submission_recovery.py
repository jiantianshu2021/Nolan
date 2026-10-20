"""Synthetic legacy 64 MB recovery; no provider or GPT calls are real."""
import copy
import hashlib
import unittest
from unittest.mock import patch

from norlan import core, local_submission, pipeline, prompt_gate, storyboard_revision
import test_prompt_gate as fixture


class LocalSubmissionRecoveryTests(unittest.TestCase):
    setUp = fixture.PromptGateTests.setUp
    answer = fixture.PromptGateTests.answer
    amendment = fixture.PromptGateTests.amendment
    shot = fixture.PromptGateTests.shot
    confirm = fixture.PromptGateTests.confirm

    def fail_locally(self, *, manual=False, storyboard=False):
        if storyboard:
            with core.transaction(self.root) as state:
                shot = state["episodes"][0]["shots"][0]
                before = storyboard_revision.baseline(shot)
                subshots = [{"start": 0, "end": shot["duration"], "prompt": "合成完整单镜头。"}]
                proposal = {"summary": "合成局部分镜草稿", "duration": shot["duration"], "subshots": subshots}
                shot["subshots"] = subshots
                shot["storyboard_revision"] = {**proposal, "before_count": 1, "before_duration": shot["duration"]}
                shot["storyboard_revision_binding"] = {"signature": storyboard_revision.feedback_signature(shot),
                    "revision": 0, "version": 0, "before_count": 1, "before": before,
                    "plan_hash": storyboard_revision._hash(proposal)}
        self.assertEqual(pipeline.run(self.root)["status"], "awaiting_prompt_approval")
        current = self.shot()
        candidate = current["prompt_candidate"]
        if manual:
            raw = "  合成手动定稿：保留首尾空白。\n本句必须逐字传给视频模型。\n "
            core.edit_prompt_and_approve(self.root, current["id"], candidate["id"], candidate["request_hash"], raw)
        else:
            self.confirm()
        candidate = copy.deepcopy(self.shot()["prompt_candidate"])
        identifier = "synthetic-local-body-too-large"
        core.reserve(self.root, identifier, "video", current["id"], 9.1425)
        detail = {"stage": "local_validation", "message": local_submission.OVERSIZED_REQUEST_MESSAGE,
                  "uncertain": False}
        core.update_job(self.root, identifier, state="failed", accounting="spent", failure=detail)
        with core.transaction(self.root) as state:
            shot = state["episodes"][0]["shots"][0]
            shot.update(status="generation_failed", version=candidate["version"], job_id=identifier,
                        generation_error=detail["message"])
            shot.setdefault("attempts", []).append({"job_id": identifier, "revision": shot.get("revision", 0),
                "version": shot["version"], "time": core.now()})
            job = state["jobs"][identifier]
            job["failure_history"] = [{**detail, "recorded_at": core.now()}]
            state["status"] = "generation_failed"
        self.brain.reset_mock()
        self.api.reset_mock()
        return identifier, candidate, core.load_project(self.root)

    def assert_no_calls(self):
        self.brain.assert_not_called()
        self.api.submit_video.assert_not_called()
        self.api.poll_video.assert_not_called()

    def test_proven_local_failure_releases_only_budget_and_requires_fresh_human_consent(self):
        identifier, old_candidate, before = self.fail_locally()
        originals = copy.deepcopy(before["episodes"][0]["shots"][1:])
        self.assertEqual(local_submission.recover_oversized_requests(self.root), [self.shot()["id"]])
        state, shot = core.load_project(self.root), self.shot()
        job, candidate = state["jobs"][identifier], shot["prompt_candidate"]
        self.assertEqual(job["accounting"], "released")
        self.assertEqual(job["state"], "failed")
        self.assertEqual(job["amount"], 9.1425)
        self.assertEqual(job["failure"], before["jobs"][identifier]["failure"])
        self.assertEqual(job["failure_history"], before["jobs"][identifier]["failure_history"])
        self.assertAlmostEqual(state["budget"]["available"] - before["budget"]["available"], 9.1425)
        self.assertEqual(state["budget"]["cost_pending_jobs"], before["budget"]["cost_pending_jobs"] - 1)
        self.assertEqual(shot["status"], "awaiting_prompt_approval")
        self.assertEqual(shot["version"], old_candidate["version"])
        self.assertEqual(shot["revision"], 1)
        self.assertEqual(candidate["version"], old_candidate["version"] + 1)
        self.assertEqual(candidate["revision"], 1)
        self.assertEqual(candidate["prompt"], old_candidate["prompt"])
        self.assertEqual(candidate["refined"], old_candidate["refined"])
        self.assertEqual(candidate["reference_manifest"], old_candidate["reference_manifest"])
        self.assertEqual(candidate.get("video_model"), old_candidate.get("video_model"))
        self.assertNotEqual(candidate["id"], old_candidate["id"])
        self.assertNotEqual(candidate["request_hash"], old_candidate["request_hash"])
        self.assertNotIn("approved_at", candidate)
        for field in ("job_id", "prompt_approval", "generation_error"):
            self.assertNotIn(field, shot)
        self.assertEqual(shot["attempts"], before["episodes"][0]["shots"][0]["attempts"])
        self.assertEqual(state["episodes"][0]["shots"][1:], originals)
        prompt_gate._validate_inputs(self.root, state, shot, candidate)
        self.assert_no_calls()
        for _ in range(3):
            self.assertEqual(local_submission.recover_oversized_requests(self.root), [])
            self.assertEqual(pipeline.run(self.root)["status"], "awaiting_prompt_approval")
        state = core.load_project(self.root)
        self.assertEqual(sum(row["action"] == "local_video_submission_recovered" for row in state["events"]), 1)
        self.assertEqual(self.shot()["prompt_candidate"], candidate)
        self.assert_no_calls()

    def test_manual_exact_text_and_original_source_survive_then_one_confirm_submits_once(self):
        identifier, old_candidate, before = self.fail_locally(manual=True)
        self.assertEqual(local_submission.recover_oversized_requests(self.root), [self.shot()["id"]])
        candidate = self.shot()["prompt_candidate"]
        for key in ("mode", "source_prompt_sha256", "edited_prompt_sha256"):
            self.assertEqual(candidate["manual_prompt"][key], old_candidate["manual_prompt"][key])
        self.assertNotEqual(candidate["manual_prompt"]["source_candidate_id"], old_candidate["manual_prompt"]["source_candidate_id"])
        self.assertNotEqual(candidate["manual_prompt"]["source_request_hash"], old_candidate["manual_prompt"]["source_request_hash"])
        self.assertEqual(candidate["refined"], prompt_gate._manual_refined(candidate["prompt"], candidate["manual_prompt"]))
        self.assertEqual(candidate["prompt"].encode("utf-8"), old_candidate["prompt"].encode("utf-8"))
        self.assertEqual(candidate["manual_prompt"]["edited_prompt_sha256"],
                         hashlib.sha256(candidate["prompt"].encode("utf-8")).hexdigest())
        self.assertEqual(self.shot()["prompt_history"][:-2], before["episodes"][0]["shots"][0]["prompt_history"])
        self.assertEqual(self.shot()["prompt_history"][-2]["failed_job_id"], identifier)
        rebound_source = self.shot()["prompt_history"][-1]
        self.assertEqual(rebound_source["id"], candidate["manual_prompt"]["source_candidate_id"])
        self.assertEqual(rebound_source["version"], candidate["version"])
        self.assertEqual(rebound_source["revision"], candidate["revision"])
        self.assertEqual(rebound_source["local_submission_recovery_origin"]["id"], old_candidate["manual_prompt"]["source_candidate_id"])
        old_source = before["episodes"][0]["shots"][0]["prompt_history"][-1]
        for key in ("prompt", "refined", "reference_manifest"):
            self.assertEqual(rebound_source[key], old_source[key])
        prompt_gate._validate_inputs(self.root, core.load_project(self.root), self.shot(), candidate)
        with self.assertRaises(core.GateError):
            core.approve_prompt(self.root, self.shot()["id"], old_candidate["id"], old_candidate["request_hash"])
        self.assert_no_calls()
        self.confirm()
        self.brain.side_effect = AssertionError("Recovered manual prompt must never be rewritten by GPT")
        self.assertEqual(pipeline.run(self.root)["status"], "review_required")
        self.assertEqual(self.api.submit_video.call_args.args[0], old_candidate["prompt"])
        self.api.submit_video.assert_called_once()
        self.brain.assert_not_called()
        for _ in range(2):
            pipeline.run(self.root)
        self.api.submit_video.assert_called_once()
        self.assertEqual(core.load_project(self.root)["jobs"][identifier]["accounting"], "released")

    def test_recovered_normal_prompt_with_cached_feedback_submits_without_replanning(self):
        with core.transaction(self.root) as state:
            state["episodes"][0]["shots"][0]["feedback"] = "合成已有返修意见：落地时踉跄，再站稳。"
        identifier, original, before = self.fail_locally()
        original_shot = before["episodes"][0]["shots"][0]
        self.assertTrue(original_shot["revision_analysis"]["signature"])
        self.assertEqual(local_submission.recover_oversized_requests(self.root), [original_shot["id"]])
        recovered = self.shot()
        self.assertEqual(recovered["feedback"], original_shot["feedback"])
        self.assertEqual(recovered["revision_analysis"], original_shot["revision_analysis"])
        self.assertNotEqual(recovered["revision_analysis"]["signature"], storyboard_revision.feedback_signature(recovered))
        self.assertEqual(recovered["prompt_candidate"]["prompt"], original["prompt"])
        self.assert_no_calls()
        self.confirm()
        self.brain.side_effect = AssertionError("Confirmed recovered prompts must not call GPT again")
        with patch.object(pipeline, "_revision_analysis", side_effect=AssertionError("Do not re-plan approved text")) as planner, \
             patch.object(pipeline, "_system_voice_reference", side_effect=AssertionError("Do not change approved references")) as voices:
            self.assertEqual(pipeline.run(self.root)["status"], "review_required")
        planner.assert_not_called()
        voices.assert_not_called()
        self.brain.assert_not_called()
        self.api.submit_video.assert_called_once()
        self.assertEqual(self.api.submit_video.call_args.args[0].encode("utf-8"), original["prompt"].encode("utf-8"))
        for _ in range(2):
            self.assertEqual(pipeline.run(self.root)["status"], "review_required")
        self.api.submit_video.assert_called_once()
        self.assertEqual(core.load_project(self.root)["jobs"][identifier]["accounting"], "released")

    def test_recovered_manual_prompt_can_be_edited_again_without_nesting_its_source(self):
        identifier, original, before = self.fail_locally(manual=True)
        self.assertEqual(local_submission.recover_oversized_requests(self.root), [self.shot()["id"]])
        restored = copy.deepcopy(self.shot()["prompt_candidate"])
        source = copy.deepcopy(prompt_gate._manual_source(self.shot(), restored["manual_prompt"], restored["prompt"]))
        self.assertNotIn("manual_prompt", source)
        replacement = "  合成二次人工定稿：保持慢速落地。\n踉跄两步以后站稳，保留首尾空白。  "
        core.edit_prompt_and_approve(self.root, self.shot()["id"], restored["id"], restored["request_hash"], replacement)
        edited = self.shot()["prompt_candidate"]
        for key in ("source_candidate_id", "source_request_hash", "source_prompt_sha256"):
            self.assertEqual(edited["manual_prompt"][key], restored["manual_prompt"][key])
        self.assertNotEqual(edited["manual_prompt"]["edited_prompt_sha256"], restored["manual_prompt"]["edited_prompt_sha256"])
        self.assertEqual(prompt_gate._manual_source(self.shot(), edited["manual_prompt"], replacement), source)
        self.assertNotEqual(edited["id"], restored["id"])
        self.assertNotEqual(edited["request_hash"], restored["request_hash"])
        history = self.shot()["prompt_history"]
        self.assertEqual(history[:len(before["episodes"][0]["shots"][0]["prompt_history"])],
                         before["episodes"][0]["shots"][0]["prompt_history"])
        self.assertTrue(any(row.get("id") == original["id"] and row.get("prompt") == original["prompt"] for row in history))
        self.assertTrue(any(row.get("id") == restored["id"] and row.get("prompt") == original["prompt"] for row in history))
        self.assertEqual(edited["prompt"].encode("utf-8"), replacement.encode("utf-8"))
        prompt_gate._validate_inputs(self.root, core.load_project(self.root), self.shot(), edited)
        self.assert_no_calls()
        self.brain.side_effect = AssertionError("Second manual edit must go directly to the video model")
        self.assertEqual(pipeline.run(self.root)["status"], "review_required")
        self.assertEqual(self.api.submit_video.call_args.args[0].encode("utf-8"), replacement.encode("utf-8"))
        self.assertEqual(self.shot()["prompt_candidate"]["prompt"], replacement)
        self.api.submit_video.assert_called_once()
        self.brain.assert_not_called()
        for _ in range(2):
            self.assertEqual(pipeline.run(self.root)["status"], "review_required")
        self.api.submit_video.assert_called_once()
        self.assertEqual(core.load_project(self.root)["jobs"][identifier]["accounting"], "released")

    def test_any_remote_or_billing_evidence_prevents_refund_and_restoration(self):
        identifier, _, baseline = self.fail_locally()
        cases = [("job", "task_id", "synthetic-task"), ("job", "request_id", "synthetic-request"),
                 ("job", "output", "media/synthetic.mp4"), ("job", "usage", {"completion_tokens": 0}),
                 ("failure", "request_id", "synthetic-request"), ("failure", "http_status", 400),
                 ("failure", "provider_code", "InvalidParameter"), ("failure", "uncertain", True),
                 ("failure", "stage", "submission"), ("failure", "message", "请求太大。"),
                 ("billing", "usage_cost", 0), ("billing", "actual_cost", 0),
                 ("billing", "bill_reference", "synthetic-bill"),
                 ("billing", "usage", {"total_tokens": 0}),
                 ("billing", "request_id", "synthetic-request"),
                 ("history", "http_status", 400)]
        for section, key, value in cases:
            with self.subTest(section=section, key=key):
                with core.transaction(self.root) as state:
                    state.clear(); state.update(copy.deepcopy(baseline))
                    job = state["jobs"][identifier]
                    target = (job if section == "job" else job["failure_history"][0] if section == "history"
                              else job.setdefault(section, {}))
                    target[key] = value
                self.assertEqual(local_submission.recover_oversized_requests(self.root), [])
                state = core.load_project(self.root)
                self.assertEqual(state["jobs"][identifier]["accounting"], "spent")
                self.assertEqual(self.shot()["status"], "generation_failed")
                self.assertIn("prompt_approval", self.shot())
                self.assert_no_calls()

    def test_missing_or_changed_consent_sources_are_retained_without_refund(self):
        identifier, _, baseline = self.fail_locally(manual=True)
        for field in ("candidate_hash", "approval_hash", "candidate_prompt", "source_history", "version", "attempt"):
            with self.subTest(field=field):
                with core.transaction(self.root) as state:
                    state.clear(); state.update(copy.deepcopy(baseline))
                    shot = state["episodes"][0]["shots"][0]
                    if field == "candidate_hash": shot["prompt_candidate"]["request_hash"] = "0" * 64
                    elif field == "approval_hash": shot["prompt_approval"]["request_hash"] = "0" * 64
                    elif field == "candidate_prompt": shot["prompt_candidate"]["prompt"] += "变化"
                    elif field == "source_history": shot["prompt_history"] = []
                    elif field == "version": shot["version"] += 1
                    else: shot["attempts"] = []
                self.assertEqual(local_submission.recover_oversized_requests(self.root), [])
                first = core.load_project(self.root)
                self.assertEqual(first["jobs"][identifier]["accounting"], "spent")
                self.assertEqual(self.shot()["status"], "generation_failed")
                self.assertEqual(first["jobs"][identifier]["local_submission_recovery"]["status"], "blocked")
                events = len(first["events"])
                self.assertEqual(local_submission.recover_oversized_requests(self.root), [])
                self.assertEqual(len(core.load_project(self.root)["events"]), events)
                self.assert_no_calls()

    def test_changed_reference_bytes_never_release_budget(self):
        identifier, candidate, _ = self.fail_locally()
        image = self.root / candidate["reference_manifest"]["images"][0]
        image.write_bytes(b"SYNTHETIC changed immutable reference")
        self.assertEqual(local_submission.recover_oversized_requests(self.root), [])
        state = core.load_project(self.root)
        self.assertEqual(state["jobs"][identifier]["accounting"], "spent")
        self.assertEqual(self.shot()["status"], "generation_failed")
        self.assert_no_calls()

    def test_storyboard_is_rebound_locally_without_changing_definition_or_neighbors(self):
        identifier, _, before = self.fail_locally(manual=True, storyboard=True)
        old_shot = before["episodes"][0]["shots"][0]
        self.assertEqual(local_submission.recover_oversized_requests(self.root), [old_shot["id"]])
        shot = self.shot()
        self.assertEqual(storyboard_revision.definition(shot), storyboard_revision.definition(old_shot))
        self.assertEqual(shot["storyboard_revision"], old_shot["storyboard_revision"])
        self.assertEqual(storyboard_revision.public(shot), old_shot["storyboard_revision"])
        self.assertEqual(shot["storyboard_revision_binding"]["revision"], shot["revision"])
        self.assertEqual(shot["storyboard_revision_binding"]["version"], shot["version"])
        self.assertEqual(core.load_project(self.root)["episodes"][0]["shots"][1:], before["episodes"][0]["shots"][1:])
        self.assertEqual(core.load_project(self.root)["jobs"][identifier]["accounting"], "released")
        self.assert_no_calls()

    def test_future_continuity_binding_is_declined_without_mutating_neighbor(self):
        identifier, _, _ = self.fail_locally()
        with core.transaction(self.root) as state:
            state["episodes"][0]["shots"][0]["future_continuity_binding"] = {"signature": "synthetic-signature", "updates": []}
        before = core.load_project(self.root)
        self.assertEqual(local_submission.recover_oversized_requests(self.root), [])
        state = core.load_project(self.root)
        self.assertEqual(state["jobs"][identifier]["accounting"], "spent")
        self.assertIn("后段", state["jobs"][identifier]["local_submission_recovery"]["reason"])
        self.assertEqual(state["episodes"], before["episodes"])
        self.assert_no_calls()

    def test_paused_recovery_preserves_pause_and_no_consent_is_reused(self):
        identifier, _, _ = self.fail_locally(manual=True)
        core.set_paused(self.root, True)
        self.assertEqual(local_submission.recover_oversized_requests(self.root), [self.shot()["id"]])
        self.assertTrue(core.load_project(self.root)["paused"])
        self.assertNotIn("prompt_approval", self.shot())
        self.assertEqual(core.load_project(self.root)["jobs"][identifier]["accounting"], "released")
        self.assert_no_calls()

    def test_run_lock_and_existing_unsettled_jobs_prevent_recovery(self):
        identifier, _, before = self.fail_locally()
        with core.run_lock(self.root):
            with self.assertRaises(core.GateError):
                local_submission.recover_oversized_requests(self.root)
        with core.transaction(self.root) as state:
            state["jobs"]["synthetic-other-in-flight"] = {"id": "synthetic-other-in-flight", "kind": "video",
                "item_id": "synthetic-other", "state": "submitted", "accounting": "reserved", "amount": 1,
                "task_id": "synthetic-provider-task"}
        self.assertEqual(local_submission.recover_oversized_requests(self.root), [])
        self.assertEqual(core.load_project(self.root)["jobs"][identifier], before["jobs"][identifier])
        self.assertEqual(self.shot(), before["episodes"][0]["shots"][0])
        self.assert_no_calls()

    def test_already_locked_mode_uses_callers_lock_and_spent_callback_protection_stays(self):
        identifier, _, _ = self.fail_locally()
        with self.assertRaises(core.GateError):
            core.update_job(self.root, identifier, accounting="released")
        with core.run_lock(self.root):
            self.assertEqual(local_submission.recover_oversized_requests(self.root, already_locked=True), [self.shot()["id"]])
        self.assert_no_calls()


if __name__ == "__main__":
    unittest.main()
