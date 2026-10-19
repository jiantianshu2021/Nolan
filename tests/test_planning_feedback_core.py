"""Synthetic offline human feedback for cached planning blockers; no services."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from norlan import core, pipeline


class PlanningFeedbackCoreTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        core.create_project(self.root, "合成规划反馈测试小说，不调用任何真实服务。")
        core.install_plan(self.root, pipeline.demo_plan())
        core.approve(self.root, "plan", "plan", 1)
        core.configure(self.root, limit=100)
        first = self.artifact("media/E001/E001-S001-v1.mp4", b"synthetic accepted first video")
        core.reserve(self.root, "existing-video", "video", "E001-S001", 10)
        core.update_job(self.root, "existing-video", state="completed", accounting="spent", task_id="synthetic-old-task", output=first["file"])
        with core.transaction(self.root) as state:
            for asset in state["assets"]:
                record = self.artifact("media/assets/" + asset["id"] + ".png", b"synthetic approved image")
                asset.update(record, status="approved", version=1, approved_at=core.now(), approved_sha256=record["sha256"])
            core._find(state, "shot", "E001-S001").update(first, status="review_required", version=1, job_id="existing-video")
        core.approve(self.root, "shot", "E001-S001", 1)
        self.shot_id = "E001-S002"
        self.old_feedback = "每次任务框出现时需要固定女性播报；保留当前镜头时长。"
        self.revision = 2
        self.signature = self.analysis_signature(self.revision, self.old_feedback)
        self.analysis = {"signature": self.signature, "summary": "播报内容与既有镜头切点存在安排问题",
                         "blockers": ["请明确任务框出现时间，以及播报是否允许跨越内部切镜点。"],
                         "original_model_metadata": {"synthetic": True}}
        with core.transaction(self.root) as state:
            core._find(state, "shot", self.shot_id).update(version=0, revision=self.revision,
                feedback=self.old_feedback, revision_analysis=copy.deepcopy(self.analysis))
            state.update(status="blocked", last_error=self.analysis["blockers"][0])
        core.reserve(self.root, "lost-advisory", "llm", "review", 5)
        core.update_job(self.root, "lost-advisory", state="uncertain", accounting="unknown",
                        advisory_recovery={"status": "interrupted", "no_retry": True})

    @staticmethod
    def analysis_signature(revision, feedback):
        return hashlib.sha256(json.dumps([revision, feedback], ensure_ascii=False).encode()).hexdigest()

    def artifact(self, relative, body):
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(body)
        return {"file": relative, "sha256": core.digest(path)}

    def submit(self, feedback="任务框在第 1 秒出现，播报可以跨越内部切镜，但保持原定总时长。", version=0, signature=None):
        return core.submit_planning_feedback(self.root, self.shot_id, feedback, version,
                                             self.signature if signature is None else signature)

    def test_human_feedback_is_appended_and_old_blocker_is_archived_without_paid_work(self):
        original = core.load_project(self.root)
        with patch.object(pipeline, "_brain", side_effect=AssertionError("No GPT calls")) as brain, \
                patch.object(pipeline, "_ark", side_effect=AssertionError("No provider calls")) as ark, \
                patch.object(core, "reserve", side_effect=AssertionError("No new jobs")) as reserve:
            state = self.submit()
        shot = core._find(state, "shot", self.shot_id)
        self.assertEqual((shot["status"], shot["version"], shot["revision"]), ("planned", 0, 3))
        self.assertTrue(shot["feedback"].startswith(self.old_feedback))
        self.assertIn("用户对规划阻塞的补充意见", shot["feedback"])
        self.assertIn("播报可以跨越内部切镜", shot["feedback"])
        self.assertNotIn("revision_analysis", shot)
        archived = shot["revision_analysis_history"][-1]
        for key, value in self.analysis.items():
            self.assertEqual(archived[key], value)
        self.assertEqual(archived["resolution"], "human_planning_feedback")
        self.assertIn("superseded_at", archived)
        self.assertEqual(state["status"], "ready")
        self.assertIsNone(state["last_error"])
        self.assertNotEqual(self.analysis_signature(shot["revision"], shot["feedback"]), self.signature)
        for key in ("assets", "jobs", "budget", "llm_calls", "plan"):
            self.assertEqual(state[key], original[key])
        self.assertEqual(state["episodes"][0]["shots"][0], original["episodes"][0]["shots"][0])
        self.assertEqual(state["episodes"][1], original["episodes"][1])
        for key in ("prompt", "dialogue", "duration", "scene_id", "asset_ids", "attempts", "history"):
            self.assertEqual(shot.get(key), core._find(original, "shot", self.shot_id).get(key))
        self.assertEqual(state["jobs"]["lost-advisory"]["accounting"], "unknown")
        brain.assert_not_called(); ark.assert_not_called(); reserve.assert_not_called()

    def test_feedback_never_chooses_an_option_on_the_users_behalf(self):
        state = self.submit("先保留原有安排，请给我两个可选方案，不要直接生成。")
        shot = core._find(state, "shot", self.shot_id)
        self.assertIn("给我两个可选方案", shot["feedback"])
        self.assertFalse({"timing_choice", "audio_schedule_decision", "prompt_approval"}.intersection(shot))
        self.assertFalse(any(event.get("action") == "human_approved_prompt" for event in state["events"]))

    def test_stale_or_duplicate_submission_is_rejected_atomically(self):
        for version, signature in ((1, self.signature), (True, self.signature), (-1, self.signature),
                                   (0, "0" * 64), (0, "bad"), (0, None)):
            before = (self.root / "project.json").read_bytes()
            with self.subTest(version=version, signature=signature), self.assertRaises((ValueError, core.GateError)):
                core.submit_planning_feedback(self.root, self.shot_id, "新的人工意见", version, signature)
            self.assertEqual((self.root / "project.json").read_bytes(), before)
        self.submit()
        before = (self.root / "project.json").read_bytes()
        with self.assertRaises(core.GateError):
            self.submit()
        self.assertEqual((self.root / "project.json").read_bytes(), before)

    def test_blockers_and_signature_must_match_current_feedback_and_revision(self):
        original = core.load_project(self.root)
        variants = [{"feedback": self.old_feedback + "另有新意见"}, {"revision": self.revision + 1},
                    {"revision_analysis": {**self.analysis, "signature": "0" * 64}},
                    {"revision_analysis": {**self.analysis, "blockers": []}},
                    {"revision_analysis": {**self.analysis, "blockers": "错误类型"}},
                    {"revision_analysis": {**self.analysis, "blockers": [" "]}},
                    {"revision_analysis": {"signature": self.signature}}]
        for changes in variants:
            with self.subTest(changes=changes):
                with core.transaction(self.root) as state:
                    shot = core._find(state, "shot", self.shot_id)
                    shot.clear(); shot.update(copy.deepcopy(core._find(original, "shot", self.shot_id))); shot.update(changes)
                before = (self.root / "project.json").read_bytes()
                with self.assertRaises(core.GateError):
                    self.submit()
                self.assertEqual((self.root / "project.json").read_bytes(), before)

    def test_nonempty_bounded_human_text_is_required(self):
        for feedback in (None, 123, "", " \n ", "字" * 8001):
            before = (self.root / "project.json").read_bytes()
            with self.subTest(feedback_type=type(feedback).__name__), self.assertRaises(ValueError):
                self.submit(feedback=feedback)
            self.assertEqual((self.root / "project.json").read_bytes(), before)

    def test_submitted_generated_and_prompt_approved_states_cannot_be_rewritten(self):
        for status in ("generating", "review_required", "accepted", "awaiting_prompt_approval", "prompt_approved", "generation_failed"):
            with self.subTest(status=status):
                with core.transaction(self.root) as state:
                    core._find(state, "shot", self.shot_id)["status"] = status
                before = (self.root / "project.json").read_bytes()
                with self.assertRaises(core.GateError):
                    self.submit()
                self.assertEqual((self.root / "project.json").read_bytes(), before)

    def test_any_current_media_file_or_job_binding_blocks_planning_feedback(self):
        for extra in ({"file": "media/current.mp4"}, {"job_id": "existing-video"}):
            with self.subTest(extra=extra):
                with core.transaction(self.root) as state:
                    shot = core._find(state, "shot", self.shot_id)
                    shot.pop("file", None); shot.pop("job_id", None); shot.update(extra)
                before = (self.root / "project.json").read_bytes()
                with self.assertRaises(core.GateError):
                    self.submit()
                self.assertEqual((self.root / "project.json").read_bytes(), before)

    def test_unknown_media_or_active_nonreview_llm_blocks_feedback(self):
        for kind, item_id, status, accounting in (("video", self.shot_id, "submitted", "reserved"),
                ("video", self.shot_id, "uncertain", "unknown"), ("llm", "plan_revision", "submitting", "reserved"),
                ("llm", "refine_shot", "uncertain", "unknown")):
            with self.subTest(kind=kind, status=status):
                with core.transaction(self.root) as state:
                    state["jobs"]["other-active-job"] = {"id": "other-active-job", "kind": kind, "item_id": item_id,
                        "state": status, "accounting": accounting, "amount": 0, "task_id": "synthetic-task" if kind == "video" else None}
                before = (self.root / "project.json").read_bytes()
                with self.assertRaises(core.GateError):
                    self.submit()
                self.assertEqual((self.root / "project.json").read_bytes(), before)

    def test_pending_amendment_affecting_this_shot_must_be_handled_first(self):
        for status in ("candidate", "awaiting_assets"):
            with self.subTest(status=status):
                with core.transaction(self.root) as state:
                    state["asset_amendments"] = [{"id": "pending-amendment", "status": status,
                        "target_shot_id": "other-shot", "shot_updates": [{"id": self.shot_id}]}]
                with self.assertRaises(core.GateError):
                    self.submit()

    def test_pause_is_preserved_and_running_worker_lock_is_respected(self):
        core.set_paused(self.root, True)
        state = self.submit()
        self.assertTrue(state["paused"])
        self.assertEqual(state["status"], "paused")
        with core.transaction(self.root) as current:
            shot = core._find(current, "shot", self.shot_id)
            signature = self.analysis_signature(shot["revision"], shot["feedback"])
            shot["revision_analysis"] = {**self.analysis, "signature": signature}
        before = (self.root / "project.json").read_bytes()
        with core.run_lock(self.root), self.assertRaises(core.GateError):
            self.submit(signature=signature)
        self.assertEqual((self.root / "project.json").read_bytes(), before)

    def test_prepared_prompt_consent_is_archived_and_cleared_without_touching_media_history(self):
        historical = self.artifact("media/E001/E001-S002-v1.mp4", b"synthetic historical immutable video")
        old_candidate = {"id": "old-prompt", "status": "approved", "request_hash": "a" * 64}
        with core.transaction(self.root) as state:
            shot = core._find(state, "shot", self.shot_id)
            shot.update(version=1, prompt_candidate=copy.deepcopy(old_candidate), prompt_approval={"candidate_id": "old-prompt"},
                        refined_prompt={"prompt": "旧草稿"}, history=[{**historical, "version": 1}])
        state = self.submit(version=1)
        shot = core._find(state, "shot", self.shot_id)
        self.assertFalse({"prompt_candidate", "prompt_approval", "refined_prompt"}.intersection(shot))
        self.assertEqual(shot["prompt_history"][-1]["id"], old_candidate["id"])
        self.assertEqual(shot["prompt_history"][-1]["status"], "invalidated")
        self.assertEqual(shot["history"], [{**historical, "version": 1}])
        self.assertEqual(core.digest(self.root / historical["file"]), historical["sha256"])

    def test_never_generated_legacy_shot_without_version_uses_zero_not_revision_as_version(self):
        with core.transaction(self.root) as state:
            core._find(state, "shot", self.shot_id).pop("version")
        state = self.submit(version=0)
        self.assertEqual(core._find(state, "shot", self.shot_id)["revision"], 3)


if __name__ == "__main__":
    unittest.main()
