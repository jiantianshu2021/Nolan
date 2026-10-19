"""Synthetic durable shot/cut progress and media exposure tests; no providers."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from norlan import core, dashboard, pipeline


class DashboardGenerationTests(unittest.TestCase):
    def setUp(self):
        self.state = {"id": "synthetic", "title": "合成进度测试", "status": "generating", "paused": False,
                      "plan": {"status": "approved", "version": 1}, "assets": [],
                      "budget": {"limit": 0, "currency": "CNY"},
                      "jobs": {"job-one": {"id": "job-one", "kind": "video", "item_id": "S1", "state": "completed",
                                "accounting": "spent", "amount": 0, "created_at": "2026-01-01T00:00:00+00:00",
                                "updated_at": "2026-01-01T00:00:12+00:00", "provider_raw": "SECRET-JOB"}},
                      "episodes": [{"id": "E1", "status": "generating", "reviews": [],
                                    "progress_stage": {"stage": "shot_preparation", "shot_id": "S2",
                                                       "started_at": "2026-01-01T00:00:20+00:00"},
                                    "shots": [
                                        {"id": "S1", "status": "accepted", "version": 1, "duration": 4,
                                         "file": "media/E1/S1-v1.mp4", "sha256": "a" * 64, "job_id": "job-one"},
                                        {"id": "S2", "status": "planned", "duration": 4}]}]}

    def public(self, time="2026-01-01T00:00:25+00:00"):
        with patch("norlan.core.now", return_value=time):
            return dashboard.public_state(self.state)

    def cut(self, count=1):
        shots = self.state["episodes"][0]["shots"][:count]
        key = hashlib.sha256("-".join(shot["sha256"] for shot in shots).encode()).hexdigest()
        return {"key": key, "file": f"exports/E1-rough-{count}-{key[:16]}.mp4", "sha256": "f" * 64,
                "shot_count": count, "shots": [{"id": shot["id"], "version": shot["version"], "sha256": shot["sha256"]}
                                               for shot in shots], "created_at": "2026-01-01T00:00:15+00:00"}

    def test_finished_shot_stays_visible_with_frozen_task_duration_and_next_version(self):
        before = copy.deepcopy(self.state)
        first, second = self.public()["episodes"][0]["shots"]
        self.assertEqual(first["file"], "media/E1/S1-v1.mp4")
        self.assertEqual(first["progress"]["stage"], "accepted")
        self.assertTrue(first["progress"]["completed"])
        self.assertTrue(first["progress"]["accepted"])
        self.assertEqual(first["progress"]["elapsed_seconds"], 12)
        self.assertEqual(second["version"], 0)
        self.assertEqual(second["progress"]["target_version"], 1)
        self.assertEqual(second["progress"]["stage"], "shot_preparation")
        self.assertEqual(second["progress"]["elapsed_seconds"], 5)
        later = self.public("2026-01-01T00:01:25+00:00")["episodes"][0]["shots"]
        self.assertEqual(later[0]["progress"]["elapsed_seconds"], 12)
        self.assertEqual(later[1]["progress"]["elapsed_seconds"], 65)
        self.assertEqual(self.state, before, "public projection must not mutate production state")
        self.assertNotIn("SECRET-JOB", json.dumps(self.public()))

    def test_provider_phases_are_descriptions_not_invented_percent(self):
        shot = self.state["episodes"][0]["shots"][1]
        shot.update(status="generating", version=2, job_id="job-two")
        job = {"id": "job-two", "kind": "video", "item_id": "S2", "accounting": "reserved", "amount": 0,
               "created_at": "2026-01-01T00:00:21+00:00", "updated_at": "2026-01-01T00:00:23+00:00"}
        self.state["jobs"]["job-two"] = job
        for raw, phase in (("submitting", "shot_submission"), ("submitted", "shot_generation"),
                           ("download_pending", "shot_download"), ("completed", "shot_checks")):
            job["state"] = raw
            progress = self.public()["episodes"][0]["shots"][1]["progress"]
            self.assertEqual(progress["stage"], phase)
            self.assertEqual(progress["version"], 2)
            self.assertEqual(progress["target_version"], 2)
            self.assertNotIn("percent", progress)
        shot.update(status="review_required", file="media/E1/S2-v2.mp4", sha256="b" * 64)
        progress = self.public()["episodes"][0]["shots"][1]["progress"]
        self.assertEqual(progress["stage"], "review_required")
        self.assertTrue(progress["waiting_for_human"])
        self.assertTrue(progress["completed"])
        self.assertFalse(progress["accepted"])

    def test_failure_diagnostics_are_bound_sanitized_and_never_become_review(self):
        shot = self.state["episodes"][0]["shots"][1]
        shot.update(version=1, status="generation_failed", job_id="failed-one", generation_error="PRIVATE-ERROR",
                    attempts=[{"job_id": "failed-one", "version": 1}])
        job = {"kind": "video", "item_id": "S2", "state": "failed", "accounting": "released", "amount": 0,
               "failure": {"stage": "submission", "message": "合成错误：参数不支持",
                           "provider_code": "InvalidParameter", "parameter": "resolution", "http_status": 400,
                           "request_id": "synthetic-request-001", "provider_message": "size exceeds limit; token=synthetic-secret",
                           "provider_raw": "PRIVATE-RESPONSE"}}
        self.state["jobs"]["failed-one"] = job
        self.state["status"] = "generation_failed"
        before = copy.deepcopy(self.state)
        result = self.public()
        public = result["episodes"][0]["shots"][1]
        self.assertEqual(result["status"], "generation_failed")
        self.assertEqual(public["progress"]["stage"], "failed")
        self.assertEqual(result["episodes"][0]["progress"]["stage"], "failed")
        self.assertEqual(public["failure"]["provider_code"], "InvalidParameter")
        self.assertEqual(public["failure"]["request_id"], "synthetic-request-001")
        self.assertTrue(public["can_retry"])
        self.assertNotIn("review", public)
        self.assertNotIn("synthetic-secret", json.dumps(result))
        self.assertNotIn("PRIVATE-", json.dumps(result))
        self.assertEqual(self.state, before)
        job.update(state="uncertain", accounting="unknown")
        public = self.public()["episodes"][0]["shots"][1]
        self.assertEqual(public["progress"]["stage"], "unknown")
        self.assertTrue(public["failure"]["uncertain"])
        self.assertFalse(public["can_retry"])

    def test_legacy_failure_and_new_attempt_keep_diagnostics_separate(self):
        shot = self.state["episodes"][0]["shots"][1]
        shot.update(version=1, status="review_required", job_id="failed-one", generation_error="PRIVATE-OLD",
                    attempts=[{"job_id": "failed-one", "version": 1}])
        self.state["jobs"]["failed-one"] = {"kind": "video", "item_id": "S2", "state": "failed",
                                                   "accounting": "released", "amount": 0}
        self.state["status"] = "review_required"
        public = self.public()
        self.assertEqual(public["status"], "generation_failed")
        current = public["episodes"][0]["shots"][1]
        self.assertEqual(current["status"], "generation_failed")
        self.assertIn("旧版本未保存具体失败原因", current["failure"]["message"])
        self.assertTrue(current["can_retry"])
        shot.update(version=2, status="generating", job_id="new-one")
        self.state["jobs"]["new-one"] = {"kind": "video", "item_id": "S2", "state": "submitted",
                                                "accounting": "reserved", "amount": 0}
        current = self.public()["episodes"][0]["shots"][1]
        self.assertNotIn("failure", current)
        self.assertNotIn("generation_error", current)
        self.assertEqual(current["progress"]["stage"], "shot_generation")
        self.assertEqual(current["failure_history"][0]["version"], 1)
        self.assertIn("旧版本未保存具体失败原因", current["failure_history"][0]["failure"]["message"])

    def test_wrong_shot_job_and_unsettled_job_cannot_authorize_retry(self):
        shot = self.state["episodes"][0]["shots"][1]
        shot.update(version=1, status="generation_failed", job_id="failed-one", generation_error="legacy")
        job = {"kind": "video", "item_id": "OTHER", "state": "failed", "accounting": "released", "amount": 0,
               "failure": {"message": "OTHER-SHOT-ERROR"}}
        self.state["jobs"]["failed-one"] = job
        current = self.public()["episodes"][0]["shots"][1]
        self.assertFalse(current["can_retry"])
        self.assertNotIn("OTHER-SHOT-ERROR", json.dumps(current))
        job["item_id"] = "S2"
        self.state["jobs"]["unknown-job"] = {"kind": "image", "state": "uncertain", "accounting": "unknown", "amount": 0}
        self.assertFalse(self.public()["episodes"][0]["shots"][1]["can_retry"])

    def test_known_task_query_or_download_failure_only_allows_recovery(self):
        shot = self.state["episodes"][0]["shots"][1]
        shot.update(version=1, status="generating", job_id="recoverable", attempts=[{"job_id": "recoverable", "version": 1}])
        job = {"kind": "video", "item_id": "S2", "task_id": "synthetic-existing-task", "accounting": "spent", "amount": 0}
        self.state["jobs"]["recoverable"] = job
        for stage, status in (("poll", "submitted"), ("download", "download_pending")):
            job.update(state=status, failure={"stage": stage, "message": "合成测试：恢复原任务暂时失败"})
            public = self.public()["episodes"][0]["shots"][1]
            self.assertEqual(public["failure"]["stage"], stage)
            self.assertTrue(public["failure_recoverable"])
            self.assertFalse(public["can_retry"])
            self.assertNotEqual(public["status"], "generation_failed")
            self.assertTrue(public["progress"]["has_active_task"])
        job.update(state="completed", failure_history=[job["failure"]], failure=None)
        public = self.public()["episodes"][0]["shots"][1]
        self.assertNotIn("failure", public)
        self.assertEqual(public["failure_history"][0]["failure"]["stage"], "download")

    def test_subshot_schedule_and_proposal_are_public_without_private_metadata(self):
        episode = self.state["episodes"][0]
        episode["shots"][0].update(duration=10, subshots=[
            {"start": 0, "end": 5, "prompt": "DEMO 全景", "provider_raw": "SECRET-SUBSHOT"},
            {"start": 5, "end": 10, "prompt": "DEMO 近景"}])
        episode["segment_proposal"] = {"id": "proposal-one", "status": "candidate", "source_ids": ["S2"],
            "base_hash": "SECRET-BASE", "shots": [{"id": "G1", "duration": 10, "prompt": "DEMO", "dialogue": "",
            "source_shot_ids": ["S2"], "private": "SECRET-PROPOSED", "subshots": [
                {"start": 0, "end": 10, "prompt": "DEMO 单镜", "private": "SECRET-SCHEDULE"}]}]}
        result = self.public()["episodes"][0]
        self.assertEqual(result["progress"]["total_segments"], 2)
        self.assertEqual(result["progress"]["completed_segments"], 1)
        self.assertEqual(result["progress"]["planned_camera_shots"], 3)
        self.assertEqual(result["progress"]["stage"], "awaiting_segment_plan")
        self.assertEqual(result["shots"][0]["subshots"][0], {"start": 0, "end": 5, "prompt": "DEMO 全景"})
        self.assertEqual(result["segment_proposal"]["shots"][0]["source_shot_ids"], ["S2"])
        self.assertNotIn("SECRET-", json.dumps(result))
        episode["segment_proposal"]["status"] = "approved"
        self.assertNotIn("segment_proposal", self.public()["episodes"][0])

    def test_previous_version_is_history_not_mislabelled_as_new_generation(self):
        shot = self.state["episodes"][0]["shots"][1]
        shot.update(version=2, file="media/E1/S2-v1.mp4", sha256="b" * 64, history=[
            {"file": "media/E1/S2-v1.mp4", "sha256": "b" * 64, "provider_raw": "SECRET-HISTORY"}])
        result = self.public()["episodes"][0]["shots"][1]
        self.assertNotIn("file", result)
        self.assertFalse(result["progress"]["completed"])
        self.assertEqual(result["history"], [{"file": "media/E1/S2-v1.mp4", "version": 1}])
        self.assertNotIn("SECRET-HISTORY", json.dumps(result))

    def test_cut_manifest_currentness_is_bound_to_versions_and_hashes(self):
        episode = self.state["episodes"][0]
        episode["rough_cuts"] = [self.cut()]
        exposed = self.public()["episodes"][0]["rough_cuts"][0]
        self.assertTrue(exposed["current"])
        self.assertFalse(exposed["legacy"])
        self.assertEqual(exposed["shots"], [{"id": "S1", "version": 1}])
        episode["shots"][0].update(version=2, file="media/E1/S1-v2.mp4")
        self.assertFalse(self.public()["episodes"][0]["rough_cuts"][0]["current"])
        episode["shots"][0].update(version=1, file="media/E1/S1-v1.mp4", sha256="c" * 64)
        self.assertFalse(self.public()["episodes"][0]["rough_cuts"][0]["current"])

    def test_pending_preview_updates_approval_label_without_changing_saved_cut(self):
        episode = self.state["episodes"][0]
        first, second = episode["shots"]
        first.update(approved_at="2026-01-01T00:00:15+00:00", approved_sha256=first["sha256"],
                     review={"human_override": True})
        second.update(status="reviewing", version=1, file="media/E1/S2-v1.mp4", sha256="b" * 64)
        cut = self.cut(2)
        cut["pending_shot_ids"] = ["SYNTHETIC-STALE-ID"]
        episode["rough_cuts"] = [cut]
        before = copy.deepcopy(cut)
        for status in ("reviewing", "review_required"):
            second["status"] = status
            preview = self.public()["episodes"][0]["rough_cuts"][0]
            self.assertTrue(preview["current"])
            self.assertEqual(preview["pending_shot_ids"], ["S2"])
        second.update(status="accepted", approved_at="2026-01-01T00:00:25+00:00", approved_sha256=second["sha256"],
                      review={"human_override": True})
        self.assertEqual(self.public()["episodes"][0]["rough_cuts"][0]["pending_shot_ids"], [])
        self.assertEqual(cut, before)

    def test_upstream_revision_retires_old_preview_until_entire_prefix_exists(self):
        episode = self.state["episodes"][0]
        first, second = episode["shots"]
        second.update(status="review_required", version=1, file="media/E1/S2-v1.mp4", sha256="b" * 64)
        old = self.cut(2)
        episode["rough_cuts"] = [old]
        first.update(status="planned")
        first.pop("file")
        retired = self.public()["episodes"][0]["rough_cuts"][0]
        self.assertFalse(retired["current"])
        self.assertEqual(retired["pending_shot_ids"], [])
        first.update(status="review_required", version=2, file="media/E1/S1-v2.mp4", sha256="d" * 64)
        new = self.cut(2)
        episode["rough_cuts"].append(new)
        previews = self.public()["episodes"][0]["rough_cuts"]
        self.assertFalse(previews[0]["current"])
        self.assertTrue(previews[1]["current"])
        self.assertEqual(previews[1]["pending_shot_ids"], ["S1", "S2"])
        second["status"] = "planned"
        self.assertFalse(self.public()["episodes"][0]["rough_cuts"][1]["current"])

    def test_preview_failure_is_safe_and_does_not_invent_a_cut(self):
        self.state["episodes"][0]["preview_error"] = "PRIVATE-FFMPEG-DETAIL"
        result = self.public()["episodes"][0]
        self.assertIn("片段已保留", result["preview_error"])
        self.assertEqual(result["rough_cuts"], [])
        self.assertNotIn("PRIVATE-FFMPEG-DETAIL", json.dumps(result))

    def test_legacy_cut_verified_by_prefix_hash_or_marked_historical(self):
        episode = self.state["episodes"][0]
        legacy = self.cut()
        for key in ("shots", "shot_count", "created_at"):
            legacy.pop(key)
        legacy["provider_raw"] = "SECRET-REVIEW"
        episode["reviews"] = [legacy]
        cut = self.public()["episodes"][0]["rough_cuts"][0]
        self.assertTrue(cut["legacy"])
        self.assertTrue(cut["current"])
        self.assertEqual(cut["shot_count"], 1)
        episode["shots"][0]["sha256"] = "c" * 64
        historical = self.public()["episodes"][0]["rough_cuts"][0]
        self.assertFalse(historical["current"])
        self.assertEqual(historical["shots"], [])
        self.assertNotIn("SECRET-REVIEW", json.dumps(historical))
        episode["rough_cuts"] = [self.cut()]
        episode["reviews"].append(copy.deepcopy(episode["rough_cuts"][0]))
        self.assertEqual(len(self.public()["episodes"][0]["rough_cuts"]), 2)

    def test_only_whitelisted_cut_and_history_files_become_accessible(self):
        episode = self.state["episodes"][0]
        cut = self.cut()
        episode["rough_cuts"] = [cut, {**cut, "file": "exports/../../source/secret.mp4"},
                                 {**cut, "file": "source/secret.mp4"}]
        episode["shots"][0]["history"] = [{"file": "media/E1/S1-v9.mp4", "version": 9, "sha256": "d" * 64,
                                            "created_at": {"secret": "SECRET-TIME"}},
                                           {"file": "source/secret.mp4", "sha256": "e" * 64}]
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp).resolve()
            for file in (cut["file"], "media/E1/S1-v9.mp4", "source/secret.mp4", "exports/unreferenced.mp4"):
                path = root / file
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"synthetic local fixture")
            self.assertIsNotNone(dashboard.resolve_media(root, cut["file"], self.state))
            self.assertIsNotNone(dashboard.resolve_media(root, "media/E1/S1-v9.mp4", self.state))
            self.assertIsNone(dashboard.resolve_media(root, "source/secret.mp4", self.state))
            self.assertIsNone(dashboard.resolve_media(root, "exports/unreferenced.mp4", self.state))
        exposed = self.public()["episodes"][0]
        self.assertEqual(len(exposed["rough_cuts"]), 1)
        self.assertNotIn("SECRET-TIME", json.dumps(exposed))

    def test_external_run_lock_not_saved_hint_controls_actual_worker_state(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp).resolve()
            core.create_project(root, "合成本地锁测试")
            core.install_plan(root, pipeline.demo_plan())
            context = dashboard._Context(root)
            self.assertFalse(context.state()["dashboard"]["running"])
            self.assertFalse((root / ".run.lock").exists(), "read-only probe must not create lock file")
            with core.transaction(root) as state:
                state["episodes"][0]["progress_stage"] = {"stage": "shot_generation", "shot_id": "E001-S001"}
            self.assertFalse(context.state()["dashboard"]["running"])
            with core.run_lock(root):
                self.assertTrue(context.state()["dashboard"]["running"])
            self.assertFalse(context.state()["dashboard"]["running"])
            context.running = True
            self.assertTrue(context.state()["dashboard"]["running"])


if __name__ == "__main__":
    unittest.main()
