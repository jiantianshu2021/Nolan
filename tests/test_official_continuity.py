"""Synthetic offline regressions for official Ark continuity assets.

These tests use conspicuously synthetic bytes. Every model operation, media
download, and GPT operation is mocked; no paid provider is contacted.
"""

import copy
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import Mock, patch

from norlan import core, pipeline
from norlan.billing import VIDEO_MODEL
from norlan.providers import ArkProvider, ProviderError
from test_pipeline import run_confirming_prompts, choose_gpt_assets, fake_media, write


class OfficialContinuityTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.enterContext(fake_media())
        self.enterContext(patch("norlan.providers._api_open", side_effect=AssertionError("Real API IO is forbidden")))
        self.enterContext(patch("norlan.providers.urlopen", side_effect=AssertionError("Real media IO is forbidden")))
        core.create_project(self.root, "合成官方连续性测试，不调用真实生成服务。")
        core.install_plan(self.root, pipeline.demo_plan())
        core.approve(self.root, "plan", "plan", 1)
        choose_gpt_assets(self.root)
        state = run_confirming_prompts(self.root)
        for asset in state["assets"]:
            core.approve(self.root, "asset", asset["id"], asset["version"])
        core.configure(self.root, {"ark": {"video_model": VIDEO_MODEL, "resolution": "720p"},
                                   "caps": {"video_second": 1}}, limit=100)
        with core.transaction(self.root) as state:
            state["mode"] = "live"
        self.created_at = int(time.time())
        self.with_tail = True
        self.download_tail_error = False
        self.video_bytes = {}
        self.api = Mock()
        self.api.capabilities = dict(ArkProvider.capabilities)
        self.api.submit_video.side_effect = self.submit
        self.api.poll_video.side_effect = self.artifacts
        self.api.video_artifacts.side_effect = self.artifacts
        self.api.download.side_effect = self.download
        self.enterContext(patch.object(pipeline, "_ark", return_value=self.api))
        self.brain = self.enterContext(patch.object(pipeline, "_brain", side_effect=self.fake_brain))

    @staticmethod
    def fake_brain(root, method, *args):
        if method == "plan_revision":
            return {"needs_assets": False, "additions": [], "shot_updates": [],
                    "summary": "合成普通返修", "blockers": []}
        if method == "refine_shot":
            return {"prompt": "SYNTHETIC continuity prompt", "dialogue": "", "continuity_notes": "synthetic"}
        return {"verdict": "pass", "issues": [], "observed_end_state": "合成角色位于左侧", "repair_prompt": ""}

    def submit(self, *args, **kwargs):
        task_id = "synthetic-task-" + str(self.api.submit_video.call_count)
        self.video_bytes[task_id] = ("SYNTHETIC ORIGINAL VIDEO " + task_id).encode()
        return {"id": task_id, "request_id": "synthetic-submit-request"}

    def artifacts(self, task_id):
        return {"id": task_id, "status": "completed", "provider_status": "succeeded",
                "url": "https://example.test/" + task_id + ".mp4?signature=SYNTHETIC-PRIVATE",
                "last_frame_url": ("https://example.test/" + task_id + ".jpg?signature=SYNTHETIC-PRIVATE") if self.with_tail else None,
                "model": VIDEO_MODEL, "created_at": self.created_at, "duration": 4,
                "resolution": "720p", "usage": {"completion_tokens": 100000},
                "request_id": "synthetic-poll-request"}

    def download(self, url, output):
        name = url.split("?", 1)[0].rsplit("/", 1)[-1]
        if name.endswith(".jpg"):
            if self.download_tail_error:
                raise ProviderError("合成测试：官方尾帧暂时下载失败。", stage="download")
            return write(output, b"\xff\xd8\xffSYNTHETIC OFFICIAL JPEG " + name.encode())
        return write(output, self.video_bytes[name.removesuffix(".mp4")])

    def shot(self, index=0):
        return core.load_project(self.root)["episodes"][0]["shots"][index]

    def job(self, index=0):
        state = core.load_project(self.root)
        return state["jobs"][self.shot(index)["job_id"]]

    def approve_first(self):
        self.assertEqual(run_confirming_prompts(self.root)["status"], "review_required")
        shot = self.shot()
        core.approve(self.root, "shot", shot["id"], shot["version"])
        return self.shot()

    def set_video_override(self, *, stale=False):
        first = self.shot()
        with core.transaction(self.root) as state:
            second = state["episodes"][0]["shots"][1]
            second["continuity_reference"] = {"mode": "original_video",
                "revision": second.get("revision", 0) - int(stale),
                "source_shot_id": first["id"], "source_sha256": first["sha256"]}

    def forget_official_tail(self):
        with core.transaction(self.root) as state:
            job = state["jobs"][state["episodes"][0]["shots"][0]["job_id"]]
            job.pop("official_tail", None)
            job.pop("official_tail_error", None)

    def test_completed_video_saves_official_originals_with_task_and_hash(self):
        self.assertEqual(run_confirming_prompts(self.root)["status"], "review_required")
        shot, job = self.shot(), self.job()
        tail, origin = job["official_tail"], job["origin_video"]
        self.assertEqual(tail["source"], "ark_official_tail")
        self.assertEqual(tail["task_id"], job["task_id"])
        self.assertEqual(tail["video_sha256"], shot["sha256"])
        self.assertEqual(tail["sha256"], core.digest(self.root / tail["file"]))
        self.assertTrue((self.root / tail["file"]).read_bytes().startswith(b"\xff\xd8\xff"))
        self.assertIn(Path(tail["file"]).suffix, {".jpg", ".jpeg"})
        self.assertNotEqual(tail["file"], shot["evidence"]["tail_frame"])
        self.assertEqual(origin["file"], shot["file"])
        self.assertEqual(origin["sha256"], shot["sha256"])
        self.assertEqual(origin["task_id"], job["task_id"])
        self.assertNotIn("SYNTHETIC-PRIVATE", (self.root / "project.json").read_text())

    def test_same_scene_uses_official_tail_and_local_frame_is_observation_only(self):
        first = self.approve_first()
        official = self.root / self.job()["official_tail"]["file"]
        local_tail = self.root / first["evidence"]["tail_frame"]
        self.assertEqual(run_confirming_prompts(self.root)["status"], "review_required")
        call = self.api.submit_video.call_args
        refs = [Path(path) for path in call.args[2]]
        self.assertIn(official, refs)
        self.assertNotIn(local_tail, refs)
        self.assertIsNone(call.kwargs.get("previous_video"))
        self.assertEqual(self.api.submit_video.call_count, 2)
        self.assertEqual(self.shot()["sha256"], first["sha256"])
        self.assertEqual(self.shot()["human_decision"], first["human_decision"])

    def test_missing_tail_gets_original_task_then_blocks_without_new_post(self):
        self.approve_first()
        self.forget_official_tail()
        self.with_tail = False
        original_jobs = set(core.load_project(self.root)["jobs"])
        state = run_confirming_prompts(self.root)
        self.assertEqual(state["status"], "blocked")
        self.api.video_artifacts.assert_called_once_with(self.job()["task_id"])
        self.api.submit_video.assert_called_once()
        self.assertEqual(set(state["jobs"]), original_jobs)
        self.assertFalse(self.shot(1).get("file"))
        self.assertEqual(self.shot()["status"], "accepted")

    def test_missing_tail_can_be_recovered_by_get_before_one_new_submission(self):
        self.approve_first()
        self.forget_official_tail()
        self.assertEqual(run_confirming_prompts(self.root)["status"], "review_required")
        self.api.video_artifacts.assert_called_once_with(self.job()["task_id"])
        tail = self.root / self.job()["official_tail"]["file"]
        self.assertIn(tail, [Path(path) for path in self.api.submit_video.call_args.args[2]])
        self.assertEqual(self.api.submit_video.call_count, 2)

    def test_changed_official_tail_is_blocked_and_never_silently_overwritten(self):
        self.approve_first()
        tail = self.root / self.job()["official_tail"]["file"]
        changed = b"\xff\xd8\xffSYNTHETIC TAMPERED TAIL"
        tail.write_bytes(changed)
        downloads = self.api.download.call_count
        state = run_confirming_prompts(self.root)
        self.assertEqual(state["status"], "blocked")
        self.api.submit_video.assert_called_once()
        self.api.video_artifacts.assert_not_called()
        self.assertEqual(self.api.download.call_count, downloads)
        self.assertEqual(tail.read_bytes(), changed)

    def test_tail_download_failure_preserves_reviewable_video_and_never_reposts(self):
        self.download_tail_error = True
        self.assertEqual(run_confirming_prompts(self.root)["status"], "review_required")
        shot, job = self.shot(), self.job()
        self.assertTrue((self.root / shot["file"]).is_file())
        self.assertEqual(job["state"], "completed")
        self.assertTrue(job.get("official_tail_error"))
        self.assertFalse(job.get("official_tail"))
        for _ in range(2):
            self.assertEqual(run_confirming_prompts(self.root)["status"], "review_required")
        self.api.submit_video.assert_called_once()
        core.approve(self.root, "shot", shot["id"], shot["version"])
        self.assertEqual(self.shot()["status"], "accepted")

    def test_original_video_reference_is_one_submission_and_uses_video_input_price(self):
        first = self.approve_first()
        original = (self.root / first["file"]).read_bytes()
        self.set_video_override()
        self.assertEqual(run_confirming_prompts(self.root)["status"], "review_required")
        call = self.api.submit_video.call_args
        self.assertEqual(call.kwargs["previous_video"], self.artifacts(self.job()["task_id"])["url"])
        refs = [Path(path) for path in call.args[2]]
        self.assertNotIn(self.root / first["evidence"]["tail_frame"], refs)
        self.assertNotIn(self.root / self.job()["official_tail"]["file"], refs)
        self.assertEqual(self.api.submit_video.call_count, 2)
        second_job = self.job(1)
        self.assertEqual(second_job["amount"], 8)
        self.assertTrue(second_job["billing"]["price"]["has_video_input"])
        self.assertEqual(second_job["billing"]["price"]["rate"], 22)
        self.assertEqual(second_job["billing"]["usage_cost"], 2.2)
        manifest = self.shot(1)["reference_manifest"]["context"]["video_references"]
        self.assertEqual(len(manifest), 1)
        self.assertEqual(manifest[0]["sha256"], first["sha256"])
        self.assertNotIn("SYNTHETIC-PRIVATE", (self.root / "project.json").read_text())
        self.assertEqual((self.root / first["file"]).read_bytes(), original)
        self.assertEqual(self.shot(), first)
        run_confirming_prompts(self.root)
        self.assertEqual(self.api.submit_video.call_count, 2)

    def test_remote_original_video_hash_mismatch_blocks_without_replacing_approved_file(self):
        first = self.approve_first()
        original = (self.root / first["file"]).read_bytes()
        self.set_video_override()
        self.video_bytes[self.job()["task_id"]] = b"SYNTHETIC MISMATCHED REMOTE VIDEO"
        self.assertEqual(run_confirming_prompts(self.root)["status"], "blocked")
        self.api.submit_video.assert_called_once()
        self.assertEqual((self.root / first["file"]).read_bytes(), original)
        self.assertEqual(self.shot(), first)

    def test_stale_revision_override_does_not_reuse_original_video_authorization(self):
        self.approve_first()
        self.set_video_override(stale=True)
        self.assertEqual(run_confirming_prompts(self.root)["status"], "review_required")
        call = self.api.submit_video.call_args
        self.assertIsNone(call.kwargs.get("previous_video"))
        official = self.root / self.job()["official_tail"]["file"]
        self.assertIn(official, [Path(path) for path in call.args[2]])
        self.assertFalse(self.job(1)["billing"]["price"]["has_video_input"])

    def test_submitted_task_resume_does_not_resolve_source_or_create_another_task(self):
        first = self.approve_first()
        self.set_video_override()
        self.api.poll_video.side_effect = ProviderError("合成测试：轮询暂时不可用。", stage="poll")
        self.assertEqual(run_confirming_prompts(self.root)["status"], "blocked")
        pending = self.job(1)
        self.assertEqual(pending["state"], "submitted")
        self.assertEqual(self.api.submit_video.call_count, 2)
        source_calls = self.api.video_artifacts.call_count
        manifest = copy.deepcopy(self.shot(1)["reference_manifest"])
        self.api.video_artifacts.side_effect = AssertionError("Resume must not re-resolve prior inputs")
        self.api.poll_video.side_effect = self.artifacts
        self.assertEqual(run_confirming_prompts(self.root)["status"], "review_required")
        self.assertEqual(self.api.submit_video.call_count, 2)
        self.assertEqual(self.api.video_artifacts.call_count, source_calls)
        self.assertEqual(self.job(1)["task_id"], pending["task_id"])
        self.assertEqual(self.shot(1)["reference_manifest"], manifest)
        self.assertEqual(self.shot()["sha256"], first["sha256"])

    def test_completed_but_unpublished_original_is_not_replaced_on_resume(self):
        self.assertEqual(run_confirming_prompts(self.root)["status"], "review_required")
        first, job = self.shot(), self.job()
        original_provenance = copy.deepcopy(job["origin_video"])
        output = self.root / first["file"]
        output.write_bytes(b"SYNTHETIC ALTERED AFTER COMPLETION")
        # Simulate a completed download whose shot publication was interrupted.
        with core.transaction(self.root) as state:
            shot = state["episodes"][0]["shots"][0]
            shot["status"] = "generating"
            for field in ("file", "sha256", "evidence", "review"):
                shot.pop(field, None)
        self.assertEqual(run_confirming_prompts(self.root)["status"], "blocked")
        self.api.submit_video.assert_called_once()
        self.assertEqual(self.job()["origin_video"], original_provenance)
        self.assertFalse(self.shot().get("file"))

    def test_expired_official_original_blocks_before_source_download_or_submission(self):
        self.approve_first()
        self.set_video_override()
        with core.transaction(self.root) as state:
            job = state["jobs"][state["episodes"][0]["shots"][0]["job_id"]]
            job["origin_video"]["created_at"] = self.created_at - 31 * 86400
        self.assertEqual(run_confirming_prompts(self.root)["status"], "blocked")
        self.api.submit_video.assert_called_once()
        self.api.video_artifacts.assert_not_called()

    def test_expired_official_tail_blocks_without_new_post(self):
        self.approve_first()
        with core.transaction(self.root) as state:
            job = state["jobs"][state["episodes"][0]["shots"][0]["job_id"]]
            job["official_tail"]["created_at"] = self.created_at - 31 * 86400
        self.assertEqual(run_confirming_prompts(self.root)["status"], "blocked")
        self.api.submit_video.assert_called_once()
        self.api.video_artifacts.assert_not_called()

    def test_missing_observation_tail_is_not_claimed_in_gpt_review_scope(self):
        core.set_video_review_enabled(self.root, True)
        first = self.approve_first()
        self.set_video_override()
        (self.root / first["evidence"]["tail_frame"]).unlink()
        self.assertEqual(run_confirming_prompts(self.root)["status"], "review_required")
        context = [call.args[2] for call in self.brain.call_args_list if call.args[1] == "review"][-1]
        scope = context["review_scope"]
        self.assertNotIn("与前镜实际尾帧", scope)
        self.assertNotIn("包含实测首帧、最后一帧与前段实际尾帧参考", scope)
        self.assertTrue(any(word in scope for word in ("缺失", "未提供", "缺少", "不可用", "没有前", "未包含", "未获得")), scope)
        self.assertEqual(self.api.submit_video.call_count, 2)

    def test_revised_previous_segment_uses_its_own_tail_and_keeps_old_official_file(self):
        first = self.approve_first()
        old_job = copy.deepcopy(self.job())
        old_tail = self.root / old_job["official_tail"]["file"]
        old_bytes = old_tail.read_bytes()
        core.reject(self.root, "shot", first["id"], first["version"], "合成用户授权：修正站位一次。")
        self.assertEqual(run_confirming_prompts(self.root)["status"], "review_required")
        revised = self.shot()
        self.assertEqual(revised["version"], first["version"] + 1)
        new_tail = self.root / self.job()["official_tail"]["file"]
        self.assertNotEqual(old_tail, new_tail)
        core.approve(self.root, "shot", revised["id"], revised["version"])
        self.assertEqual(run_confirming_prompts(self.root)["status"], "review_required")
        refs = [Path(path) for path in self.api.submit_video.call_args.args[2]]
        self.assertIn(new_tail, refs)
        self.assertNotIn(old_tail, refs)
        self.assertEqual(old_tail.read_bytes(), old_bytes)
        self.assertEqual(core.load_project(self.root)["jobs"][old_job["id"]], old_job)

    def test_override_bound_to_a_different_approved_version_is_blocked(self):
        self.approve_first()
        self.set_video_override()
        with core.transaction(self.root) as state:
            state["episodes"][0]["shots"][1]["continuity_reference"]["source_sha256"] = "0" * 64
        self.assertEqual(run_confirming_prompts(self.root)["status"], "blocked")
        self.api.submit_video.assert_called_once()
        self.api.video_artifacts.assert_not_called()

    def test_insufficient_budget_for_input_plus_output_blocks_before_post(self):
        self.approve_first()
        self.set_video_override()
        # First output reserves four. Six remain: enough for output alone,
        # insufficient for the four-second reference plus four-second output.
        core.configure(self.root, {}, limit=10)
        self.assertEqual(run_confirming_prompts(self.root)["status"], "blocked")
        self.api.submit_video.assert_called_once()
        self.assertFalse(self.shot(1).get("job_id"))

    def test_demo_keeps_explicit_synthetic_tail_behavior_without_provider_calls(self):
        with core.transaction(self.root) as state:
            state["mode"] = "demo"
        first = self.approve_first()
        self.assertEqual(run_confirming_prompts(self.root)["status"], "review_required")
        context = self.shot(1)["reference_manifest"]["context"]
        self.assertEqual(context["image_references"][-1]["role"], "previous_actual_tail")
        self.assertEqual(context["previous_actual"]["sha256"], first["sha256"])
        self.api.submit_video.assert_not_called()
        self.api.video_artifacts.assert_not_called()
        self.brain.assert_not_called()


if __name__ == "__main__":
    unittest.main()
