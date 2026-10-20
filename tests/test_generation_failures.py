"""Synthetic, offline regression tests for durable video failure diagnosis.

Every provider and GPT operation is mocked. Re-entering the pipeline must never
turn a failure, an unknown POST outcome, or a GET recovery into another POST.
"""
import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from norlan import core, pipeline
from norlan.providers import ProviderError
from test_pipeline import run_confirming_prompts, choose_gpt_assets, fake_media, write


class GenerationFailureTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.enterContext(fake_media())
        core.create_project(self.root, "合成故障诊断测试，不调用真实服务。")
        core.install_plan(self.root, pipeline.demo_plan())
        core.approve(self.root, "plan", "plan", 1)
        choose_gpt_assets(self.root)
        state = run_confirming_prompts(self.root)
        for asset in state["assets"]:
            core.approve(self.root, "asset", asset["id"], asset["version"])
        core.configure(self.root, {"caps": {"video_second": 1}}, limit=100)
        with core.transaction(self.root) as state:
            state["mode"] = "live"
        self.api = Mock()
        self.api.submit_video.side_effect = lambda *a, **kw: {
            "id": "synthetic-" + str(self.api.submit_video.call_count),
            "request_id": "synthetic-submit-request"}
        self.api.poll_video.return_value = {
            "status": "completed", "url": "https://example.test/synthetic.mp4",
            "request_id": "synthetic-poll-request"}
        self.api.download.side_effect = lambda url, output: write(output, b"SYNTHETIC " + str(output).encode())
        self.enterContext(patch.object(pipeline, "_ark", return_value=self.api))
        self.brain = self.enterContext(patch.object(pipeline, "_brain", side_effect=self.fake_brain))

    @staticmethod
    def fake_brain(root, method, *args, **kwargs):
        if method == "plan_revision":
            return {"needs_assets": False, "additions": [], "shot_updates": [],
                    "summary": "合成普通返修", "blockers": []}
        if method == "refine_shot":
            return {"prompt": "SYNTHETIC", "dialogue": "", "continuity_notes": "synthetic"}
        return {"verdict": "pass", "issues": [], "observed_end_state": "合成角色在左侧", "repair_prompt": ""}

    def shot(self, index=0):
        return core.load_project(self.root)["episodes"][0]["shots"][index]

    def job(self):
        state = core.load_project(self.root)
        return state["jobs"][self.shot()["job_id"]]

    def rejection(self, **updates):
        failure = {"stage": "submission", "message": "合成测试：声音参考不受支持。",
                   "provider_code": "InvalidParameter.Reference", "http_status": 400,
                   "request_id": "synthetic-request-400", "parameter": "content.audio_url",
                   "uncertain": False}
        failure.update(updates)
        return ProviderError(failure["message"], uncertain=failure["uncertain"],
                             request_id=failure["request_id"], failure=failure)

    def fail_submission(self):
        self.api.submit_video.side_effect = self.rejection()
        return run_confirming_prompts(self.root)

    def assert_not_approvable(self):
        shot = self.shot()
        self.assertFalse(shot.get("file"))
        with self.assertRaises(core.GateError):
            core.approve(self.root, "shot", shot["id"], shot["version"])

    def test_definite_submission_failure_persists_specific_diagnosis_and_cannot_approve(self):
        state = self.fail_submission()
        self.assertEqual(state["status"], "generation_failed")
        self.assertEqual(self.shot()["status"], "generation_failed")
        job = self.job()
        self.assertEqual(job["state"], "failed")
        self.assertFalse(job.get("task_id"))
        self.assertEqual(job["failure"]["stage"], "submission")
        self.assertEqual(job["failure"]["provider_code"], "InvalidParameter.Reference")
        self.assertEqual(job["failure"]["http_status"], 400)
        self.assertEqual(job["failure"]["request_id"], "synthetic-request-400")
        self.assertEqual(job["failure"]["parameter"], "content.audio_url")
        self.assertIn("声音参考不受支持", job["failure"]["message"])
        self.assert_not_approvable()
        self.api.poll_video.assert_not_called()
        self.assertEqual([call.args[1] for call in self.brain.call_args_list], ["refine_shot"])

    def test_repeated_resume_preserves_failed_attempt_and_makes_no_paid_calls(self):
        state = self.fail_submission()
        jobs = copy.deepcopy(state["jobs"])
        calls = self.brain.call_count
        for _ in range(3):
            self.assertEqual(run_confirming_prompts(self.root)["status"], "generation_failed")
        self.assertEqual(core.load_project(self.root)["jobs"], jobs)
        self.assertEqual(self.brain.call_count, calls)
        self.api.submit_video.assert_called_once()
        self.assertEqual(self.shot(1).get("version", 0), 0)

    def test_explicit_rejection_authorizes_one_retry_only_and_keeps_original_diagnosis(self):
        self.fail_submission()
        first_job = copy.deepcopy(self.job())
        shot = self.shot()
        core.reject(self.root, "shot", shot["id"], shot["version"], "合成授权：只重试一次")
        self.assertEqual(self.shot()["revision"], 1)
        self.assertEqual(run_confirming_prompts(self.root)["status"], "generation_failed")
        for _ in range(2):
            run_confirming_prompts(self.root)
        current = core.load_project(self.root)
        self.assertEqual(self.api.submit_video.call_count, 2)
        self.assertEqual(self.shot()["version"], 2)
        self.assertNotEqual(self.job()["id"], first_job["id"])
        self.assertEqual(current["jobs"][first_job["id"]], first_job)
        self.assert_not_approvable()

    def test_unknown_submission_remains_blocked_and_reject_cannot_resubmit(self):
        self.api.submit_video.side_effect = self.rejection(
            message="合成测试：提交响应中断，结果未知。", uncertain=True, http_status=503)
        self.assertEqual(run_confirming_prompts(self.root)["status"], "blocked")
        self.assertEqual(self.job()["state"], "uncertain")
        self.assertEqual(self.job()["accounting"], "unknown")
        self.assertTrue(self.job()["failure"]["uncertain"])
        self.assert_not_approvable()
        shot = self.shot()
        with self.assertRaises(core.GateError):
            core.reject(self.root, "shot", shot["id"], shot["version"], "不能把未知结果误当失败")
        for _ in range(3):
            self.assertEqual(run_confirming_prompts(self.root)["status"], "blocked")
        self.api.submit_video.assert_called_once()
        self.api.poll_video.assert_not_called()

    def test_provider_failed_result_keeps_task_id_and_diagnostic_then_waits_for_user(self):
        self.api.poll_video.return_value = {
            "status": "failed", "provider_status": "failed", "request_id": "synthetic-generation-request",
            "error": "Synthetic.ContentRejected",
            "failure": {"stage": "generation", "message": "合成测试：服务端内容校验未通过。",
                        "provider_code": "Synthetic.ContentRejected", "uncertain": False,
                        "request_id": "synthetic-generation-request"}}
        self.assertEqual(run_confirming_prompts(self.root)["status"], "generation_failed")
        original = copy.deepcopy(self.job())
        self.assertEqual(original["state"], "failed")
        self.assertEqual(original["task_id"], "synthetic-1")
        self.assertEqual(original["failure"]["stage"], "generation")
        self.assertEqual(original["failure"]["provider_code"], "Synthetic.ContentRejected")
        self.assertEqual(original["failure"]["request_id"], "synthetic-generation-request")
        self.assert_not_approvable()
        run_confirming_prompts(self.root)
        self.assertEqual(self.job(), original)
        self.api.submit_video.assert_called_once()
        self.api.poll_video.assert_called_once_with("synthetic-1")
        self.api.download.assert_not_called()

    def test_poll_failure_recovers_same_task_with_get_only_and_clears_current_failure(self):
        self.api.poll_video.side_effect = self.rejection(
            stage="poll", message="合成测试：查询暂时不可用。", http_status=502)
        self.assertEqual(run_confirming_prompts(self.root)["status"], "blocked")
        job_id = self.job()["id"]
        self.assertEqual(self.job()["state"], "submitted")
        self.assertEqual(self.job()["failure"]["stage"], "poll")
        self.assertEqual(self.job()["task_id"], "synthetic-1")
        self.api.poll_video.side_effect = None
        self.assertEqual(run_confirming_prompts(self.root)["status"], "review_required")
        self.assertEqual(self.job()["id"], job_id)
        self.assertEqual(self.job()["state"], "completed")
        self.assertFalse(self.job().get("failure"))
        self.assertEqual(self.job()["failure_history"][-1]["stage"], "poll")
        self.assertEqual(self.shot()["version"], 1)
        self.api.submit_video.assert_called_once()
        self.assertEqual([call.args[0] for call in self.api.poll_video.call_args_list],
                         ["synthetic-1", "synthetic-1"])

    def test_download_failure_recovers_original_task_without_another_submission(self):
        self.api.download.side_effect = ProviderError("合成测试：下载中断。", stage="download")
        self.assertEqual(run_confirming_prompts(self.root)["status"], "blocked")
        original_id = self.job()["id"]
        self.assertEqual(self.job()["state"], "download_pending")
        self.assertEqual(self.job()["failure"]["stage"], "download")
        self.assertEqual(self.job()["task_id"], "synthetic-1")
        self.assert_not_approvable()
        self.api.download.side_effect = lambda url, output: write(output, b"SYNTHETIC RECOVERED")
        self.assertEqual(run_confirming_prompts(self.root)["status"], "review_required")
        self.assertEqual(self.job()["id"], original_id)
        self.assertFalse(self.job().get("failure"))
        self.assertEqual(self.job()["failure_history"][-1]["stage"], "download")
        self.assertTrue((self.root / self.shot()["file"]).is_file())
        self.api.submit_video.assert_called_once()

    def test_completed_task_missing_url_is_download_problem_not_new_generation_permission(self):
        self.api.poll_video.return_value["url"] = None
        self.assertEqual(run_confirming_prompts(self.root)["status"], "blocked")
        original_id = self.job()["id"]
        self.assertEqual(self.job()["state"], "download_pending")
        self.assertEqual(self.job()["failure"]["stage"], "download")
        self.api.download.assert_not_called()
        self.api.poll_video.return_value["url"] = "https://example.test/recovered.mp4"
        self.assertEqual(run_confirming_prompts(self.root)["status"], "review_required")
        self.assertEqual(self.job()["id"], original_id)
        self.api.submit_video.assert_called_once()

    def test_successful_new_version_never_inherits_previous_failed_jobs_diagnosis(self):
        core.set_video_review_enabled(self.root, True)
        self.fail_submission()
        first_job = copy.deepcopy(self.job())
        shot = self.shot()
        core.reject(self.root, "shot", shot["id"], shot["version"], "合成授权：修改后重试一次")
        self.api.submit_video.side_effect = lambda *a, **kw: {"id": "synthetic-2"}
        self.assertEqual(run_confirming_prompts(self.root)["status"], "review_required")
        self.assertEqual(self.shot()["version"], 2)
        self.assertFalse(self.shot().get("generation_error"))
        self.assertFalse(self.shot().get("failure"))
        self.assertFalse(self.job().get("failure"))
        self.assertNotEqual(self.job()["id"], first_job["id"])
        self.assertEqual(core.load_project(self.root)["jobs"][first_job["id"]], first_job)
        self.assertEqual(self.api.submit_video.call_count, 2)
        self.assertEqual(self.shot()["review"]["version"], 2)

    def test_legacy_failed_without_file_migrates_from_review_to_generation_failure(self):
        self.fail_submission()
        with core.transaction(self.root) as state:
            shot = state["episodes"][0]["shots"][0]
            shot.update(status="review_required", generation_error="旧版：本次视频生成未得到可用结果。")
            state["status"] = "review_required"
            state["jobs"][shot["job_id"]].pop("failure", None)
        result = run_confirming_prompts(self.root)
        self.assertEqual(result["status"], "generation_failed")
        self.assertEqual(self.shot()["status"], "generation_failed")
        self.assert_not_approvable()
        self.api.submit_video.assert_called_once()
        self.assertEqual([call.args[1] for call in self.brain.call_args_list], ["refine_shot"])

    def test_legacy_waiting_candidate_is_not_reclassified_by_an_unrelated_failed_job(self):
        self.assertEqual(run_confirming_prompts(self.root)["status"], "review_required")
        original = copy.deepcopy(self.shot())
        with core.transaction(self.root) as state:
            unrelated = copy.deepcopy(state["jobs"][original["job_id"]])
            unrelated.update(id="synthetic-other-failure", item_id="E001-S002", state="failed",
                             failure={"stage": "submission", "message": "另一片段的合成错误。"})
            state["jobs"][unrelated["id"]] = unrelated
        self.assertEqual(run_confirming_prompts(self.root)["status"], "review_required")
        self.assertEqual(self.shot(), original)
        self.api.submit_video.assert_called_once()
        self.assertFalse(self.job().get("failure"))

    def test_legacy_failure_migrates_while_paused_without_starting_any_work(self):
        self.fail_submission()
        with core.transaction(self.root) as state:
            state["episodes"][0]["shots"][0]["status"] = "review_required"
            state["paused"] = True
        calls = self.brain.call_count
        self.assertEqual(run_confirming_prompts(self.root)["status"], "paused")
        self.assertEqual(self.shot()["status"], "generation_failed")
        self.assertEqual(self.brain.call_count, calls)
        self.api.submit_video.assert_called_once()

    def test_mismatched_job_binding_blocks_without_presenting_another_shots_failure(self):
        self.fail_submission()
        with core.transaction(self.root) as state:
            shot = state["episodes"][0]["shots"][0]
            shot["status"] = "review_required"
            shot.pop("generation_error", None)
            state["jobs"][shot["job_id"]]["item_id"] = "E001-S002"
        self.assertEqual(run_confirming_prompts(self.root)["status"], "blocked")
        self.assertFalse(self.shot().get("generation_error"))
        self.api.submit_video.assert_called_once()

