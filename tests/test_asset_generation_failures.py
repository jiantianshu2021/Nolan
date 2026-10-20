"""Synthetic offline asset failure/retry gates; no real APIs or generated media."""
import copy
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import Mock, patch

from norlan import asset_inputs, core, pipeline
from norlan.providers import ProviderError
from test_pipeline import choose_gpt_assets, fake_media, run_confirming_asset_prompts, write


class AssetGenerationFailureTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.enterContext(fake_media())
        self.line = "当前设备已经完成初始化，请按照操作说明继续查看本次任务。"
        self.source = "演示人物卡：" + self.line
        core.create_project(self.root, self.source)
        core.install_plan(self.root, pipeline.demo_plan())
        core.approve(self.root, "plan", "plan", 1)
        choose_gpt_assets(self.root)
        state = pipeline.run(self.root)
        for asset in state["assets"]:
            core.approve(self.root, "asset", asset["id"], asset["version"])
        core.configure(self.root, {"caps": {"image": 1, "video_second": 1}}, limit=200)
        with core.transaction(self.root) as state:
            state["mode"] = "live"
        self.original_assets = copy.deepcopy(core.load_project(self.root)["assets"])
        self.provider = Mock()
        self.provider.submit_video.return_value = {"id": "synthetic-voice-task"}
        self.provider.poll_video.return_value = {"status": "completed", "url": "https://example.test/synthetic.mp4",
            "last_frame_url": "https://example.test/synthetic.jpg", "created_at": int(time.time())}
        self.provider.download.side_effect = lambda url, target: write(target, b"\xff\xd8\xffSYNTHETIC" if url.endswith("jpg") else b"SYNTHETIC VOICE")
        self.provider.generate_image.side_effect = lambda prompt, target, **kwargs: (write(target, b"SYNTHETIC IMAGE") and {})
        self.enterContext(patch.object(pipeline, "_ark", return_value=self.provider))
        self.brain = self.enterContext(patch.object(pipeline, "_brain", side_effect=self.answer))

    def answer(self, _root, method, *args, **kwargs):
        if method == "select_voice_audition":
            return {"audition_text": self.line, "audition_source": self.source, "audition_reason": "合成原文中的完整中性陈述"}
        raise AssertionError("Asset failure tests must stop before video preparation")

    def add_asset(self, kind):
        identifier = "new-voice" if kind == "voice" else "new-prop"
        asset = {"id": identifier, "kind": "voice" if kind == "voice" else "prop", "name": "合成新增声音" if kind == "voice" else "合成新增道具",
                 "prompt": "成年女声，中音，清澈" if kind == "voice" else "合成道具图片", "status": "planned", "version": 0,
                 "history": [], "feedback": "", "creation_input": {**asset_inputs.empty_input(), "mode": "gpt", "confirmed": True, "revision": 1}}
        if kind == "voice":
            asset.update(character_id="hero", audition_text=self.line, audition_source=self.source,
                         audition_reason="合成原文的完整稳定陈述")
        with core.transaction(self.root) as state:
            state["assets"].append(asset)
        return identifier

    @staticmethod
    def definite_failure():
        return ProviderError("合成明确失败", failure={"stage": "submission", "message": "合成明确失败",
            "provider_code": "Synthetic.InputRejected", "http_status": 400, "request_id": "synthetic-request", "uncertain": False})

    def fail_asset(self, kind):
        identifier = self.add_asset(kind)
        call = self.provider.submit_video if kind == "voice" else self.provider.generate_image
        call.side_effect = self.definite_failure()
        # Explicit synthetic human consent for the new asset prompt gate;
        # subsequent plain Continue calls below must never authorize retries.
        state = run_confirming_asset_prompts(self.root)
        asset = core._find(state, "asset", identifier)
        self.assertEqual(state["status"], "asset_generation_failed")
        self.assertEqual(asset["status"], "generation_failed")
        self.assertTrue(asset["generation_error"])
        self.assertEqual(call.call_count, 1)
        self.assertEqual(state["jobs"][asset["job_id"]]["state"], "failed")
        return identifier, call

    def test_definite_voice_failure_resume_does_not_submit_again(self):
        identifier, call = self.fail_asset("voice")
        original = core.load_project(self.root)
        job_id = core._find(original, "asset", identifier)["job_id"]
        for _ in range(3):
            state = pipeline.run(self.root)
            self.assertEqual(state["status"], "asset_generation_failed")
        self.assertEqual(call.call_count, 1)
        self.assertEqual(set(state["jobs"]), set(original["jobs"]))
        self.assertEqual(core._find(state, "asset", identifier)["version"], 1)
        self.assertEqual(state["jobs"][job_id]["failure"]["provider_code"], "Synthetic.InputRejected")
        self.assertEqual(state["assets"][:2], self.original_assets)

    def test_explicit_voice_rework_produces_one_candidate_and_retains_failure_job(self):
        identifier, call = self.fail_asset("voice")
        failed = core._find(core.load_project(self.root), "asset", identifier)
        old_job = copy.deepcopy(core.load_project(self.root)["jobs"][failed["job_id"]])
        core.reject(self.root, "asset", identifier, failed["version"], "保持已确认女声，重试一次试音")
        state = core.load_project(self.root)
        requested = core._find(state, "asset", identifier)
        self.assertEqual(requested["status"], "rejected")
        self.assertNotIn("job_id", requested)
        self.assertNotIn("generation_error", requested)
        self.assertEqual(requested["failure_history"][-1]["job_id"], failed["job_id"])
        self.assertEqual(state["jobs"][failed["job_id"]], old_job)
        call.side_effect = None
        state = run_confirming_asset_prompts(self.root)
        self.assertEqual(state["status"], "awaiting_assets")
        current = core._find(state, "asset", identifier)
        self.assertEqual((current["status"], current["version"]), ("candidate", 2))
        self.assertEqual(call.call_count, 2)
        for _ in range(2):
            pipeline.run(self.root)
        self.assertEqual(call.call_count, 2)
        self.assertEqual(core.load_project(self.root)["jobs"][failed["job_id"]], old_job)
        self.assertEqual(core.load_project(self.root)["assets"][:2], self.original_assets)

    def test_image_failure_also_requires_explicit_rework(self):
        identifier, call = self.fail_asset("image")
        state = pipeline.run(self.root)
        self.assertEqual(call.call_count, 1)
        failed = core._find(state, "asset", identifier)
        core.reject(self.root, "asset", identifier, failed["version"], "保留道具定义，只重试一次")
        call.side_effect = lambda prompt, target, **kwargs: (write(target, b"SYNTHETIC RETRIED IMAGE") and {})
        state = run_confirming_asset_prompts(self.root)
        self.assertEqual((core._find(state, "asset", identifier)["status"], call.call_count), ("candidate", 2))
        pipeline.run(self.root)
        self.assertEqual(call.call_count, 2)
        self.provider.submit_video.assert_not_called()

    def test_pause_after_explicit_rework_does_not_generate_until_resumed(self):
        identifier, call = self.fail_asset("voice")
        core.set_paused(self.root, True)
        core.reject(self.root, "asset", identifier, 1, "恢复后仅重试一次试音")
        state = pipeline.run(self.root)
        self.assertEqual(state["status"], "paused")
        self.assertEqual(call.call_count, 1)
        self.assertNotIn("job_id", core._find(state, "asset", identifier))
        core.set_paused(self.root, False)
        call.side_effect = None
        self.assertEqual(run_confirming_asset_prompts(self.root)["status"], "awaiting_assets")
        self.assertEqual(call.call_count, 2)

    def test_unknown_or_inflight_failure_can_neither_be_rejected_nor_retried(self):
        identifier, call = self.fail_asset("voice")
        with core.transaction(self.root) as state:
            job = state["jobs"][core._find(state, "asset", identifier)["job_id"]]
            job.update(state="uncertain", accounting="unknown")
        before = (self.root / "project.json").read_bytes()
        with self.assertRaises(core.GateError):
            core.reject(self.root, "asset", identifier, 1, "不能越过待核实任务")
        self.assertEqual((self.root / "project.json").read_bytes(), before)
        state = pipeline.run(self.root)
        self.assertEqual(state["status"], "blocked")
        self.assertEqual(call.call_count, 1)
        with core.transaction(self.root) as state:
            job = state["jobs"][core._find(state, "asset", identifier)["job_id"]]
            job.update(state="submitted", accounting="reserved", task_id="synthetic-still-running")
        with self.assertRaises(core.GateError):
            core.reject(self.root, "asset", identifier, 1, "仍不可重试")
        self.assertEqual(call.call_count, 1)

    def test_legacy_failed_voice_job_is_stopped_before_version_or_post_changes(self):
        identifier = self.add_asset("voice")
        core.reserve(self.root, "legacy-failed-voice", "voice", identifier, 5)
        core.update_job(self.root, "legacy-failed-voice", state="failed", accounting="spent")
        with core.transaction(self.root) as state:
            core._find(state, "asset", identifier).update(status="generating", version=1, job_id="legacy-failed-voice")
        state = pipeline.run(self.root)
        asset = core._find(state, "asset", identifier)
        self.assertEqual(state["status"], "asset_generation_failed")
        self.assertEqual((asset["status"], asset["version"]), ("generation_failed", 1))
        self.provider.submit_video.assert_not_called()
        self.provider.generate_image.assert_not_called()

    def test_retry_needs_feedback_fresh_version_and_exact_settled_asset_job(self):
        identifier, call = self.fail_asset("voice")
        before = (self.root / "project.json").read_bytes()
        for feedback, version in (("", 1), (" ", 1), ("重试", 0), ("重试", True)):
            with self.subTest(feedback=feedback, version=version):
                with self.assertRaises((ValueError, core.GateError)):
                    core.reject(self.root, "asset", identifier, version, feedback)
                self.assertEqual((self.root / "project.json").read_bytes(), before)
        for mutations in ({"item_id": "hero"}, {"kind": "image"}, {"kind": "video"}, {"state": "completed"}, {"accounting": "unknown"}):
            with self.subTest(mutations=mutations):
                current = core.load_project(self.root)
                job_id = core._find(current, "asset", identifier)["job_id"]
                original = copy.deepcopy(current["jobs"][job_id])
                with core.transaction(self.root) as state:
                    state["jobs"][job_id].update(mutations)
                before = (self.root / "project.json").read_bytes()
                with self.assertRaises(core.GateError):
                    core.reject(self.root, "asset", identifier, 1, "不能重试不匹配调用")
                self.assertEqual((self.root / "project.json").read_bytes(), before)
                with core.transaction(self.root) as state:
                    state["jobs"][job_id] = original
        self.assertEqual(call.call_count, 1)

    def test_explicit_retry_accepts_verified_released_failure_without_erasing_ledger(self):
        identifier, call = self.fail_asset("image")
        with core.transaction(self.root) as state:
            job_id = core._find(state, "asset", identifier)["job_id"]
            state["jobs"][job_id]["accounting"] = "released"
        core.reject(self.root, "asset", identifier, 1, "已核实未扣费，重试一次图片")
        state = core.load_project(self.root)
        self.assertEqual(state["jobs"][job_id]["accounting"], "released")
        self.assertEqual(core._find(state, "asset", identifier)["failure_history"][-1]["job_id"], job_id)
        self.assertEqual(call.call_count, 1)


if __name__ == "__main__":
    unittest.main()
