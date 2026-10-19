"""Offline human consent and asset-first rework regressions; no real APIs."""
import copy
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import Mock, patch

from norlan import core, pipeline
from test_pipeline import choose_gpt_assets, fake_media, write


class PromptGateTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name).resolve()
        self.enterContext(fake_media())
        self.text = "欢迎使用系统，当前设备已经就绪，请按照操作说明继续。"
        self.source = "系统字幕：" + self.text
        core.create_project(self.root, "合成测试小说。\n" + self.source)
        core.install_plan(self.root, pipeline.demo_plan())
        core.approve(self.root, "plan", "plan", 1)
        choose_gpt_assets(self.root)
        state = pipeline.run(self.root)
        for asset in state["assets"]:
            core.approve(self.root, "asset", asset["id"], asset["version"])
        core.configure(self.root, {"caps": {"video_second": 1}}, limit=100)
        with core.transaction(self.root) as state:
            state["mode"] = "live"
        self.api = Mock()
        self.api.submit_video.side_effect = lambda *args, **kwargs: {"id": "synthetic-" + str(self.api.submit_video.call_count)}
        self.api.poll_video.return_value = {"status": "completed", "url": "https://example.test/clip.mp4",
            "last_frame_url": "https://example.test/tail.jpg", "created_at": int(time.time())}
        self.api.download.side_effect = lambda url, out: write(out, b"\xff\xd8\xffSYNTHETIC" if url.endswith(".jpg") else b"SYNTHETIC VIDEO")
        self.enterContext(patch.object(pipeline, "_ark", return_value=self.api))
        self.brain = self.enterContext(patch.object(pipeline, "_brain", side_effect=self.answer))

    def answer(self, root, method, *args, **kwargs):
        if method == "refine_shot":
            return {"prompt": "合成最终画面提示。", "dialogue": "", "continuity_notes": "合成连续性说明"}
        if method == "plan_revision":
            if "女性" in args[-1] or "声音资产" in args[-1]:
                return self.amendment()
            return {"needs_assets": False, "additions": [], "shot_updates": [], "summary": "仅调整原镜头", "blockers": []}
        return {"verdict": "pass", "issues": [], "observed_end_state": "合成尾态", "repair_prompt": ""}

    def shot(self):
        return core.load_project(self.root)["episodes"][0]["shots"][0]

    def confirm(self):
        shot = self.shot()
        candidate = shot["prompt_candidate"]
        return core.approve_prompt(self.root, shot["id"], candidate["id"], candidate["request_hash"])

    def amendment(self):
        return {"needs_assets": True, "summary": "新增合成系统女声", "blockers": [],
            "additions": [
                {"id": "system-speaker", "kind": "character", "name": "系统播报员", "prompt": "仅画外声源，不入画。", "voice_only": True},
                {"id": "system-voice", "kind": "voice", "name": "系统女声", "prompt": "成年女性，中音区，音色清澈。",
                 "character_id": "system-speaker", "audition_speaker": "系统", "audition_text": self.text,
                 "audition_reason": "合成原文中的平稳完整说明，用于中性干声试音。", "audition_source": self.source}],
            "shot_updates": [{"id": self.shot()["id"], "add_asset_ids": ["system-speaker", "system-voice"],
                              "prompt": "合成系统界面弹出，画外系统女声说：系统就绪。", "dialogue": "系统播报员：系统就绪。"}],
            "audio_policy": {"system_broadcast": {"enabled": True, "character_id": "system-speaker", "voice_id": "system-voice",
                                                   "instruction": "每次系统任务框都用同一女声播报本次明确通知。"}}}

    def test_first_generation_waits_and_repeated_run_never_submits(self):
        before = copy.deepcopy(core.load_project(self.root)["assets"])
        self.assertEqual(pipeline.run(self.root)["status"], "awaiting_prompt_approval")
        shot = self.shot()
        self.assertEqual(shot.get("version", 0), 0)
        self.assertIn("不添加背景配乐", shot["prompt_candidate"]["prompt"])
        calls = self.brain.call_count
        for _ in range(3):
            self.assertEqual(pipeline.run(self.root)["status"], "awaiting_prompt_approval")
        self.api.submit_video.assert_not_called()
        self.assertEqual(self.brain.call_count, calls)
        self.assertEqual(core.load_project(self.root)["assets"], before)

    def test_exact_candidate_confirmation_submits_once_then_waits_for_video_review(self):
        pipeline.run(self.root)
        prompt = self.shot()["prompt_candidate"]["prompt"]
        self.confirm()
        self.assertEqual(pipeline.run(self.root)["status"], "review_required")
        self.assertEqual(self.api.submit_video.call_args.args[0], prompt)
        self.assertEqual(self.shot()["version"], 1)
        pipeline.run(self.root)
        self.api.submit_video.assert_called_once()
        self.assertFalse(core.human_approved_shot(self.shot()))

    def test_old_hash_duplicate_and_paused_confirmations_are_rejected(self):
        pipeline.run(self.root)
        shot = self.shot(); candidate = shot["prompt_candidate"]
        with self.assertRaises(core.GateError):
            core.approve_prompt(self.root, shot["id"], candidate["id"], "0" * 64)
        core.set_paused(self.root, True)
        with self.assertRaises(core.GateError):
            self.confirm()
        core.set_paused(self.root, False)
        self.confirm()
        with self.assertRaises(core.GateError):
            self.confirm()
        self.api.submit_video.assert_not_called()

    def test_changed_asset_or_provider_configuration_cannot_use_old_confirmation(self):
        pipeline.run(self.root)
        self.confirm()
        core.configure(self.root, {"ark": {"resolution": "480p"}})
        self.assertEqual(pipeline.run(self.root)["status"], "blocked")
        self.api.submit_video.assert_not_called()
        self.assertEqual(self.shot()["status"], "planned")
        self.assertNotIn("prompt_approval", self.shot())
        self.assertEqual(pipeline.run(self.root)["status"], "awaiting_prompt_approval")
        self.api.submit_video.assert_not_called()

    def test_prompt_rewrite_requires_a_new_human_confirmation(self):
        pipeline.run(self.root)
        shot = self.shot(); old = shot["prompt_candidate"]
        core.reject_prompt(self.root, shot["id"], old["id"], old["request_hash"], "镜头改为缓慢推近")
        self.assertEqual(pipeline.run(self.root)["status"], "awaiting_prompt_approval")
        self.assertNotEqual(self.shot()["prompt_candidate"]["id"], old["id"])
        self.api.submit_video.assert_not_called()

    def test_rework_also_requires_prompt_approval_and_does_not_reuse_old_consent(self):
        pipeline.run(self.root); self.confirm(); pipeline.run(self.root)
        shot = self.shot(); original = (self.root / shot["file"]).read_bytes()
        core.reject(self.root, "shot", shot["id"], shot["version"], "修正人物转身动作")
        self.assertEqual(pipeline.run(self.root)["status"], "awaiting_prompt_approval")
        self.api.submit_video.assert_called_once()
        self.assertEqual((self.root / shot["file"]).read_bytes(), original)
        self.confirm(); pipeline.run(self.root)
        self.assertEqual(self.api.submit_video.call_count, 2)

    def test_feedback_with_new_voice_returns_to_assets_before_rework(self):
        pipeline.run(self.root)
        shot = self.shot(); candidate = shot["prompt_candidate"]
        original_assets = copy.deepcopy(core.load_project(self.root)["assets"])
        core.reject_prompt(self.root, shot["id"], candidate["id"], candidate["request_hash"], "增加固定女性系统播报声音资产")
        self.assertEqual(pipeline.run(self.root)["status"], "awaiting_asset_amendment")
        state = core.load_project(self.root); amendment = state["asset_amendments"][-1]
        self.assertEqual(state["assets"], original_assets)
        for _ in range(2):
            self.assertEqual(pipeline.run(self.root)["status"], "awaiting_asset_amendment")
        self.api.submit_video.assert_not_called()
        core.apply_asset_amendment(self.root, amendment["id"], "approve")
        self.assertEqual(pipeline.run(self.root)["status"], "awaiting_asset_input")
        source = core._find(core.load_project(self.root), "asset", "system-speaker")
        self.assertNotIn("file", source)
        core.approve(self.root, "asset", source["id"], source["version"])
        choose_gpt_assets(self.root)
        self.assertEqual(pipeline.run(self.root)["status"], "awaiting_assets")
        self.api.submit_video.assert_called_once()  # voice audition only
        self.assertEqual(self.api.submit_video.call_args.args[2], [])
        self.assertIn("不出现任何人物", self.api.submit_video.call_args.args[0])
        voice = core._find(core.load_project(self.root), "asset", "system-voice")
        core.approve(self.root, "asset", voice["id"], voice["version"])
        self.assertEqual(pipeline.run(self.root)["status"], "awaiting_prompt_approval")
        self.api.submit_video.assert_called_once()
        self.assertIn("系统播音规则", self.shot()["prompt_candidate"]["prompt"])
        self.confirm()
        self.assertEqual(pipeline.run(self.root)["status"], "review_required")
        self.assertEqual(self.api.submit_video.call_count, 2)
        self.assertTrue(self.api.submit_video.call_args.kwargs["audio_references"])
        self.assertEqual(core.load_project(self.root)["assets"][:2], original_assets)
        current = self.shot()
        core.approve(self.root, "shot", current["id"], current["version"])
        self.assertEqual(pipeline.run(self.root)["status"], "awaiting_prompt_approval")
        next_shot = core.load_project(self.root)["episodes"][0]["shots"][1]
        self.assertIn("system-voice", next_shot["asset_ids"])
        self.assertIn("没有明确系统通知", next_shot["prompt_candidate"]["prompt"])
        self.assertEqual(self.api.submit_video.call_count, 2)

    def test_reusing_an_existing_voice_goes_through_reference_plan_without_new_assets(self):
        # Existing approved source/voice, represented by explicitly synthetic
        # local media, can be added to a shot without reproducing that asset.
        definitions = self.amendment()["additions"]
        with core.transaction(self.root) as state:
            source = definitions[0]
            source.update(status="approved", version=1, sha256=core.voice_only_digest(source), approved_at=core.now())
            source["approved_sha256"] = source["sha256"]
            voice = definitions[1]
            file = write(self.root / "media/existing-voice-v1.mp4", b"SYNTHETIC EXISTING VOICE")
            audio = write(self.root / "media/existing-voice-v1.wav", b"SYNTHETIC EXISTING AUDIO")
            voice.update(status="approved", version=1, file=str(file.relative_to(self.root)), sha256=core.digest(file),
                         audio_file=str(audio.relative_to(self.root)), audio_sha256=core.digest(audio), approved_at=core.now())
            voice["approved_sha256"] = voice["sha256"]
            state["assets"] += definitions
        original_assets = copy.deepcopy(core.load_project(self.root)["assets"])
        pipeline.run(self.root)
        shot = self.shot(); candidate = shot["prompt_candidate"]
        core.reject_prompt(self.root, shot["id"], candidate["id"], candidate["request_hash"], "复用已批准的女性声音播报")
        result = self.amendment()
        result.update(needs_assets=False, additions=[])
        self.brain.side_effect = lambda root, method, *args, **kwargs: result if method == "plan_revision" else self.answer(root, method, *args, **kwargs)
        self.assertEqual(pipeline.run(self.root)["status"], "awaiting_asset_amendment")
        state = core.load_project(self.root)
        core.apply_asset_amendment(self.root, state["asset_amendments"][-1]["id"], "approve")
        self.assertEqual(pipeline.run(self.root)["status"], "awaiting_prompt_approval")
        self.api.submit_video.assert_not_called()
        self.assertEqual(core.load_project(self.root)["assets"], original_assets)
        self.assertIn("system-voice", self.shot()["asset_ids"])


if __name__ == "__main__":
    unittest.main()
