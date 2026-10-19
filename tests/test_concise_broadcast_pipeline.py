"""Synthetic concise-broadcast delivery tests; no GPT or media-provider calls."""
import copy
from pathlib import Path
import unittest
from unittest.mock import patch

from norlan import broadcast, core, pipeline
import test_prompt_gate as fixtures
from test_pipeline import write


class ConciseBroadcastPipelineTests(unittest.TestCase):
    shot = fixtures.PromptGateTests.shot
    confirm = fixtures.PromptGateTests.confirm

    def setUp(self):
        fixtures.PromptGateTests.setUp(self)
        self.readback = []
        definitions = fixtures.PromptGateTests.amendment(self)["additions"]
        with core.transaction(self.root) as state:
            source, voice = definitions
            source.update(status="planned", version=0)
            preview = write(self.root / "media/system-voice.mp4", b"SYNTHETIC APPROVED VOICE VIDEO")
            audio = write(self.root / "media/system-voice.wav", b"SYNTHETIC APPROVED AUDIO")
            voice.update(status="approved", version=1, file=str(preview.relative_to(self.root)),
                         sha256=core.digest(preview), approved_sha256=core.digest(preview),
                         audio_file=str(audio.relative_to(self.root)), audio_sha256=core.digest(audio),
                         approved_at=core.now())
            state["assets"].extend(definitions)
            state["audio_policy"] = {"system_broadcast": {"enabled": True,
                "character_id": "system-speaker", "voice_id": "system-voice",
                "instruction": "SYNTHETIC_LEGACY_READ_EVERY_RULE：全部逐字朗读，必须请用户填写秒点。"}}
            for shot in state["episodes"][0]["shots"]:
                shot.update(duration=15, dialogue="主角：知道了。\n系统播报员：" + self.text,
                    prompt="SYNTHETIC_OLD_COMPLETE_READING：将详细规则全念完。",
                    subshots=[{"start": 0, "end": 5, "prompt": "人物看向设备，系统全文播读。"},
                              {"start": 5, "end": 15, "prompt": "继续逐字朗读所有说明。"}])
        core.normalize_voice_sources(self.root)
        self.enterContext(patch.object(pipeline.media, "extract_evidence", side_effect=self.evidence))

    @staticmethod
    def evidence(video, directory, **kwargs):
        frames = [str(write(Path(directory) / f"frame-{index}.png", f"SYNTHETIC FRAME {index}".encode()))
                  for index in range(3)]
        return {"frames": frames, "tail_frame": frames[-1], "audio": None, "sha256": core.digest(video),
                "sample_times": [0, 7.5, 14.9],
                "metadata": {"duration": 15, "has_video": True, "has_audio": True}}

    def answer(self, root, method, *args, **kwargs):
        if method == "refine_shot":
            return {"prompt": "合成测试：人物先回答；任务框显示完整规则，系统女声只说关键结果，然后留出反应。",
                "dialogue": "主角：知道了。\n系统播报员：设备已经就绪。", "continuity_notes": "保留完整界面说明，播音忠实简化。",
                "broadcast_proposal": {"speaker": "系统播报员", "source_text": self.text,
                    "spoken_text": "设备已经就绪。", "display_text": self.text,
                    "summary": "只读就绪结果，完整操作规则留在任务框，不请用户另选秒点。"},
                "timing_proposal": {"summary": "人物先说，2–4秒简播，余下时间留给动作和反应。",
                    "subshots": [{"start": 0, "end": 5, "prompt": "人物回答，任务框显示完整规则，2秒开始简播。"},
                                 {"start": 5, "end": 15, "prompt": "任务框保持完整说明，人物查看设备。"}],
                    "audio_windows": [{"start": .2, "end": 1.2, "speaker": "主角", "text": "知道了。"},
                                      {"start": 2, "end": 4, "speaker": "系统播报员", "text": "设备已经就绪。"}]}}
        if method == "plan_revision":
            return {"needs_assets": False, "additions": [], "shot_updates": [], "summary": "合成已授权的简播安排。", "blockers": []}
        self.readback.append(copy.deepcopy(args[0]))
        return {"verdict": "uncertain", "issues": [], "observed_end_state": "合成抽样图终点", "repair_prompt": ""}

    def test_concise_candidate_keeps_full_visible_rules_without_reinserting_complete_audio(self):
        original = copy.deepcopy(self.shot())
        self.assertEqual(pipeline.run(self.root)["status"], "awaiting_prompt_approval")
        candidate = copy.deepcopy(self.shot()["prompt_candidate"])
        prompt = candidate["prompt"]
        self.assertIn(self.text, prompt)  # Complete rules still reach the video engine as UI text.
        self.assertIn("系统播报员：设备已经就绪。", prompt)
        self.assertNotIn("必须使用的原定对白：" + original["dialogue"], prompt)
        self.assertNotIn("SYNTHETIC_LEGACY_READ_EVERY_RULE", prompt)
        self.assertNotIn("SYNTHETIC_OLD_COMPLETE_READING", prompt)
        self.assertIn("音频1", prompt)
        self.assertEqual(self.shot()["dialogue"], original["dialogue"])
        self.assertEqual(self.shot()["subshots"], original["subshots"])
        self.assertNotIn("prompt_approval", self.shot())
        count = self.brain.call_count
        for _ in range(3):
            self.assertEqual(pipeline.run(self.root)["status"], "awaiting_prompt_approval")
        self.assertEqual(self.brain.call_count, count)
        self.assertEqual(self.shot()["prompt_candidate"], candidate)
        self.api.submit_video.assert_not_called()

    def test_one_final_confirmation_submits_exact_prompt_and_review_uses_actual_spoken_text(self):
        pipeline.run(self.root)
        candidate = copy.deepcopy(self.shot()["prompt_candidate"])
        self.confirm()
        self.assertEqual(pipeline.run(self.root)["status"], "review_required")
        self.api.submit_video.assert_called_once()
        self.assertEqual(self.api.submit_video.call_args.args[0], candidate["prompt"])
        self.assertEqual(self.api.submit_video.call_args.kwargs["audio_references"],
                         [self.root / "media/system-voice.wav"])
        review = self.readback[-1]["shot"]
        self.assertEqual(review["dialogue"], candidate["dialogue"])
        self.assertEqual(review["prompt"], candidate["refined"]["prompt"])
        self.assertEqual(review["subshots"], candidate["timing_proposal"]["subshots"])
        self.assertFalse(core.human_approved_shot(self.shot()))
        pipeline.run(self.root)
        self.api.submit_video.assert_called_once()

    def test_later_segment_reuses_the_approved_voice_and_prepares_a_new_human_confirmation(self):
        pipeline.run(self.root)
        self.confirm()
        pipeline.run(self.root)
        first = self.shot()
        core.approve(self.root, "shot", first["id"], first["version"])
        self.assertEqual(pipeline.run(self.root)["status"], "awaiting_prompt_approval")
        current = core.load_project(self.root)["episodes"][0]["shots"][1]
        self.assertIn("system-voice", current["asset_ids"])
        self.assertEqual(current["prompt_candidate"]["dialogue"], "主角：知道了。\n系统播报员：设备已经就绪。")
        self.assertNotIn("SYNTHETIC_LEGACY_READ_EVERY_RULE", current["prompt_candidate"]["prompt"])
        self.assertNotIn("prompt_approval", current)
        self.api.submit_video.assert_called_once()

    def test_single_camera_candidate_still_contains_visual_rules_and_selected_spoken_content(self):
        with core.transaction(self.root) as state:
            state["episodes"][0]["shots"][0].pop("subshots")

        def single_camera(root, method, *args, **kwargs):
            result = self.answer(root, method, *args, **kwargs)
            if method == "refine_shot":
                result["timing_proposal"] = None
            return result

        self.brain.side_effect = single_camera
        self.assertEqual(pipeline.run(self.root)["status"], "awaiting_prompt_approval")
        candidate = copy.deepcopy(self.shot()["prompt_candidate"])
        self.assertIn(self.text, candidate["prompt"])
        self.assertIn("系统播报员画外播报：设备已经就绪。", candidate["prompt"])
        self.assertNotIn("timing_proposal", candidate)
        self.assertNotIn("subshots", self.shot())
        self.api.submit_video.assert_not_called()
        self.confirm()
        self.assertEqual(pipeline.run(self.root)["status"], "review_required")
        self.assertEqual(self.readback[-1]["shot"]["dialogue"], candidate["dialogue"])
        self.assertEqual(self.api.submit_video.call_args.args[0], candidate["prompt"])

    def test_normalized_editorial_instruction_does_not_create_another_asset_approval(self):
        with core.transaction(self.root) as state:
            state["episodes"][0]["shots"][0].update(feedback="系统播报取舍和时点交给GPT，准备完整提示词。", revision=1)
        original_policy = copy.deepcopy(core.load_project(self.root)["audio_policy"])
        normalized = broadcast.policy_for(core.load_project(self.root))
        returned_policy = {key: normalized[key] for key in ("enabled", "character_id", "voice_id", "instruction")}

        def normalized_revision(root, method, *args, **kwargs):
            result = self.answer(root, method, *args, **kwargs)
            if method == "plan_revision":
                result["audio_policy"] = {"system_broadcast": returned_policy}
            return result

        self.brain.side_effect = normalized_revision
        self.assertEqual(pipeline.run(self.root)["status"], "awaiting_prompt_approval")
        state = core.load_project(self.root)
        self.assertEqual(state.get("asset_amendments", []), [])
        self.assertEqual(state["audio_policy"], original_policy)
        self.assertNotIn("amendment_id", self.shot()["revision_analysis"])
        self.assertNotIn("prompt_approval", self.shot())
        self.api.submit_video.assert_not_called()

    def test_explicitly_disabling_the_system_voice_still_requires_a_policy_decision(self):
        with core.transaction(self.root) as state:
            state["episodes"][0]["shots"][0].update(feedback="以后不要系统播音。", revision=1)
        disabled = copy.deepcopy(core.load_project(self.root)["audio_policy"])
        disabled["system_broadcast"]["enabled"] = False

        def disable_revision(root, method, *args, **kwargs):
            result = self.answer(root, method, *args, **kwargs)
            if method == "plan_revision":
                result["audio_policy"] = disabled
                current = self.shot()
                result["shot_updates"] = [{"id": current["id"], "add_asset_ids": [],
                    "prompt": "合成任务框仅显示文字，系统声静默。", "dialogue": "主角：知道了。",
                    "continuity_notes": "按明确要求关闭系统声源。"}]
            return result

        self.brain.side_effect = disable_revision
        self.assertFalse(pipeline._revision_analysis(self.root, self.shot()))
        state = core.load_project(self.root)
        self.assertEqual(state["status"], "awaiting_asset_amendment")
        self.assertEqual(state["asset_amendments"][-1]["audio_policy"], disabled)
        self.assertTrue(state["audio_policy"]["system_broadcast"]["enabled"])
        self.api.submit_video.assert_not_called()


if __name__ == "__main__":
    unittest.main()
