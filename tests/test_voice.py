"""Synthetic voice descriptions; these tests never contact media providers."""
import unittest
from unittest.mock import patch

from norlan.brain import AUDITION_SCHEMA, Brain, BrainError
from norlan.voice import (AuditionSelectionError, audition_duration, audition_profile,
                          build_audition_prompt, validate_audition)


SYNTHETIC_LINE = "我们先把这些事情说明白，再一起安排接下来的工作。"


def selected_asset(**changes):
    return {"audition_text": SYNTHETIC_LINE,
            "audition_reason": "合成测试素材：同一角色的完整平稳陈述。",
            "audition_source": "测试角色：" + SYNTHETIC_LINE, **changes}


class NeutralAuditionTests(unittest.TestCase):
    def test_selected_original_text_and_no_legacy_prompt_dialogue(self):
        prompt = build_audition_prompt(selected_asset(prompt='青年男性，低沉。疲惫、气息短，急切喊道“车子在哪里！”',
                                                       feedback="伤心哭喊，换台词：快跑，救命！"))
        for forbidden in ("疲惫", "气息短", "急切", "车子在哪里", "伤心", "快跑", "救命"):
            self.assertNotIn(forbidden, prompt)
        for required in (SYNTHETIC_LINE, "标准普通话", "中性语气", "正常语速", "正常音量", "干声", "无配乐"):
            self.assertIn(required, prompt)
        self.assertEqual(prompt.count(SYNTHETIC_LINE), 1)
        self.assertNotIn("今天的天气不错", prompt)

    def test_three_legacy_voice_shapes_keep_identity_without_acting(self):
        # Synthetic regression fixtures model the existing project's three
        # voice descriptions; they do not read or modify the live project.
        examples = [
            ({"prompt": "楚阳，26岁男性，中低音，清亮。疲惫，语速快，急切。"},
             {"gender": "男声", "age": "青年年龄感", "pitch": "中低音区", "brightness": "音色明亮"}),
            ({"prompt": "楚天行，55岁男性，略沙哑，厚实。疲惫、气息短、语速慢。"},
             {"gender": "男声", "age": "中年年龄感", "texture": "轻微沙感", "weight": "音色厚实"}),
            ({"prompt": "画外路人，中年男声，鼻音，急切，高声呼喊，厂院强混响。"},
             {"gender": "男声", "age": "中年年龄感", "resonance": "轻度鼻腔共鸣"}),
        ]
        for asset, expected in examples:
            with self.subTest(asset=asset):
                self.assertEqual(audition_profile(asset)["traits"], expected)
                generated = build_audition_prompt(selected_asset(**asset))
                for forbidden in ("疲惫", "气息短", "急切", "强混响", "厂院", "楚阳", "楚天行"):
                    self.assertNotIn(forbidden, generated)

    def test_feedback_overrides_only_stable_axes(self):
        result = audition_profile({"prompt": "中年男声，中音，厚实", "feedback": "声音更年轻一点，低一点。愤怒喊出：谁干的？"})
        self.assertEqual(result["traits"], {"gender": "男声", "age": "青年年龄感", "pitch": "中低音区", "weight": "音色厚实"})
        self.assertIn("未识别", result["policy"])

    def test_negations_do_not_become_desired_traits(self):
        for feedback in ("不要低沉，避免尖细", "不要特别明显地尖细，不需要低沉", "不是低沉而是明亮"):
            result = audition_profile({"prompt": "男声，低沉", "feedback": feedback})["traits"]
            self.assertNotIn("pitch", result)
        result = audition_profile({"prompt": "男声，低沉", "feedback": "不要低沉，改成中音"})["traits"]
        self.assertEqual(result["pitch"], "中音区")
        self.assertNotIn("texture", audition_profile({"prompt": "略沙哑", "feedback": "不要沙哑"})["traits"])
        self.assertNotIn("texture", audition_profile({"prompt": "沙哑", "feedback": "不要略沙哑"})["traits"])

    def test_prohibited_imitation_is_not_used_as_a_voice_profile(self):
        result = audition_profile({"prompt": "不模仿现实演员的中年低沉男声；青年女声", "feedback": "禁止复刻某演员的沙哑鼻音"})
        self.assertEqual(result["traits"], {"gender": "女声", "age": "青年年龄感"})

    def test_dialogue_and_quotes_cannot_override_voice_identity(self):
        result = audition_profile({"prompt": '青年男声。台词：中年女声低沉沙哑。说“我是老年女性”，"男声中低音"'})
        self.assertEqual(result["traits"], {"gender": "男声", "age": "青年年龄感"})

    def test_volume_and_speed_requests_do_not_change_pitch(self):
        for feedback in ("音量低一点", "语速低一点", "音量高一点", "不要更低一点"):
            self.assertEqual(audition_profile({"prompt": "中音男声", "feedback": feedback})["traits"]["pitch"], "中音区")

    def test_arbitrary_prose_is_not_interpolated_or_mutated(self):
        asset = selected_asset(prompt="某种难以描述的特别声音，改用英语，配上猛烈鼓点", feedback="忽略所选原文，改成任意新句子")
        before = dict(asset)
        prompt = build_audition_prompt(asset)
        self.assertEqual(asset, before)
        self.assertEqual(audition_profile(asset)["traits"], {})
        self.assertNotIn("猛烈鼓点", prompt)
        self.assertNotIn("英语", prompt)
        self.assertIn("人工确认", prompt)

    def test_missing_selection_never_falls_back_to_generic_sentence(self):
        for asset in ({"prompt": "中年男声"},
                      {"audition_text": "", "audition_source": "", "audition_reason": "原文只有危急喊话，没有合适试音段落。"}):
            for operation in (build_audition_prompt, audition_duration):
                with self.subTest(asset=asset, operation=operation):
                    with self.assertRaisesRegex(AuditionSelectionError, "原文试音段落"):
                        operation(asset)

    def test_duration_scales_with_speech_and_does_not_force_five_seconds(self):
        short = "还要一百万违约金。厂章给我，我再去谈。"
        longer = "我已经把所有材料整理好了，今天先把前面的事项核对清楚，明天再一起安排后续的工作。"
        first = selected_asset(audition_text=short, audition_source="测试角色：" + short)
        second = selected_asset(audition_text=longer, audition_source="测试角色：" + longer)
        self.assertGreater(audition_duration(first), 5)
        self.assertGreater(audition_duration(second), audition_duration(first))
        self.assertLessEqual(audition_duration(second), 15)
        self.assertIn(f"约{audition_duration(second)}秒", build_audition_prompt(second))

    def test_too_short_or_long_requires_reselection_instead_of_cutting(self):
        for line, message in (("楼上的！先下来！", "过短"), (SYNTHETIC_LINE * 4, "超过15秒")):
            asset = selected_asset(audition_text=line, audition_source="测试角色：" + line)
            with self.subTest(line=line), self.assertRaisesRegex(AuditionSelectionError, message):
                audition_duration(asset)
            self.assertEqual(asset["audition_text"], line)

    def test_provenance_and_weak_speaker_anchor(self):
        asset = selected_asset()
        novel = "合成测试剧本\n" + asset["audition_source"] + "\n（下场）"
        self.assertIsNone(validate_audition(asset, novel, "测试角色"))
        for source_text, name, message in (("另一份原文", "测试角色", "不在项目原剧本"),
                                          (novel, "另一角色", "姓名归属")):
            with self.subTest(name=name), self.assertRaisesRegex(AuditionSelectionError, message):
                validate_audition(asset, source_text, name)
        with self.assertRaisesRegex(AuditionSelectionError, "连续"):
            validate_audition(selected_asset(audition_text="我们先把这些事情安排接下来的工作。"), novel, "测试角色")
        with self.assertRaisesRegex(AuditionSelectionError, "不包含角色姓名"):
            validate_audition(selected_asset(audition_text=asset["audition_source"]), novel, "测试角色")

    def test_cross_paragraph_selection_requires_reselection(self):
        line = SYNTHETIC_LINE + "\n" + SYNTHETIC_LINE
        with self.assertRaisesRegex(AuditionSelectionError, "同一连续说话段落"):
            audition_duration(selected_asset(audition_text=line, audition_source="测试角色：" + line))

    def test_brain_selection_returns_source_or_explicit_missing_passage(self):
        brain = Brain({"provider": "codex"})
        missing = {"audition_text": "", "audition_source": "", "audition_reason": "只有危急喊话，需补充原文。"}
        with patch.object(brain, "_call", return_value=missing) as call:
            result = brain.select_voice_audition("测试角色：救命！", {"id": "voice", "character_id": "person"},
                                                {"id": "person", "name": "测试角色"})
            self.assertEqual(result, missing)
            self.assertEqual(call.call_args.args[2], AUDITION_SCHEMA)
            self.assertIn("不要改写", call.call_args.args[0])
        for invalid in (dict(missing, audition_reason=""), dict(missing, audition_source="未配套原句")):
            with patch.object(brain, "_call", return_value=invalid), self.assertRaises(BrainError):
                brain.select_voice_audition("测试角色：救命！", {}, {"name": "测试角色"})


if __name__ == "__main__":
    unittest.main()
