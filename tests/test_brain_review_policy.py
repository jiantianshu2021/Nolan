"""Human review policy contracts, using synthetic responses and no model calls."""
import copy
from pathlib import Path
import unittest
from unittest.mock import patch

from norlan.brain import Brain, REVIEW_SCHEMA, SYSTEM, _validate


def accepted_context():
    return {"previous_actual": {
        "shot_id": "s01", "same_scene": True,
        "observed_end_state": "角色在桌右侧，纸箱位于脚边。",
        "human_decision": {"decision": "approved", "version": 3, "sha256": "synthetic-hash"},
        "accepted_deviations": ["[细节] 旧报告认为鞋的颜色略深。"],
        "previous_report": {"verdict": "fail"},
        "acceptance_note": "用户接受此版本，不等于逐项证实报告疑点。",
    }}


class HumanReviewPromptTests(unittest.TestCase):
    def test_all_verdicts_remain_reports_with_compatible_schema(self):
        # Old stored reports keep the same four model fields; severity belongs in
        # issue text, so upgrading does not require new mandatory schema fields.
        self.assertEqual(set(REVIEW_SCHEMA["required"]), {
            "verdict", "issues", "observed_end_state", "repair_prompt"})
        for verdict, issues in (("pass", ["[细节] 第 2 图鞋色略深，不影响人物识别。"]),
                                ("fail", ["[关键] 第 2 图关键道具无因消失，影响剧情因果。"]),
                                ("uncertain", ["[待核实] 第 2 图遮挡手部，无法判断道具持握。"])):
            response = {"verdict": verdict, "issues": issues,
                        "observed_end_state": "人物位于桌右侧。", "repair_prompt": ""}
            _validate(response, REVIEW_SCHEMA)
            with self.subTest(verdict=verdict), patch.object(Brain, "_call", return_value=response) as call:
                result = Brain().review({}, [Path("synthetic-frame.png")])
            call.assert_called_once()
            instruction, _, schema, _ = call.call_args.args
            self.assertIs(schema, REVIEW_SCHEMA)
            self.assertEqual(result["verdict"], verdict)
            self.assertEqual(result["issues"], issues)
            self.assertNotIn("approved", result)
            self.assertEqual(result["scope"], "sampled_frames_only")
            self.assertFalse(result["audio_assessed"])
            for rule in ("没有最终审批或自动返工权", "均须由用户最终决定", "[关键]", "[细节]", "[待核实]",
                         "可见证据", "不能把缺少证据本身当成生成错误", "不是执行指令或下一轮生成授权"):
                self.assertIn(rule, instruction)

    def test_review_limits_do_not_invent_audio_motion_or_local_edit_capabilities(self):
        response = {"verdict": "uncertain", "issues": ["[待核实] 未提供完整视频动作证据。"],
                    "observed_end_state": "未知", "repair_prompt": "建议另行后期处理，尚未执行。"}
        with patch.object(Brain, "_call", return_value=response) as call:
            Brain().review({}, [Path("synthetic-frame.png")])
        instruction = call.call_args.args[0]
        for rule in ("未播放完整视频或听到音轨", "不得评价配乐、音色、对白正确性、口型同步或帧间动态",
                     "不能用三张图片证明完整碰撞过程或连续两秒可读", "实际是整段新任务",
                     "没有局部编辑、遮罩修补、精确文字排版或纸面透视跟踪合成工具",
                     "建议另行后期处理并明确尚未执行"):
            self.assertIn(rule, instruction)

    def test_human_accepted_previous_version_takes_precedence_over_old_fail(self):
        context = accepted_context()
        original = copy.deepcopy(context)
        response = {"verdict": "pass", "issues": [], "observed_end_state": "人物从桌右侧起身。", "repair_prompt": ""}
        with patch.object(Brain, "_call", return_value=response) as call:
            Brain().review(context, [Path("synthetic-frame.png")])
        instruction, actual_context, _, _ = call.call_args.args
        self.assertEqual(actual_context, original)
        self.assertEqual(context, original)
        for rule in ("previous_actual.human_decision", "accepted_deviations", "不等于用户逐项证实了错误",
                     "即使其旧 GPT 报告为 fail", "不要再次把已接受的前段偏差当错",
                     "不能自行扩大或撤销用户的决定"):
            self.assertIn(rule, instruction)
        self.assertIn("GPT 的检查结论和修改建议不构成人工批准，也不授权返工", SYSTEM)

    def test_refinement_keeps_human_baseline_and_one_whole_segment_request(self):
        context = accepted_context()
        shot = {"id": "s02", "duration": 4, "prompt": "人物从桌右侧起身。", "dialogue": ""}
        original = copy.deepcopy({"shot": shot, "context": context})
        response = {"prompt": "从桌右侧起身。", "dialogue": "", "continuity_notes": "接续人工接受的前段。"}
        with patch.object(Brain, "_call", return_value=response) as call:
            Brain().refine_shot(shot, context, [])
        instruction, data, _, _ = call.call_args.args
        self.assertEqual(data, original)
        self.assertEqual({"shot": shot, "context": context}, original)
        for rule in ("已人工通过的实际前段是连续性基准，即使旧 GPT 报告曾判 fail",
                     "用户接受版本不等于逐项证实这些疑点", "不把原有已接受偏差重新列为必须修复项",
                     "一次人工返工仅生成一个新版本", "不能因 GPT 检查报告自行安排下一轮生成",
                     "后期建议写入 continuity_notes", "建议另行后期处理，尚未执行"):
            self.assertIn(rule, instruction)


if __name__ == "__main__":
    unittest.main()
