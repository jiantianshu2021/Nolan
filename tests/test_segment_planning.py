"""GPT planning contracts for multi-shot generation segments; no model calls."""
import copy
import unittest
from unittest.mock import patch

from norlan import core
from norlan.brain import Brain, BrainError, PLAN_SCHEMA, REFINE_SCHEMA, validate_plan
from norlan.segments import SEGMENT_PLAN_RULES
from test_providers import plan


def multi_shot_plan():
    value = plan()
    shot = value["episodes"][0]["shots"][0]
    shot.update(duration=12, subshots=[
        {"start": 0, "end": 5, "prompt": "中景，人物持信封向右走，镜头平移跟随。"},
        {"start": 5, "end": 9, "prompt": "手部近景，右手将信封放上桌；脚步停止，环境底声持续。"},
        {"start": 9, "end": 12, "prompt": "面部近景，抬眼看门外，接续说完原对白；不重做放信封动作。"},
    ])
    return value


class SegmentPlanningTests(unittest.TestCase):
    def test_new_plan_uses_shared_strategy_and_requires_subshot_timeline(self):
        value = multi_shot_plan()
        with patch.object(Brain, "_call", return_value=value) as call:
            result = Brain().plan("人物拿着信封走到桌边，望向门外。")
        instruction, data, schema = call.call_args.args
        self.assertIn(SEGMENT_PLAN_RULES, instruction)
        self.assertIn("10–15", instruction)
        self.assertIn("2–3", instruction)
        self.assertIn("subshots", instruction)
        self.assertEqual(data["novel"], "人物拿着信封走到桌边，望向门外。")
        self.assertIs(schema, PLAN_SCHEMA)
        shot_schema = schema["properties"]["episodes"]["items"]["properties"]["shots"]["items"]
        self.assertIn("subshots", shot_schema["required"])
        self.assertFalse(shot_schema["properties"]["subshots"]["items"]["additionalProperties"])
        self.assertEqual(result, value)
        missing = copy.deepcopy(value)
        missing["episodes"][0]["shots"][0].pop("subshots")
        with self.assertRaises(BrainError):
            validate_plan(missing)

    def test_invalid_model_timeline_is_rejected_before_installation(self):
        for start, end in ((6, 9), (4, 9), (5, 5), (5, 13)):
            value = multi_shot_plan()
            value["episodes"][0]["shots"][0]["subshots"][1].update(start=start, end=end)
            with self.subTest(start=start, end=end), self.assertRaises(BrainError):
                validate_plan(value)
        value = multi_shot_plan()
        value["episodes"][0]["shots"][0]["subshots"][2]["prompt"] = "a" * 2001
        with self.assertRaises(BrainError):
            validate_plan(value)

    def test_legacy_stored_plan_remains_valid_without_new_model_fields(self):
        value = plan()
        value["episodes"][0]["shots"][0].pop("subshots")
        before = copy.deepcopy(value)
        core.validate_plan(value)
        self.assertEqual(value, before)

    def test_refinement_preserves_approved_structure_and_continuous_audio_contract(self):
        for legacy in (False, True):
            shot = multi_shot_plan()["episodes"][0]["shots"][0]
            if legacy:
                shot.pop("subshots")
            original = copy.deepcopy(shot)
            response = {"prompt": "合成提示词", "dialogue": shot["dialogue"], "continuity_notes": "连续"}
            with self.subTest(legacy=legacy), patch.object(Brain, "_call", return_value=response) as call:
                self.assertEqual(Brain().refine_shot(shot, {}, []), response)
            instruction, data, schema, images = call.call_args.args
            self.assertEqual(data["shot"], original)
            self.assertEqual(shot, original)
            self.assertIs(schema, REFINE_SCHEMA)
            self.assertEqual(images, [])
            for required in ("数量、顺序、start/end 和 duration 完全不变", "不重说台词", "环境底声连续",
                             "整集后期", "旧项目没有 subshots", "一次人工返工仅生成一个新版本"):
                self.assertIn(required, instruction)


if __name__ == "__main__":
    unittest.main()
