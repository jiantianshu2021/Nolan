"""Synthetic system-voice editing contracts; no paid calls or media generation."""
import copy
import json
import unittest
from unittest.mock import patch

from norlan.brain import (Brain, BrainError, REFINE_SCHEMA, SYSTEM_BROADCAST_EDITORIAL_RULES,
                          _validate, refinement_task_context, revision_task_context)


SOURCE = "技术须以民用产品发布。普通售价2999元。震惊值升级加工台；真实使用满意值兑换新图纸。"
BRIEF = "技术须以民用产品发布。"


def shot():
    return {"id": "s08", "duration": 15, "scene_id": "room", "asset_ids": ["hero", "role", "system_voice"],
            "prompt": "人物放下工具，系统框出现，女声读系统全文。", "dialogue": "楚阳：要我喂你？\n系统播报员：" + SOURCE,
            "continuity_notes": "保持人物位置。",
            "subshots": [{"start": 0, "end": 5, "prompt": "中景，楚阳放下工具并问原句。"},
                         {"start": 5, "end": 15, "prompt": "近景，系统界面出现并读全文。"}]}


def context():
    return {"novel": "不应随细化上下文发送的小说。" * 1000,
            "assets": [{"id": "hero", "kind": "character", "name": "楚阳"},
                       {"id": "role", "kind": "character", "name": "系统播报员", "voice_only": True},
                       {"id": "system_voice", "kind": "voice", "name": "系统播报员声音", "character_id": "role",
                        "audition_speaker": "系统", "status": "approved"}],
            "audio_policy": {"system_broadcast": {"enabled": True, "character_id": "role", "voice_id": "system_voice",
                                                  "instruction": "旧规则：每次系统框逐字完整朗读。"}},
            "episodes": [{"id": "ep01", "shots": [shot()]}]}


def candidate():
    return {"prompt": "0–5秒中景，楚阳0.4–1.6秒问：要我喂你？5–15秒近景，系统界面5秒出现。"
                      "系统女声5.2–8.2秒播报：" + BRIEF + "系统界面完整显示至15秒：" + SOURCE,
            "dialogue": "楚阳：要我喂你？\n系统播报员：" + BRIEF,
            "continuity_notes": "人物对白不变；系统仅播关键规则，全部细则保留在画面。等待完整提示词审批。",
            "broadcast_proposal": {"speaker": "系统播报员", "source_text": SOURCE, "spoken_text": BRIEF,
                                   "display_text": SOURCE, "summary": "只播发布要求，价格及积分规则留在界面，为人物反应留时间。"},
            "timing_proposal": {"summary": "原15秒及第5秒切点不变，短播5.2–8.2秒，仍待最终提示词确认。",
                                "subshots": [{"start": 0, "end": 5, "prompt": "中景，楚阳放下工具，0.4–1.6秒问：要我喂你？"},
                                             {"start": 5, "end": 15, "prompt": "近景，系统界面5秒出现，显示：" + SOURCE + "女声5.2–8.2秒只读：" + BRIEF}],
                                "audio_windows": [{"start": 0.4, "end": 1.6, "speaker": "楚阳", "text": "要我喂你？"},
                                                  {"start": 5.2, "end": 8.2, "speaker": "系统播报员", "text": BRIEF}]}}


class SystemBroadcastBrainTests(unittest.TestCase):
    def test_brief_voice_and_full_rules_are_one_unapproved_consistent_candidate(self):
        source, data, output = shot(), context(), candidate()
        before = copy.deepcopy((source, data))
        _validate(output, REFINE_SCHEMA)
        with patch.object(Brain, "_call", return_value=output) as model:
            result = Brain().refine_shot(source, data, [])
        self.assertEqual(result, output)
        self.assertEqual((source, data), before)
        instruction, payload, schema, _ = model.call_args.args
        self.assertIs(schema, REFINE_SCHEMA)
        self.assertIn(SYSTEM_BROADCAST_EDITORIAL_RULES, instruction)
        for requirement in ("忠实压缩", "不为常规选句、跨镜或具体秒数增加人工选择步骤", "display_text 必须与 source_text 完全相同",
                            "人物原对白逐字保持", "等待用户点击确认提示词后才提交一次视频"):
            self.assertIn(requirement, instruction)
        self.assertEqual(payload["shot"]["dialogue"], source["dialogue"])
        self.assertNotIn("novel", payload["context"])

    def test_both_model_projections_replace_legacy_full_read_policy_without_mutating_it(self):
        data = context()
        before = copy.deepcopy(data)
        for projected in (refinement_task_context(shot(), data), revision_task_context(shot(), data)):
            policy = projected["audio_policy"]["system_broadcast"]
            self.assertEqual(policy["editorial_mode"], "key_points")
            self.assertIn("系统播报员", policy["speaker_labels"])
            self.assertNotIn("旧规则：", policy["instruction"])
            self.assertIn("不要求用户另行确认具体秒数", policy["instruction"])
        self.assertEqual(data, before)

    def test_brief_proposal_cannot_edit_a_character_or_omit_visible_rule_details(self):
        for mutate in (lambda value: value.update(dialogue="楚阳：改了人物对白。\n系统播报员：" + BRIEF),
                       lambda value: value["broadcast_proposal"].update(display_text=BRIEF),
                       lambda value: value["broadcast_proposal"].update(spoken_text="售价3999元。")):
            output = candidate()
            mutate(output)
            with self.subTest(output=output), patch.object(Brain, "_call", return_value=output):
                with self.assertRaises(BrainError):
                    Brain().refine_shot(shot(), context(), [])

    def test_strict_nullable_fields_support_old_three_field_transport_result(self):
        output = {"prompt": "原对白不变。", "dialogue": shot()["dialogue"], "continuity_notes": "仅合成测试。"}
        with patch.object(Brain, "_codex", return_value=output):
            result = Brain().refine_shot(shot(), {}, [])
        self.assertEqual(result, output)
        self.assertEqual(set(REFINE_SCHEMA["required"]), set(REFINE_SCHEMA["properties"]))
        self.assertLess(len(json.dumps(refinement_task_context(shot(), context()), ensure_ascii=False)), 4000)


if __name__ == "__main__":
    unittest.main()
