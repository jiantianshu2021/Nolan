"""Synthetic next-segment continuity planning; no model or media requests."""
import copy
import json
import unittest
from unittest.mock import patch

from norlan.brain import Brain, BrainError, REVISION_SCHEMA, _validate, revision_task_context


def current():
    return {"id": "s01", "duration": 6, "scene_id": "yard", "asset_ids": ["father", "yard", "board"],
            "prompt": "父亲正在坠落，飞板接近。", "dialogue": "", "continuity_notes": "保持厂院朝向。",
            "subshots": [{"start": 0, "end": 3, "prompt": "父亲坠落。"},
                         {"start": 3, "end": 6, "prompt": "飞板接近父亲。"}],
            "status": "planned", "version": 3, "revision": 2,
            "feedback": "先快速坠落，切父亲绝望表情，再切回飞板接住，父亲摇晃后站稳。"}


def following():
    return {"id": "s02", "duration": 15, "scene_id": "yard", "asset_ids": ["father", "yard", "board"],
            "prompt": "父亲衣摆初托，趴稳后查看飞板。", "dialogue": "父亲：这是什么？",
            "continuity_notes": "飞板托住父亲，保持厂院朝向。",
            "subshots": [{"start": 0, "end": 7, "prompt": "飞板初托父亲，父亲趴稳。"},
                         {"start": 7, "end": 15, "prompt": "父亲看飞板，说：这是什么？"}],
            "status": "planned", "version": 0, "revision": 0, "feedback": ""}


def context():
    return {"novel": "合成测试剧本：父亲坠落后被飞板接住，随后查看飞板。",
            "assets": [{"id": "father", "kind": "character", "name": "父亲", "prompt": "固定父亲。"},
                       {"id": "yard", "kind": "location", "name": "厂院", "prompt": "固定厂院。"},
                       {"id": "board", "kind": "prop", "name": "飞板", "prompt": "固定飞板。"}],
            "episodes": [{"id": "ep02", "shots": [current(), following()]}]}


def response():
    return {"needs_assets": False, "additions": [], "shot_updates": [],
            "audio_policy": {"system_broadcast": {"enabled": False, "character_id": "", "voice_id": "", "instruction": ""}},
            "storyboard_revision": {"summary": "本段改为三镜，结尾父亲摇晃后站稳，待提示词确认。", "duration": 10,
                "subshots": [{"start": 0, "end": 3, "prompt": "父亲快速坠落。"},
                             {"start": 3, "end": 5, "prompt": "父亲绝望的脸部近景。"},
                             {"start": 5, "end": 10, "prompt": "飞板接住父亲，他摇晃后站稳。"}]},
            "future_continuity_updates": [{"id": "s02", "summary": "从已经站稳继续，不重演初托与趴稳，仍待单独确认完整提示词。",
                "prompt": "承接父亲已站稳的姿态，继续查看飞板，不重复坠落与初托。",
                "continuity_notes": "承接上一段站稳结尾，保留后续查看飞板剧情。",
                "subshots": [{"start": 0, "end": 7, "prompt": "父亲已站稳，低头观察飞板。"},
                             {"start": 7, "end": 15, "prompt": "父亲看飞板，说：这是什么？"}]}],
            "summary": "本段按最新意见安排，下一段仅协调开头姿态；两段完整提示词均须人工确认。",
            "blockers": []}


class FutureContinuityBrainTests(unittest.TestCase):
    def plan(self, result=None, data=None):
        data = context() if data is None else data
        target = data["episodes"][0]["shots"][0]
        result = response() if result is None else result
        with patch.object(Brain, "_call", return_value=copy.deepcopy(result)) as model:
            actual = Brain().plan_revision(target, data, target["feedback"])
        return actual, model

    def test_wire_field_is_strict_required_and_legacy_omission_remains_empty(self):
        self.assertIn("future_continuity_updates", REVISION_SCHEMA["required"])
        legacy = response()
        del legacy["future_continuity_updates"]
        _validate(legacy, REVISION_SCHEMA)
        actual, _ = self.plan(legacy)
        self.assertEqual(actual, legacy)
        self.assertNotIn("future_continuity_updates", actual)
        for value in (None, {}, "", [response()["future_continuity_updates"][0]] * 2):
            invalid = response()
            invalid["future_continuity_updates"] = value
            with self.subTest(value=repr(value)[:80]), self.assertRaises(BrainError):
                self.plan(invalid)

    def test_new_current_ending_can_coordinate_only_unstarted_next_draft(self):
        data, proposed = context(), response()
        before = copy.deepcopy(data)
        actual, model = self.plan(proposed, data)
        self.assertEqual(actual, proposed)
        self.assertEqual(data, before)
        self.assertEqual(actual["blockers"], [])
        self.assertEqual(actual["storyboard_revision"]["duration"], 10)
        self.assertEqual(actual["future_continuity_updates"][0]["subshots"][-1]["end"], 15)
        instructions = model.call_args.args[0]
        for rule in ("最新用户明确的动作和结尾优先", "消除重演", "不要再次要求用户确认是否允许协调",
                     "子镜数量和每个 start/end", "单独确认完整提示词", "fixed_next_shot", "blocker_repair"):
            self.assertIn(rule, instructions)

    def test_future_update_cannot_carry_execution_duration_asset_or_dialogue_fields(self):
        for field, value in (("duration", 10), ("scene_id", "elsewhere"), ("dialogue", "新对白"),
                             ("asset_ids", []), ("version", 2), ("approved", True), ("job_id", "paid-task")):
            invalid = response()
            invalid["future_continuity_updates"][0][field] = value
            with self.subTest(field=field), self.assertRaises(BrainError):
                self.plan(invalid)

    def test_next_schedule_count_and_cut_times_cannot_change(self):
        for schedule in ([{"start": 0, "end": 15, "prompt": "合并成一镜。"}],
                         [{"start": 0, "end": 8, "prompt": "前镜。"}, {"start": 8, "end": 15, "prompt": "后镜。"}]):
            invalid = response()
            invalid["future_continuity_updates"][0]["subshots"] = schedule
            with self.subTest(schedule=schedule), self.assertRaises(BrainError):
                self.plan(invalid)

    def test_fixed_or_user_edited_neighbor_is_never_eligible(self):
        for fields in ({"version": 1}, {"status": "generating"}, {"file": "existing.mp4"},
                       {"job_id": "task"}, {"attempts": [{}]}, {"prompt_candidate": {"id": "p"}},
                       {"prompt_approval": {"approved": True}}, {"revision": 1},
                       {"feedback": "保留这个已指定开头。"}, {"pending_asset_amendment": "a"}):
            data = context()
            data["episodes"][0]["shots"][1].update(fields)
            selected = revision_task_context(data["episodes"][0]["shots"][0], data)
            with self.subTest(fields=fields):
                self.assertNotIn("eligible_next_shot", selected)
                self.assertEqual(selected["fixed_next_shot"]["id"], "s02")
                with self.assertRaises(BrainError):
                    self.plan(data=data)

    def test_cannot_skip_adjacent_segment_or_cross_episode_boundary(self):
        data = context()
        data["episodes"][0]["shots"].append({**following(), "id": "s03"})
        invalid = response()
        invalid["future_continuity_updates"][0]["id"] = "s03"
        with self.assertRaises(BrainError):
            self.plan(invalid, data)
        data = context()
        neighbor = data["episodes"][0]["shots"].pop()
        data["episodes"].append({"id": "ep03", "shots": [neighbor]})
        selected = revision_task_context(data["episodes"][0]["shots"][0], data)
        self.assertNotIn("eligible_next_shot", selected)
        with self.assertRaises(BrainError):
            self.plan(data=data)

    def test_context_exposes_only_small_dependency_and_current_blocker_repair(self):
        data = context()
        following_shot = data["episodes"][0]["shots"][1]
        following_shot["history"] = [{"prompt": "OLD_MEDIA_HISTORY" * 10000}]
        following_shot["reference_manifest"] = {"context": "NESTED_CONTEXT" * 10000}
        data["blocker_repair"] = {"summary": "旧开头冲突。", "blockers": ["旧规则不能协调。"],
            "planning_policy_version": 1, "history": ["OLD_BLOCKER_HISTORY" * 10000]}
        selected = revision_task_context(data["episodes"][0]["shots"][0], data)
        self.assertEqual(selected["eligible_next_shot"]["subshots"], following_shot["subshots"])
        self.assertEqual(set(selected["blocker_repair"]), {"summary", "blockers", "planning_policy_version"})
        self.assertEqual(selected["future_continuity_policy"]["scope"], "same_episode_immediate_next_only")
        encoded = json.dumps(selected, ensure_ascii=False)
        for hidden in ("OLD_MEDIA_HISTORY", "NESTED_CONTEXT", "OLD_BLOCKER_HISTORY"):
            self.assertNotIn(hidden, encoded)
        self.assertLess(len(encoded), 6000)

    def test_validation_baselines_do_not_leak_into_model_result(self):
        proposed = response()
        with patch("norlan.future_continuity.validate_updates", return_value=[{"id": "s02", "baseline": "local-only"}]) as check:
            actual, _ = self.plan(proposed)
        self.assertEqual(actual, proposed)
        self.assertNotIn("baseline", actual["future_continuity_updates"][0])
        self.assertEqual(check.call_count, 1)


if __name__ == "__main__":
    unittest.main()
