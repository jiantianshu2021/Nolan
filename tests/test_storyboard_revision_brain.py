"""Offline GPT contracts for one-feedback asset and internal-storyboard repair."""
import copy
import json
import unittest
from unittest.mock import patch

from norlan.brain import (Brain, BrainError, REVISION_SCHEMA, _validate,
                          task_shot_summary)


def target():
    return {"id": "s16", "duration": 5, "scene_id": "workshop", "asset_ids": ["hero", "workshop"],
            "prompt": "旧结构：5秒单镜，主角拿起工具。", "dialogue": "主角：先试试。",
            "continuity_notes": "承接上一段站位。", "status": "planned", "version": 1,
            "feedback": "改成两个镜头，先看工具，再看人物。",
            "subshots": [{"start": 0, "end": 5, "prompt": "单镜，主角拿起工具并说原对白。"}]}


def context(shot):
    return {"assets": [{"id": "hero", "kind": "character", "name": "主角", "prompt": "已批准人物。"},
                       {"id": "workshop", "kind": "location", "name": "车间", "prompt": "已批准车间。"}],
            "episodes": [{"id": "ep01", "shots": [shot, {**target(), "id": "s17"}]}]}


def revision(storyboard=True):
    response = {"needs_assets": False, "additions": [], "shot_updates": [], "summary": "按反馈准备待审草稿。",
                "blockers": [], "audio_policy": {"system_broadcast": {
                    "enabled": False, "character_id": "", "voice_id": "", "instruction": ""}},
                "storyboard_revision": None}
    if storyboard:
        response["storyboard_revision"] = {
            "summary": "本段改为两个镜头，2秒看工具、3秒看人物，待完整提示词确认。",
            "duration": 5,
            "subshots": [{"start": 0, "end": 2, "prompt": "工具特写，主角右手拿起工具，无对白。"},
                         {"start": 2, "end": 5, "prompt": "切主角近景，原站位和视线承接，主角：先试试。"}]}
    return response


class StoryboardRevisionBrainTests(unittest.TestCase):
    def plan(self, response=None, shot=None):
        shot = target() if shot is None else shot
        response = revision() if response is None else response
        with patch.object(Brain, "_call", return_value=copy.deepcopy(response)) as call:
            result = Brain().plan_revision(shot, context(shot), shot["feedback"])
        return result, call

    def test_same_feedback_can_plan_two_shots_without_new_assets_or_mutating_original(self):
        shot, response = target(), revision()
        before = copy.deepcopy(shot)
        result, call = self.plan(response, shot)
        self.assertEqual(shot, before)
        self.assertFalse(result["needs_assets"])
        self.assertEqual(result["storyboard_revision"], response["storyboard_revision"])
        self.assertEqual(result["storyboard_revision"]["subshots"][-1]["end"], shot["duration"])
        instruction, payload, schema = call.call_args.args
        self.assertIs(schema, REVISION_SCHEMA)
        self.assertEqual(payload["feedback"], shot["feedback"])
        for requirement in ("按完整语义自主判断", "不靠固定关键词", "不要求用户换表单", "不再要求用户确认秒数",
                            "不拆成多个生成任务", "人物 canonical dialogue", "等待最终提示词确认"):
            self.assertIn(requirement, instruction)

    def test_asset_and_storyboard_plans_are_returned_in_the_same_call(self):
        response = revision()
        response.update(needs_assets=True, additions=[{
            "id": "new_cup", "kind": "prop", "name": "茶杯", "prompt": "新茶杯。", "character_id": "",
            "audition_speaker": "", "audition_text": "", "audition_reason": "", "audition_source": "",
            "voice_only": False}], shot_updates=[{
                "id": "s16", "add_asset_ids": ["new_cup"], "prompt": "主角按两镜草稿拿起茶杯。",
                "dialogue": target()["dialogue"], "continuity_notes": "待资产审批，再准备完整提示词。"}])
        result, call = self.plan(response)
        self.assertEqual(call.call_count, 1)
        self.assertTrue(result["needs_assets"])
        self.assertIsNotNone(result["storyboard_revision"])
        self.assertIn("先走资产人工确认", call.call_args.args[0])

    def test_nullable_wire_field_is_required_but_old_cache_shape_is_compatible(self):
        self.assertIn("storyboard_revision", REVISION_SCHEMA["required"])
        self.assertEqual(set(REVISION_SCHEMA["required"]), set(REVISION_SCHEMA["properties"]))
        legacy = revision(False)
        del legacy["storyboard_revision"]
        _validate(legacy, REVISION_SCHEMA)
        actual, _ = self.plan(legacy)
        self.assertEqual(actual, legacy)
        self.assertNotIn("storyboard_revision", actual)
        with patch.object(Brain, "_codex", return_value=legacy):
            actual = Brain().plan_revision(target(), context(target()), "只改表演。")
        self.assertEqual(actual, legacy)

    def test_null_preserves_existing_schedule_and_can_describe_no_structural_change(self):
        for summary in ("不重新分镜，只调整表演。", "无需重新分镜，保留现有切点。", "保留分镜。"):
            response = revision(False)
            response["summary"] = summary
            with self.subTest(summary=summary):
                result, _ = self.plan(response)
                self.assertIsNone(result["storyboard_revision"])

    def test_structural_claim_cannot_be_hidden_in_summary_or_prompt(self):
        for field, claim in (("summary", "本次重新分镜，两个镜头。"),
                             ("prompt", "改成两个镜头，先看工具，再看主角。"),
                             ("continuity_notes", "摄影切点改为2秒。")):
            response = revision(False)
            if field == "summary":
                response[field] = claim
            else:
                response["shot_updates"] = [{"id": "s16", "add_asset_ids": [], "prompt": "拿起工具。",
                    "dialogue": target()["dialogue"], "continuity_notes": "同一场景。", field: claim}]
            with self.subTest(field=field), self.assertRaisesRegex(BrainError, "没有提供 storyboard_revision"):
                self.plan(response)

    def test_only_the_exact_draft_keys_are_accepted(self):
        for key, value in (("id", "s17"), ("approved", True), ("scene_id", "elsewhere")):
            response = revision()
            response["storyboard_revision"][key] = value
            with self.subTest(key=key), self.assertRaises(BrainError):
                self.plan(response)
        response = revision()
        response["storyboard_revision"]["subshots"][0]["approved"] = True
        with self.assertRaises(BrainError):
            self.plan(response)

    def test_blank_long_or_null_character_summary_is_rejected(self):
        for summary in (" ", "x" * 2001, "说明\x00", 123):
            response = revision()
            response["storyboard_revision"]["summary"] = summary
            with self.subTest(summary=repr(summary)[:30]), self.assertRaises(BrainError):
                self.plan(response)

    def test_timing_must_exactly_cover_new_duration_with_one_to_three_shots(self):
        schedules = [[], [{"start": 0, "end": 1, "prompt": "镜头。"}] * 4,
                     [{"start": .1, "end": 5, "prompt": "开头缺口。"}],
                     [{"start": 0, "end": 4.9, "prompt": "尾部缺口。"}],
                     [{"start": 0, "end": 2, "prompt": "一。"}, {"start": 2.1, "end": 5, "prompt": "二。"}],
                     [{"start": 0, "end": 3, "prompt": "一。"}, {"start": 2, "end": 5, "prompt": "二。"}],
                     [{"start": 0, "end": 6, "prompt": "擅自延长。"}],
                     [{"start": 0, "end": 0, "prompt": "零长镜。"}, {"start": 0, "end": 5, "prompt": "二。"}]]
        for subshots in schedules:
            response = revision()
            response["storyboard_revision"]["subshots"] = subshots
            with self.subTest(subshots=subshots), self.assertRaises(BrainError):
                self.plan(response)

    def test_bool_nonfinite_or_overflowing_times_are_rejected(self):
        for value in (True, False, float("nan"), float("inf"), float("-inf"), "2", 10 ** 1000):
            response = revision()
            response["storyboard_revision"]["subshots"][0]["end"] = value
            with self.subTest(value=value), self.assertRaises(BrainError):
                self.plan(response)

    def test_gpt_may_extend_target_duration_without_replanning_other_segments(self):
        shot, response = target(), revision()
        response["storyboard_revision"].update(duration=10, subshots=[
            {"start": 0, "end": 4, "prompt": "工具特写。"},
            {"start": 4, "end": 10, "prompt": "主角近景，说：先试试。"}])
        before = copy.deepcopy(shot)
        result, call = self.plan(response, shot)
        self.assertEqual(result["storyboard_revision"]["duration"], 10)
        self.assertEqual(shot, before)
        for requirement in ("不必保留旧 duration", "整集总时长随目标片段的新时长相应变化", "确实超过15秒",
                            "不把 duration 假写15来掩盖超时", "不得为凑回旧130秒"):
            self.assertIn(requirement, call.call_args.args[0])

    def test_duration_must_match_engine_integer_limit(self):
        for duration in (3, 16, 10.0, 4.5, True, "10", float("nan"), float("inf"), 10 ** 1000):
            response = revision()
            response["storyboard_revision"]["duration"] = duration
            with self.subTest(duration=repr(duration)[:20]), self.assertRaises(BrainError):
                self.plan(response)

    def test_structural_revision_cannot_rewrite_character_dialogue_in_shot_updates(self):
        for dialogue in ("主角：换一句台词。", "其他角色：先试试。", "", "主角：先试试。\n主角：新增台词。"):
            response = revision()
            response["shot_updates"] = [{"id": "s16", "add_asset_ids": [], "prompt": "工具特写再切主角。",
                "dialogue": dialogue, "continuity_notes": "同一场景。"}]
            with self.subTest(dialogue=dialogue), self.assertRaisesRegex(BrainError, "人物原对白"):
                self.plan(response)

    def test_new_duration_is_used_for_capacity_when_system_source_is_added(self):
        shot, response = target(), revision()
        shot["dialogue"] = "主角：这台机器已经准备妥当现在可以开始正常工作了。"
        data = context(shot)
        data["assets"].extend([
            {"id": "system_role", "kind": "character", "name": "系统播报员", "voice_only": True},
            {"id": "system_voice", "kind": "voice", "character_id": "system_role", "name": "系统声音", "audition_speaker": "系统"}])
        response["audio_policy"]["system_broadcast"].update(enabled=True, character_id="system_role",
            voice_id="system_voice", instruction="只播关键内容，完整来源保留界面。")
        response["storyboard_revision"].update(duration=10, subshots=[
            {"start": 0, "end": 4, "prompt": "工具特写，人物完整说原对白。"},
            {"start": 4, "end": 10, "prompt": "切人物近景，系统显示详细规则。"}])
        source = "系统播报员：绑定成功，以下全部详细规则仍完整保留作为系统短播来源。"
        response["shot_updates"] = [{"id": shot["id"], "add_asset_ids": ["system_role", "system_voice"],
            "prompt": "新十秒两镜头草稿。", "dialogue": shot["dialogue"] + "\n" + source, "continuity_notes": "使用已批准系统声。"}]
        with patch.object(Brain, "_call", return_value=response):
            result = Brain().plan_revision(shot, data, "两镜头并加原系统播报，由GPT安排时长。")
        self.assertEqual(result["blockers"], [])
        self.assertTrue(result["shot_updates"][0]["dialogue"].endswith(source))

    def test_internal_prompt_must_be_useful_bounded_text(self):
        for prompt in ("", " ", "x" * 2001, "画面\x00", None):
            response = revision()
            response["storyboard_revision"]["subshots"][0]["prompt"] = prompt
            with self.subTest(prompt=repr(prompt)[:30]), self.assertRaises(BrainError):
                self.plan(response)

    def test_one_or_three_new_internal_shots_are_valid_with_fractional_cut_points(self):
        for subshots in ([{"start": 0, "end": 5, "prompt": "合并为一个完整镜头。"}],
                         [{"start": 0, "end": 1.25, "prompt": "工具特写。"},
                          {"start": 1.25, "end": 3.5, "prompt": "手拿工具。"},
                          {"start": 3.5, "end": 5, "prompt": "主角说：先试试。"}]):
            response = revision()
            response["storyboard_revision"]["subshots"] = subshots
            with self.subTest(subshots=subshots):
                result, _ = self.plan(response)
                self.assertEqual(result["storyboard_revision"]["subshots"], subshots)

    def test_present_draft_metadata_is_projected_without_before_or_history_growth(self):
        shot = target()
        marker = "RUNTIME_HISTORY_SHOULD_NOT_ENTER_MODEL"
        shot["subshots"] = revision()["storyboard_revision"]["subshots"]
        shot["storyboard_revision"] = {"summary": "本次两镜草稿。", "signature": "synthetic-signature",
            "subshots": copy.deepcopy(shot["subshots"]), "before": {"prompt": marker * 10000},
            "history": [{"raw": marker * 10000}], "binding": {"raw": marker * 10000}}
        compact = task_shot_summary(shot)
        self.assertEqual(compact["storyboard_revision"], {"summary": "本次两镜草稿。", "signature": "synthetic-signature"})
        self.assertNotIn(marker, json.dumps(compact))
        response = {"prompt": "0–2秒工具特写，2–5秒人物近景。主角：先试试。", "dialogue": shot["dialogue"],
                    "continuity_notes": "待完整提示词确认。"}
        with patch.object(Brain, "_call", return_value=response) as call:
            Brain().refine_shot(shot, context(shot), [])
        instruction, payload, _, _ = call.call_args.args
        self.assertEqual(payload["shot"]["subshots"], shot["subshots"])
        self.assertNotIn(marker, json.dumps(payload))
        self.assertLess(len(json.dumps(payload, ensure_ascii=False)), 3000)
        for requirement in ("优先于旧 shot.prompt", "不能在最终提示词中恢复旧分镜", "不重复要求用户确认秒数",
                            "没有本次新分镜时", "细化阶段都不能再次自行变动"):
            self.assertIn(requirement, instruction)


if __name__ == "__main__":
    unittest.main()
