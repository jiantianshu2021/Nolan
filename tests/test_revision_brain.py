"""Synthetic asset-amendment planning contracts; never call a real model."""
import copy
from pathlib import Path
import unittest
from unittest.mock import patch

from norlan.brain import Brain, BrainError, REVISION_SCHEMA, _validate


SYSTEM_TEXT = "赠送超维加工台、悬浮滑板图纸。仅绑定者可操作。"
SYSTEM_SOURCE = "系统字幕：" + SYSTEM_TEXT


def asset(id, kind, name, prompt, **extra):
    return {"id": id, "kind": kind, "name": name, "prompt": prompt,
            "character_id": "", "audition_text": "", "audition_reason": "", "audition_source": "",
            "voice_only": False, "audition_speaker": "", **extra}


def shot(id="s06", **extra):
    return {"id": id, "duration": 10, "prompt": "系统框出现，系统无配音。", "dialogue": "",
            "asset_ids": ["hero", "room"], "continuity_notes": "场景保持。", "version": 0,
            "subshots": [{"start": 0, "end": 10, "prompt": "任务框弹出，不念任务框。"}], **extra}


def context():
    return {"novel": "合成回归文本。" + SYSTEM_SOURCE,
            "assets": [asset("hero", "character", "测试角色", "漫画人物"),
                       asset("room", "location", "测试房间", "漫画房间")],
            "episodes": [{"id": "ep01", "shots": [shot(), shot("s07")]}]}


def revision():
    return {"needs_assets": True,
            "additions": [asset("announcer", "character", "系统播报员", "只在画外播音，无人物形象。", voice_only=True),
                          asset("system_voice", "voice", "系统播报员声音", "成熟女声，中音，音色清澈。",
                                character_id="announcer", audition_speaker="系统", audition_text=SYSTEM_TEXT,
                                audition_source=SYSTEM_SOURCE,
                                audition_reason="系统原文为完整陈述，依用户要求将系统字幕改为中性播音，归属明确。")],
            "shot_updates": [{"id": "s06", "add_asset_ids": ["announcer", "system_voice"],
                              "prompt": "系统框弹出，画外系统播报员逐字播报：" + SYSTEM_TEXT,
                              "dialogue": "系统播报员：" + SYSTEM_TEXT,
                              "continuity_notes": "保持原时间表，新系统播报规则覆盖旧不念任务框限制。"}],
            "audio_policy": {"system_broadcast": {"enabled": True, "character_id": "announcer",
                                                  "voice_id": "system_voice",
                                                  "instruction": "每次系统框弹出复用同一批准声线逐字画外播报，其他对白保留，无配乐。"}},
            "summary": "先新增系统画外声源和声音，回到资产人工确认后才返修。", "blockers": []}


class RevisionBrainTests(unittest.TestCase):
    def call(self, response=None, target=None, data=None, feedback="新增女性系统播报员，以后系统框均需播报。"):
        response = copy.deepcopy(revision() if response is None else response)
        with patch.object(Brain, "_call", return_value=response) as call:
            result = Brain().plan_revision(target or shot(), data or context(), feedback)
        return result, call

    def test_new_system_voice_uses_original_source_without_mutating_original_plan(self):
        target, data = shot(version=2, file="old.mp4"), context()
        before = copy.deepcopy((target, data))
        result, call = self.call(target=target, data=data)
        _validate(result, REVISION_SCHEMA)
        self.assertEqual(result["blockers"], [])
        self.assertEqual((target, data), before)
        self.assertTrue(result["additions"][0]["voice_only"])
        self.assertEqual(result["additions"][1]["audition_source"], SYSTEM_SOURCE)
        self.assertEqual(set(result["shot_updates"][0]), {"id", "add_asset_ids", "prompt", "dialogue", "continuity_notes"})
        self.assertNotIn("duration", result["shot_updates"][0])
        self.assertIs(call.call_args.args[2], REVISION_SCHEMA)
        instruction = call.call_args.args[0]
        for rule in ("只规划，不生成、批准或执行", "普通构图", "先回到资产人工确认再返修", "不念任务框",
                     "已有成片的其他片段不变", "不是新取的展示名", "无人物形象、无需图片", "不能为省时删减、改写或加速"):
            self.assertIn(rule, instruction)

    def test_ordinary_rework_does_not_create_assets(self):
        result = {"needs_assets": False, "additions": [], "shot_updates": [],
                  "audio_policy": {"system_broadcast": {"enabled": False, "character_id": "", "voice_id": "", "instruction": ""}},
                  "summary": "仅修改构图，复用原资产。", "blockers": []}
        actual, _ = self.call(result, feedback="镜头拉近一点，保留原来的声音。")
        self.assertEqual(actual, result)

    def test_future_unstarted_task_frame_reuses_same_voice(self):
        response = revision()
        response["shot_updates"].append({**response["shot_updates"][0], "id": "s07"})
        result, _ = self.call(response)
        self.assertEqual(len(result["additions"]), 2)
        self.assertEqual(result["shot_updates"][1]["add_asset_ids"], ["announcer", "system_voice"])

    def test_other_generated_or_prior_shot_is_never_rewritten(self):
        for extra in ({"file": "old.mp4"}, {"version": 1}, {"status": "generating"}):
            response, data = revision(), context()
            data["episodes"][0]["shots"][1].update(extra)
            response["shot_updates"].append({**response["shot_updates"][0], "id": "s07"})
            with self.subTest(extra=extra), self.assertRaises(BrainError):
                self.call(response, data=data)
        response, data = revision(), context()
        data["episodes"][0]["shots"].insert(0, shot("s05"))
        response["shot_updates"].append({**response["shot_updates"][0], "id": "s05"})
        with self.assertRaisesRegex(BrainError, "之前"):
            self.call(response, data=data)

    def test_existing_or_duplicate_asset_ids_cannot_be_overwritten(self):
        for value in ("hero", "room", "ep01", "s06"):
            response = revision()
            response["additions"][0]["id"] = value
            with self.subTest(value=value), self.assertRaisesRegex(BrainError, "ID 已存在"):
                self.call(response)
        response = revision()
        response["additions"].append(copy.deepcopy(response["additions"][0]))
        with self.assertRaises(BrainError):
            self.call(response)

    def test_duration_or_internal_schedule_fields_rejected_by_schema(self):
        for key, value in (("duration", 15), ("subshots", []), ("asset_ids", []), ("scene_id", "room")):
            response = revision()
            response["shot_updates"][0][key] = value
            with self.subTest(key=key), self.assertRaisesRegex(BrainError, "未知项"):
                self.call(response)

    def test_source_cannot_be_fabricated_rewritten_or_pasted_from_feedback(self):
        response = revision()
        response["additions"][1].update(audition_text="这是一句新编的通用测试语句，用来测试女性声音。",
                                        audition_source="系统：这是一句新编的通用测试语句，用来测试女性声音。")
        with self.assertRaisesRegex(BrainError, "逐字选自原剧本"):
            self.call(response)

    def test_short_or_absent_audition_is_a_blocker_not_an_invented_sentence(self):
        response = revision()
        response["additions"][1].update(audition_text="绑定成功", audition_source="系统：绑定成功")
        data = context()
        data["novel"] += "系统：绑定成功"
        result, _ = self.call(response, data=data)
        self.assertIn("过短", " ".join(result["blockers"]))
        response = revision()
        response["additions"][1].update(audition_text="", audition_source="", audition_reason="没有足够平稳的原文。")
        result, _ = self.call(response)
        self.assertIn("先人工补充", " ".join(result["blockers"]))

    def test_full_system_source_is_kept_for_refinement_without_assuming_full_read_duration(self):
        result, _ = self.call(target=shot(duration=4))
        self.assertEqual(result["blockers"], [])
        self.assertEqual(result["shot_updates"][0]["dialogue"], "系统播报员：" + SYSTEM_TEXT)

    def test_impossible_character_dialogue_still_blocks_instead_of_being_cut(self):
        response = revision()
        long_line = "这是角色必须完整说出的原句，不能因为系统采用简播就删掉角色自己的任何对白。"
        response["shot_updates"][0]["dialogue"] = "测试角色：" + long_line + "\n系统播报员：" + SYSTEM_TEXT
        result, _ = self.call(response, target=shot(duration=4))
        self.assertIn("超过既定 4 秒", " ".join(result["blockers"]))
        self.assertIn(long_line, result["shot_updates"][0]["dialogue"])

    def test_existing_system_policy_is_used_for_capacity_when_no_change_response_is_empty(self):
        response, data = revision(), context()
        data["assets"].extend(copy.deepcopy(response["additions"]))
        data["audio_policy"] = copy.deepcopy(response["audio_policy"])
        target = shot(duration=4, asset_ids=["hero", "room", "announcer", "system_voice"])
        data["episodes"][0]["shots"][0] = target
        response.update(needs_assets=False, additions=[])
        response["audio_policy"] = {"system_broadcast": {"enabled": False, "character_id": "", "voice_id": "", "instruction": ""}}
        for line, expects_blocker in (("系统播报员：" + SYSTEM_TEXT, False),
                                      ("测试角色：" + "人物对白必须完整保留，不得借用系统简播规则删减。" * 3 + "\n系统播报员：" + SYSTEM_TEXT, True)):
            with self.subTest(expects_blocker=expects_blocker):
                output = copy.deepcopy(response)
                output["shot_updates"][0]["dialogue"] = line
                result, _ = self.call(output, target=target, data=data, feedback="保持原声线，系统短播由GPT安排。")
                self.assertEqual(bool(result["blockers"]), expects_blocker)
                self.assertEqual(result["shot_updates"][0]["dialogue"], line)
                self.assertEqual(result["audio_policy"], response["audio_policy"])

    def test_voice_bindings_and_offscreen_source_types_are_validated(self):
        mutations = [lambda r: r["additions"][1].update(character_id="room"),
                     lambda r: r["additions"][1].update(voice_only=True),
                     lambda r: r["shot_updates"][0].update(add_asset_ids=["system_voice"]),
                     lambda r: r["audio_policy"]["system_broadcast"].update(voice_id="hero"),
                     lambda r: r["additions"][0].update(audition_text="不该出现在角色中")]
        for mutate in mutations:
            response = revision()
            mutate(response)
            with self.subTest(mutate=mutate), self.assertRaises(BrainError):
                self.call(response)

    def test_asset_flag_reference_and_missing_current_update_are_validated(self):
        mutations = [lambda r: r.update(needs_assets=False),
                     lambda r: r.update(shot_updates=[]),
                     lambda r: r["shot_updates"][0].update(add_asset_ids=["announcer", "missing"]),
                     lambda r: r["shot_updates"][0].update(add_asset_ids=["announcer", "announcer", "system_voice"]),
                     lambda r: r["additions"].append(asset("unused", "prop", "无关物件", "无关物件"))]
        for mutate in mutations:
            response = revision()
            mutate(response)
            with self.subTest(mutate=mutate), self.assertRaises(BrainError):
                self.call(response)

    def test_boolean_schema_does_not_accept_truthy_integer_or_string(self):
        for field in ("needs_assets",):
            for value in (1, "true", None):
                response = revision()
                response[field] = value
                with self.subTest(value=value), self.assertRaises(BrainError):
                    self.call(response)

    def test_nested_plan_context_and_existing_voice_reuse_are_supported(self):
        response, data = revision(), context()
        data["assets"].extend(response["additions"])
        response.update(needs_assets=False, additions=[])
        data = {"plan": {"assets": data["assets"], "episodes": data["episodes"]}, "novel": data["novel"]}
        result, _ = self.call(response, data=data)
        self.assertFalse(result["needs_assets"])
        self.assertEqual(result["audio_policy"]["system_broadcast"]["voice_id"], "system_voice")

    def test_refine_retains_global_voice_rules_and_does_not_cancel_system_broadcast(self):
        response = {"prompt": "本段画外播报。", "dialogue": SYSTEM_TEXT, "continuity_notes": "保留时间表。"}
        with patch.object(Brain, "_call", return_value=response) as call:
            Brain().refine_shot(shot(), {"audio_policy": revision()["audio_policy"]}, [Path("synthetic.png")])
        instruction = call.call_args.args[0]
        for rule in ("context.audio_policy.system_broadcast.enabled=true", "不能借保持原时间表再次取消播报",
                     "不生成额外画面字幕不等于禁止读出", "不省略原角色对白", "没有形象参考"):
            self.assertIn(rule, instruction)

    def test_audition_selection_receives_original_speaker_identity(self):
        response = {"audition_text": SYSTEM_TEXT, "audition_source": SYSTEM_SOURCE, "audition_reason": "依用户要求播读原文系统文字。"}
        with patch.object(Brain, "_call", return_value=response) as call:
            Brain().select_voice_audition(context()["novel"], revision()["additions"][1], revision()["additions"][0])
        instruction, data, _ = call.call_args.args
        self.assertEqual(data["voice"]["audition_speaker"], "系统")
        self.assertTrue(data["character"]["voice_only"])
        self.assertIn("无需原文含它的展示名", instruction)


if __name__ == "__main__":
    unittest.main()
