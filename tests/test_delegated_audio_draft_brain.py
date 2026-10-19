"""Preparing a timing draft requires no extra approval before the prompt gate."""
import copy
import unittest
from unittest.mock import patch

from norlan.brain import Brain, BrainError, CROSS_SHOT_AUDIO_RULES, REFINE_SCHEMA, _validate


SYSTEM_LINE = "检测到绑定者面临经营危机，临时任务已开启。请使用超维加工台完成首件样品，并在期限内获得客户认可。"


def source():
    return {"id": "s08", "duration": 15, "scene_id": "workshop", "asset_ids": ["hero", "voice", "system_role", "system_voice"],
            "status": "planned", "version": 0, "revision": 2,
            "prompt": "0–5秒主角询问，5–15秒系统界面出现并播报。",
            "dialogue": "楚阳：要我喂你？\n系统播报员：" + SYSTEM_LINE,
            "feedback": "无意见保持原来的就行。交给GPT来准备提示词。",
            "subshots": [{"start": 0, "end": 5, "prompt": "中景，楚阳放下手里的工具，问：要我喂你？"},
                         {"start": 5, "end": 15, "prompt": "近景，5秒系统界面首次出现，系统女声读原通知。"}]}


def project(shot):
    return {"assets": [{"id": "hero", "kind": "character"}, {"id": "voice", "kind": "voice", "character_id": "hero"},
                       {"id": "system_role", "kind": "character", "voice_only": True},
                       {"id": "system_voice", "kind": "voice", "character_id": "system_role"}],
            "episodes": [{"id": "ep01", "shots": [shot]}], "audio_policy": {}}


def plain_revision(blockers=None):
    return {"needs_assets": False, "additions": [], "shot_updates": [],
            "audio_policy": {"system_broadcast": {"enabled": False, "character_id": "", "voice_id": "", "instruction": ""}},
            "summary": "原总时长和摄影切点不变，先准备合理跨镜声音草稿，等待最终提示词确认。", "blockers": blockers or []}


def draft():
    shot = source()
    return {"prompt": "保持15秒与第5秒摄影切点。楚阳0–1.3秒问原句，系统界面1.5秒出现，女声跨第5秒切镜连续读至14.8秒。",
            "dialogue": shot["dialogue"], "continuity_notes": "声音/UI安排仅为待审草稿，尚未执行。",
            "timing_proposal": {
                "summary": "保留摄影切点与动作，界面和播报1.5秒开始，连续至14.8秒；最终提示词待人工确认。",
                "subshots": [{"start": 0, "end": 5, "prompt": "中景，楚阳放下工具并问原句；系统界面1.5秒出现且女声开始，声音跨切镜继续。"},
                             {"start": 5, "end": 15, "prompt": "近景承接已出现的系统界面，女声不重启、连续读至14.8秒，原动作不变。"}],
                "audio_windows": [{"start": 0, "end": 1.3, "speaker": "楚阳", "text": "要我喂你？"},
                                  {"start": 1.5, "end": 14.8, "speaker": "系统播报员", "text": SYSTEM_LINE}]}}


class DelegatedAudioDraftBrainTests(unittest.TestCase):
    def test_delegation_and_no_opinion_do_not_require_human_to_enter_seconds_before_drafting(self):
        for feedback in ("交给GPT来准备提示词", "由GPT决定", "无意见保持原来的就行", "继续准备候选提示词"):
            shot = {**source(), "feedback": feedback}
            before = copy.deepcopy(shot)
            with self.subTest(feedback=feedback), patch.object(Brain, "_call", return_value=plain_revision()) as call:
                response = Brain().plan_revision(shot, project(shot), feedback)
            instruction = call.call_args.args[0]
            self.assertIn(CROSS_SHOT_AUDIO_RULES, instruction)
            self.assertIn("不要求用户先亲自指定秒数", instruction)
            self.assertIn("即使没有明确委托，也可先提出这样的未批准草稿", instruction)
            self.assertIn("最终候选确认是执行许可", instruction)
            self.assertIn("不能因尚未选择具体秒点而阻塞准备候选", instruction)
            self.assertNotIn("用户明确选择后，才可在待确认", instruction)
            self.assertNotIn("必须采用用户已明确选择的安排", instruction)
            self.assertEqual(response["blockers"], [])
            self.assertEqual(shot, before)

    def test_valid_cross_cut_draft_is_returned_unapproved_without_mutating_canonical_timing(self):
        shot, response = source(), draft()
        before = copy.deepcopy(shot)
        _validate(response, REFINE_SCHEMA)
        with patch.object(Brain, "_call", return_value=response) as call:
            refined = Brain().refine_shot(shot, {}, [])
        self.assertEqual(refined, response)
        self.assertEqual(shot, before)
        instruction = call.call_args.args[0]
        for requirement in ("尚未批准", "timing_proposal", "audio_windows列出本段全部原对白", "摄影切点", "dialogue保持原文"):
            self.assertIn(requirement, instruction)
        self.assertIs(call.call_args.args[2], REFINE_SCHEMA)

    def test_invalid_cut_or_changed_dialogue_proposal_is_rejected_locally(self):
        wrong_cut = draft()
        wrong_cut["timing_proposal"]["subshots"][0]["end"] = 4.5
        wrong_cut["timing_proposal"]["subshots"][1]["start"] = 4.5
        wrong_dialogue = draft()
        wrong_dialogue["dialogue"] = "楚阳：改写了台词。"
        wrong_audio = draft()
        wrong_audio["timing_proposal"]["audio_windows"][1]["text"] += "额外编造。"
        overflow = draft()
        overflow["timing_proposal"]["audio_windows"][1]["end"] = 16
        for response in (wrong_cut, wrong_dialogue, wrong_audio, overflow):
            with self.subTest(response=response), patch.object(Brain, "_call", return_value=response):
                with self.assertRaises(BrainError):
                    Brain().refine_shot(source(), {}, [])

    def test_legacy_refine_response_and_explicit_null_remain_compatible(self):
        legacy = {key: value for key, value in draft().items() if key != "timing_proposal"}
        for response in (legacy, {**legacy, "timing_proposal": None}):
            with self.subTest(response=response), patch.object(Brain, "_call", return_value=response):
                self.assertEqual(Brain().refine_shot(source(), {}, []), response)
        self.assertEqual(set(REFINE_SCHEMA["required"]), set(REFINE_SCHEMA["properties"]))

    def test_real_full_segment_capacity_blocker_is_preserved(self):
        blocker = "全部原对白正常语速超过15秒，需改变总时长或对白；尚未提交视频。"
        shot = source()
        with patch.object(Brain, "_call", return_value=plain_revision([blocker])) as call:
            response = Brain().plan_revision(shot, project(shot), shot["feedback"])
        self.assertEqual(response["blockers"], [blocker])
        self.assertIn("仅当整个片段在正常语速和合理停顿下确实放不下", call.call_args.args[0])
        self.assertIn("不能加速、截断、重写或删除台词", call.call_args.args[0])


if __name__ == "__main__":
    unittest.main()
