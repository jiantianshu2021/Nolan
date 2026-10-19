"""Synthetic internal voice-source rule and metadata contracts; no model calls."""
import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from norlan import core, interactive, mcp_server
from norlan.brain import (ASSET_SCHEMA, PLAN_SCHEMA, Brain, BrainError,
                          INTERNAL_VOICE_SOURCE_RULES, _validate, validate_plan)


def asset(identifier, kind, **extra):
    return {"id": identifier, "kind": kind, "name": identifier, "prompt": "合成资产定义",
            "character_id": "", "voice_only": False, "audition_speaker": "", "audition_text": "",
            "audition_source": "", "audition_reason": "", **extra}


def initial_plan():
    return {"title": "合成画外声测试", "style": "漫画", "ratio": "16:9",
            "assets": [asset("system_role", "character", voice_only=True),
                       asset("system_voice", "voice", character_id="system_role", audition_speaker="系统",
                             audition_reason="合成原文暂无足够长的完整试音句，等待人工选择。"),
                       asset("room", "location")],
            "episodes": [{"id": "ep01", "title": "第一集", "summary": "系统播报", "music_prompt": "轻柔纯音乐",
                          "shots": [{"id": "s01", "duration": 10, "prompt": "系统框弹出，画外女声播报，不出现播报员。",
                                     "dialogue": "系统：合成通知。", "asset_ids": ["room", "system_role", "system_voice"],
                                     "scene_id": "room", "continuity_notes": "画外声源固定。",
                                     "subshots": [{"start": 0, "end": 10, "prompt": "固定机位，画外播报。"}]}]}]}


class InternalVoiceRuleTests(unittest.TestCase):
    def test_first_plan_strict_schema_can_bind_an_internal_source_without_portrait(self):
        planned = initial_plan()
        self.assertEqual(set(ASSET_SCHEMA["required"]), set(ASSET_SCHEMA["properties"]))
        self.assertFalse(ASSET_SCHEMA["additionalProperties"])
        self.assertIn("voice_only", ASSET_SCHEMA["properties"])
        self.assertIn("audition_speaker", ASSET_SCHEMA["properties"])
        _validate(planned, PLAN_SCHEMA)
        with patch.object(Brain, "_codex", return_value=planned) as model:
            result = Brain().plan("合成小说，系统字幕：合成通知。")
        self.assertTrue(result["assets"][0]["voice_only"])
        self.assertEqual(result["assets"][1]["audition_speaker"], "系统")
        self.assertIn(INTERNAL_VOICE_SOURCE_RULES, model.call_args.args[0])
        self.assertIn("实际 voice 仍须人工选择、试听和审批", model.call_args.args[0])

    def test_stored_legacy_plan_needs_no_flag_upgrade_or_mutation(self):
        planned = initial_plan()
        for definition in planned["assets"]:
            definition.pop("voice_only")
            definition.pop("audition_speaker")
        before = copy.deepcopy(planned)
        self.assertIs(validate_plan(planned), planned)
        self.assertEqual(planned, before)

    def test_flags_cannot_turn_voice_or_location_into_an_internal_character(self):
        for kind, index in (("voice", 1), ("location", 2)):
            planned = initial_plan()
            planned["assets"][index]["voice_only"] = True
            with self.subTest(kind=kind), self.assertRaisesRegex(BrainError, "仅适用于"):
                validate_plan(planned)
        planned = initial_plan()
        planned["assets"][0]["audition_speaker"] = "系统"
        with self.assertRaisesRegex(BrainError, "非声音资产"):
            validate_plan(planned)

    def test_native_and_mcp_explain_actual_voice_approval_and_internal_source_binding(self):
        self.assertIn(INTERNAL_VOICE_SOURCE_RULES, interactive._DIRECTOR_INSTRUCTIONS)
        for name in ("save_plan", "propose_asset_amendment"):
            with self.subTest(name=name):
                instruction = mcp_server.TOOL_MAP[name]["description"]
                self.assertIn(INTERNAL_VOICE_SOURCE_RULES, instruction)
                for rule in ("不作为单独资产卡", "不要让用户先批准画外角色定义", "试听并批准声音",
                             "声音未获人工批准就不能用于视频", "方案仍须用户确认", "不能借内部声源规则自动批准"):
                    self.assertIn(rule, instruction)
        self.assertIn("不作为单独资产或人工审批对象", mcp_server.ASSET["properties"]["voice_only"]["description"])

    def test_revision_instructs_model_to_approve_real_voice_without_extra_role_step(self):
        planned = initial_plan()
        response = {"needs_assets": False, "additions": [], "shot_updates": [], "summary": "复用既有声音。", "blockers": [],
                    "audio_policy": {"system_broadcast": {"enabled": False, "character_id": "", "voice_id": "", "instruction": ""}}}
        with patch.object(Brain, "_call", return_value=response) as call:
            Brain().plan_revision(planned["episodes"][0]["shots"][0], planned, "保留声音，修正构图。")
        instruction = call.call_args.args[0]
        self.assertIn(INTERNAL_VOICE_SOURCE_RULES, instruction)
        self.assertIn("不要求另批内部画外角色定义", instruction)
        self.assertNotIn("所有候选均待人工审批", instruction)

    def test_mcp_voice_metadata_stays_bound_by_id_after_hidden_roles_are_filtered(self):
        with tempfile.TemporaryDirectory() as directory:
            server = mcp_server.MCPServer(Path(directory) / "workspace")
            self.addCleanup(server.close)
            root = server.projects / "synthetic"
            core.create_project(root, "合成小说。", "demo")
            core.install_plan(root, initial_plan())
            public = {"assets": [{"id": "system_voice", "kind": "voice", "voice_source": {"name": "system_role"}}],
                      "episodes": [], "budget": {}}
            with patch.object(mcp_server.dashboard, "public_state", return_value=public):
                result = server._status(root)
            voice = result["assets"][0]
            self.assertEqual(voice["id"], "system_voice")
            self.assertEqual(voice["character_id"], "system_role")
            self.assertEqual(voice["audition_speaker"], "系统")
            self.assertFalse(voice["voice_only"])
            self.assertEqual(voice["voice_source"]["name"], "system_role")


if __name__ == "__main__":
    unittest.main()
