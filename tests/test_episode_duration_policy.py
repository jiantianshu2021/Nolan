"""Synthetic new-plan timing defaults; existing approved episodes stay intact."""
import copy
import os
from pathlib import Path
import tempfile
import tomllib
import unittest
from unittest.mock import patch

from norlan.brain import Brain, NEW_EPISODE_DURATION_RULES, validate_plan
from norlan import interactive, mcp_server


def synthetic_plan(seconds=60):
    count = seconds // 10
    return {"title": "合成时长测试", "style": "漫画", "ratio": "16:9",
            "assets": [{"id": "hero", "kind": "character", "name": "测试角色", "prompt": "漫画角色",
                        "character_id": "", "audition_text": "", "audition_reason": "", "audition_source": ""}],
            "episodes": [{"id": "ep01", "title": "合成第一集", "summary": "保留剧情与对白。", "music_prompt": "无歌词配乐",
                          "shots": [{"id": "s" + str(index), "duration": 10, "prompt": "人物看向门外。",
                                     "dialogue": "请把这些事情说清楚。", "asset_ids": ["hero"], "scene_id": "room",
                                     "continuity_notes": "承接前段。", "subshots": [{"start": 0, "end": 10, "prompt": "保持动作和完整对白。"}]}
                                    for index in range(count)]}]}


class EpisodeDurationPolicyTests(unittest.TestCase):
    def test_native_launch_and_advertised_save_plan_receive_the_shared_duration_rule(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            source = root / "synthetic-codex"
            source.mkdir()
            with patch.dict(os.environ, {"XDG_DATA_HOME": str(root / "data"), "CODEX_HOME": str(source)}, clear=True), \
                    patch.object(interactive, "resolve_codex", return_value="/synthetic/codex"):
                _, env = interactive.prepare_launch(connection={"provider": "openai", "model": "synthetic-model"})
            config = tomllib.loads((Path(env["CODEX_HOME"]) / "config.toml").read_text(encoding="utf-8"))
            self.assertIn(NEW_EPISODE_DURATION_RULES, config["developer_instructions"])
            server = mcp_server.MCPServer(root / "workspace")
            self.addCleanup(server.close)
            server.handle({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {
                "protocolVersion": "2025-06-18", "capabilities": {}, "clientInfo": {"name": "synthetic", "version": "1"}}})
            advertised = server.handle({"jsonrpc": "2.0", "id": 2, "method": "tools/list"})["result"]["tools"]
            save_plan = next(tool for tool in advertised if tool["name"] == "save_plan")
            self.assertIn(NEW_EPISODE_DURATION_RULES, save_plan["description"])
            self.assertIn("save_plan", config["mcp_servers"]["norlan"]["enabled_tools"])

    def test_new_plan_targets_one_minute_by_sum_and_prefers_more_episodes(self):
        planned = synthetic_plan()
        with patch.object(Brain, "_call", return_value=planned) as call:
            result = Brain().plan("合成小说正文。")
        instruction, data, _ = call.call_args.args
        self.assertEqual(data, {"novel": "合成小说正文。"})
        self.assertEqual(sum(s["duration"] for s in result["episodes"][0]["shots"]), 60)
        self.assertIn(NEW_EPISODE_DURATION_RULES, instruction)
        for rule in ("以60秒为目标", "50–70秒", "duration 之和", "优先增加集数", "合理情节断点分集",
                     "不能压缩对白", "强行加速表演", "删减剧情", "截断一句台词"):
            self.assertIn(rule, instruction)
        self.assertNotIn("默认每集约 45–60 秒", instruction)

    def test_explicit_other_duration_is_respected_without_coercion(self):
        planned = synthetic_plan(90)
        before = copy.deepcopy(planned)
        request = "合成小说正文。\n\n用户创作要求：每集90秒，保留所有对白。"
        with patch.object(Brain, "_call", return_value=planned) as call:
            result = Brain().plan(request)
        self.assertEqual(result, before)
        self.assertEqual(sum(s["duration"] for s in result["episodes"][0]["shots"]), 90)
        self.assertIn("尊重该要求", call.call_args.args[0])
        self.assertIn("默认60秒不能覆盖用户指定时长", call.call_args.args[0])

    def test_existing_130_second_plan_remains_valid_and_unchanged(self):
        planned = synthetic_plan(130)
        before = copy.deepcopy(planned)
        self.assertIs(validate_plan(planned), planned)
        self.assertEqual(planned, before)
        self.assertEqual(sum(s["duration"] for s in planned["episodes"][0]["shots"]), 130)

    def test_refinement_cannot_shorten_approved_130_second_episode(self):
        planned = synthetic_plan(130)
        shot = planned["episodes"][0]["shots"][0]
        context = {"episode": {"id": "ep01", "approved_duration": 130}, "episodes": planned["episodes"]}
        before = copy.deepcopy((shot, context))
        output = {"prompt": shot["prompt"], "dialogue": shot["dialogue"], "continuity_notes": shot["continuity_notes"]}
        with patch.object(Brain, "_call", return_value=output) as call:
            Brain().refine_shot(shot, context, [])
        instruction = call.call_args.args[0]
        self.assertEqual((shot, context), before)
        self.assertIn("默认值不适用于本次细化", instruction)
        self.assertIn("130秒第一集继续保持130秒", instruction)
        self.assertIn("片段时长和切点保持原样", instruction)
        self.assertIn("不加速或压缩原对白", instruction)

    def test_asset_revision_cannot_redivide_or_shorten_existing_episode(self):
        planned = synthetic_plan(130)
        shot = planned["episodes"][0]["shots"][0]
        context = {"assets": planned["assets"], "episodes": planned["episodes"], "novel": "合成正文。"}
        before = copy.deepcopy((shot, context))
        output = {"needs_assets": False, "additions": [], "shot_updates": [], "summary": "仅修正动作，时长不变。", "blockers": [],
                  "audio_policy": {"system_broadcast": {"enabled": False, "character_id": "", "voice_id": "", "instruction": ""}}}
        with patch.object(Brain, "_call", return_value=output) as call:
            Brain().plan_revision(shot, context, "动作保持连贯。")
        instruction = call.call_args.args[0]
        self.assertEqual((shot, context), before)
        self.assertIn("默认值不适用于本次返修", instruction)
        self.assertIn("130秒第一集继续保持130秒", instruction)
        self.assertIn("不能据此重新分集、改变已批准的整集总时长或压缩对白", instruction)


if __name__ == "__main__":
    unittest.main()
