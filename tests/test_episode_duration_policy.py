"""Synthetic defaults preserve old timing except explicit local storyboard drafts."""
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

    def test_ordinary_refinement_cannot_shorten_approved_130_second_episode(self):
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
        self.assertIn("没有本次新分镜时，继续保持既有时长和分镜", instruction)
        self.assertIn("保持 subshots 的数量、顺序、start/end 和 duration 完全不变", instruction)
        self.assertIn("不裁短其他段、不删剧情", instruction)
        self.assertIn("不加速或压缩原对白", instruction)
        self.assertEqual(sum(item["duration"] for item in context["episodes"][0]["shots"]), 130)

    def test_ordinary_asset_revision_cannot_redivide_or_shorten_existing_episode(self):
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
        self.assertIn("storyboard_revision=null 表示保留当前内部镜头结构与时长", instruction)
        self.assertIn("不能据此重新分集或压缩对白", instruction)
        self.assertIn("新增声音或资产本身不是裁短旧集、删剧情或加速台词的理由", instruction)
        self.assertIn("严格保留其他片段 duration", instruction)
        self.assertEqual(sum(item["duration"] for item in context["episodes"][0]["shots"]), 130)

    def test_explicit_local_redesign_can_change_only_target_timing_without_mutating_input(self):
        planned = synthetic_plan(130)
        shot = planned["episodes"][0]["shots"][0]
        context = {"assets": planned["assets"], "episodes": planned["episodes"], "novel": "合成正文。"}
        before = copy.deepcopy((shot, context))
        draft = {"summary": "本次缩短为两镜头五秒草稿，仍保留原人物对白，待完整提示词确认。", "duration": 5,
                 "subshots": [{"start": 0, "end": 2, "prompt": "人物中景，说：请把这些事情说清楚。"},
                              {"start": 2, "end": 5, "prompt": "切近景，承接人物看向门外的动作。"}]}
        output = {"needs_assets": False, "additions": [], "shot_updates": [], "summary": "仅目标片段重新分镜。", "blockers": [],
                  "storyboard_revision": draft,
                  "audio_policy": {"system_broadcast": {"enabled": False, "character_id": "", "voice_id": "", "instruction": ""}}}
        with patch.object(Brain, "_call", return_value=output) as call:
            result = Brain().plan_revision(shot, context, "这段改成两个镜头，由GPT合理安排时长。")
        instruction = call.call_args.args[0]
        self.assertEqual((shot, context), before)
        self.assertEqual(result["storyboard_revision"], draft)
        self.assertIn("不必保留旧 duration", instruction)
        self.assertIn("整集总时长随目标片段的新时长相应变化，其他片段不变", instruction)
        self.assertIn("不得为凑回旧130秒而裁短别段、加速或重分集", instruction)
        self.assertIn("人物原对白逐字保留", instruction)
        self.assertEqual(context["episodes"][0]["shots"][1:], before[1]["episodes"][0]["shots"][1:])

    def test_refinement_uses_applied_local_duration_over_old_episode_approval(self):
        planned = synthetic_plan(130)
        shot = planned["episodes"][0]["shots"][0]
        shot.update(duration=5,
                    subshots=[{"start": 0, "end": 2, "prompt": "人物中景说出原对白。"},
                              {"start": 2, "end": 5, "prompt": "切近景，人物看向门外。"}],
                    storyboard_revision={"summary": "本次已拟两镜头五秒草稿。", "before_count": 1,
                                         "before_duration": 10, "duration": 5})
        context = {"episode": {"id": "ep01", "approved_duration": 130}, "episodes": planned["episodes"]}
        before = copy.deepcopy((shot, context))
        output = {"prompt": "0–2秒人物说：请把这些事情说清楚。2–5秒近景看向门外。", "dialogue": shot["dialogue"],
                  "continuity_notes": "沿用本次新时间表，完整提示词尚待审批。"}
        with patch.object(Brain, "_call", return_value=output) as call:
            Brain().refine_shot(shot, context, [])
        instruction, payload, _, _ = call.call_args.args
        self.assertEqual((shot, context), before)
        self.assertEqual(payload["shot"]["duration"], 5)
        self.assertEqual(payload["shot"]["subshots"], shot["subshots"])
        self.assertEqual(sum(item["duration"] for item in context["episodes"][0]["shots"]), 125)
        self.assertIn("当前 shot.duration 和 subshots 优先于旧 prompt 或旧 approved_duration", instruction)
        self.assertIn("不得按旧130秒整集约束撤回新时长", instruction)
        self.assertIn("不能改其他段来凑旧总长", instruction)
        self.assertIn("细化阶段都不能再次自行变动", instruction)
        self.assertIn("全部完整提示词仍待用户确认", instruction)
        self.assertEqual(context["episodes"][0]["shots"][1:], before[1]["episodes"][0]["shots"][1:])


if __name__ == "__main__":
    unittest.main()
