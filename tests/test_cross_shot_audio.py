"""Synthetic timing choices: spoken audio may cross fixed camera cuts."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from norlan import core, pipeline
from norlan.brain import Brain, CROSS_SHOT_AUDIO_RULES


LINE = "赠送超维加工台、悬浮滑板图纸。仅绑定者可操作。"
CHOICE = "我选择系统框在第3秒出现并开始女声播报，声音跨第5秒切镜连续读完，保留10秒总长及原切点。"


def shot():
    return {"id": "s07", "duration": 10, "prompt": "0–5秒柜体落地，5–10秒系统框出现。",
            "dialogue": "系统播报员：" + LINE, "continuity_notes": "顺序保持。", "scene_id": "room",
            "asset_ids": ["room", "system_role", "system_voice"], "version": 0, "revision": 0,
            "subshots": [{"start": 0, "end": 5, "prompt": "中景，柜体落地。"},
                         {"start": 5, "end": 10, "prompt": "近景，系统通知。"}]}


def result(*, blocked=False):
    return {"needs_assets": False, "additions": [], "shot_updates": [],
            "audio_policy": {"system_broadcast": {"enabled": False, "character_id": "", "voice_id": "", "instruction": ""}},
            "summary": "需要用户选择跨镜播报安排。" if blocked else "按用户明确选择在原时长和切点内连续播报。",
            "blockers": ["原5秒声音窗口不足；可选择提前界面和播报时点，尚未取得同意。"] if blocked else []}


class CrossShotAudioRuleTests(unittest.TestCase):
    def context(self):
        return {"assets": [{"id": "room", "kind": "location"},
                           {"id": "system_role", "kind": "character", "voice_only": True},
                           {"id": "system_voice", "kind": "voice", "character_id": "system_role"}],
                "episodes": [{"id": "ep01", "shots": [shot()]}]}

    def test_unanswered_timing_proposal_remains_a_choice_without_changing_the_plan(self):
        source, context = shot(), self.context()
        before = copy.deepcopy((source, context))
        feedback = "增强科技感。有人建议第3秒开始播报，但用户尚未答复。"
        with patch.object(Brain, "_call", return_value=result(blocked=True)) as call:
            planned = Brain().plan_revision(source, context, feedback)
        instruction = call.call_args.args[0]
        self.assertIn(CROSS_SHOT_AUDIO_RULES, instruction)
        self.assertIn("不得把建议、未答复的问题、继续或恢复当作同意", instruction)
        self.assertIn("不能把“增强科技感”推断为同意提前播音", instruction)
        self.assertTrue(planned["blockers"])
        self.assertEqual((source, context), before)

    def test_explicit_audio_choice_can_cross_the_five_second_cut_without_new_assets_or_longer_duration(self):
        source, context = shot(), self.context()
        before = copy.deepcopy((source, context))
        response = result()
        response["shot_updates"] = [{"id": source["id"], "add_asset_ids": [],
            "prompt": "用户已选择：系统框第3秒出现，女声从3秒开始，跨5秒切镜连续读完原通知，约9.08秒结束。",
            "dialogue": source["dialogue"], "continuity_notes": "10秒总长，切点仍5秒；原动作和台词不删改。"}]
        with patch.object(Brain, "_call", return_value=response) as call:
            planned = Brain().plan_revision(source, context, CHOICE)
        self.assertEqual(planned["blockers"], [])
        self.assertFalse(planned["needs_assets"])
        self.assertEqual(planned["shot_updates"][0]["dialogue"], source["dialogue"])
        self.assertEqual((source, context), before)
        self.assertIn("不能因一句话跨摄影切点就认定超时", call.call_args.args[0])
        self.assertIn("用户点击确认提示词后才提交一次视频", call.call_args.args[0])

    def test_refinement_receives_the_same_choice_rule_and_cannot_silently_change_camera_timing(self):
        source = {**shot(), "feedback": CHOICE}
        before = copy.deepcopy(source)
        response = {"prompt": "按明确选择跨镜连续播报。", "dialogue": source["dialogue"], "continuity_notes": "原切点保持。"}
        with patch.object(Brain, "_call", return_value=response) as call:
            Brain().refine_shot(source, {"audio_references": [{"trim_duration": 6.08, "derived": False}]}, [])
        instruction = call.call_args.args[0]
        self.assertIn(CROSS_SHOT_AUDIO_RULES, instruction)
        self.assertIn("保持 subshots 的数量、顺序、start/end 和 duration 完全不变", instruction)
        self.assertIn("只生成本段计划时长的新内容", instruction)
        self.assertEqual(source, before)


class RevisionAnalysisHistoryTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        core.create_project(self.root, "合成规划阻塞测试。", "live")
        core.install_plan(self.root, pipeline.demo_plan())
        core.approve(self.root, "plan", "plan", 1)
        self.id = core.load_project(self.root)["episodes"][0]["shots"][0]["id"]

    def current(self):
        return core._find(core.load_project(self.root), "shot", self.id)

    @staticmethod
    def signature(revision, feedback):
        return hashlib.sha256(json.dumps([revision, feedback], ensure_ascii=False).encode()).hexdigest()

    def test_same_blocker_is_cached_new_human_feedback_replans_once_and_preserves_the_old_report(self):
        old_feedback = "合成历史资产修改意见。"
        old = {"signature": self.signature(0, old_feedback), "blockers": ["全部片段可用播报时窗不足。"],
               "summary": "真实容量冲突，需调整规划。", "planning_policy_version": 2}
        with core.transaction(self.root) as state:
            current = core._find(state, "shot", self.id)
            current.update(feedback=old_feedback, revision_analysis=copy.deepcopy(old))
        with patch.object(pipeline, "_brain") as brain, patch.object(pipeline, "_ark") as media:
            for _ in range(2):
                with self.assertRaisesRegex(core.GateError, "时窗不足"):
                    pipeline._revision_analysis(self.root, self.current())
            brain.assert_not_called()
            media.assert_not_called()
        with core.transaction(self.root) as state:
            current = core._find(state, "shot", self.id)
            current.update(feedback=old_feedback + "\n用户对规划阻塞的补充意见：" + CHOICE, revision=1)
        with patch.object(pipeline, "_brain", return_value=result()) as brain, patch.object(pipeline, "_ark") as media:
            self.assertTrue(pipeline._revision_analysis(self.root, self.current()))
            self.assertTrue(pipeline._revision_analysis(self.root, self.current()))
            brain.assert_called_once()
            media.assert_not_called()
        current = self.current()
        self.assertEqual(current["revision_analysis"]["blockers"], [])
        self.assertEqual(current["revision_analysis"]["signature"], self.signature(1, current["feedback"]))
        history = current["revision_analysis_history"]
        self.assertEqual(len(history), 1)
        self.assertEqual(history[0]["signature"], old["signature"])
        self.assertEqual(history[0]["blockers"], old["blockers"])
        self.assertFalse(core.load_project(self.root)["jobs"])

    def test_existing_history_from_the_human_feedback_gate_is_not_duplicated(self):
        old = {"signature": "old-feedback-signature", "blockers": ["合成旧阻塞。"], "summary": "合成说明。"}
        with core.transaction(self.root) as state:
            current = core._find(state, "shot", self.id)
            current.update(feedback=CHOICE, revision=1, revision_analysis=copy.deepcopy(old),
                           revision_analysis_history=[copy.deepcopy(old)])
        with patch.object(pipeline, "_brain", return_value=result()):
            self.assertTrue(pipeline._revision_analysis(self.root, self.current()))
        self.assertEqual(len(self.current()["revision_analysis_history"]), 1)


if __name__ == "__main__":
    unittest.main()
