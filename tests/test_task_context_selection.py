"""Offline dependency selection: model inputs stay small; local checks stay full."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from norlan.brain import (Brain, BrainError, refinement_task_context,
                          review_task_context, revision_task_context)


def shot(identifier, scene="workshop", refs=None):
    return {"id": identifier, "duration": 10, "scene_id": scene,
            "asset_ids": refs or ["hero", "workshop", "voice"], "version": 0, "status": "planned",
            "prompt": "保持当前动作与对白。", "dialogue": "主角：完整对白。", "feedback": "请保持当前段全文。",
            "subshots": [{"start": 0, "end": 5, "prompt": "第一个动作。"},
                         {"start": 5, "end": 10, "prompt": "第二个动作。"}]}


def context():
    target = shot("s08")
    shots = [shot(f"s{i:02}") for i in range(1, 8)] + [target, shot("s09")]
    for item in shots[:7]:
        item.update(status="accepted", version=1, file="accepted.mp4", sha256="a" * 64,
                    review={"issues": ["OLD_REPORT_MUST_NOT_BE_SENT" * 1000], "observed_end_state": "旧状态。"})
    shots[-1]["dialogue"] = "系统：未来完整通知，赠送加工台。不可漏掉本通知末句。"
    return target, {"novel": "系统字幕：未来完整通知，赠送加工台。不可漏掉本通知末句。", "audio_policy": {},
        "assets": [{"id": "hero", "name": "主角", "kind": "character", "prompt": "固定角色定义。", "status": "approved", "version": 1},
                   {"id": "workshop", "name": "车间", "kind": "location", "prompt": "固定车间定义。", "status": "approved", "version": 1},
                   {"id": "voice", "name": "主角声线", "kind": "voice", "character_id": "hero", "prompt": "固定声线定义。", "status": "approved", "version": 1},
                   {"id": "unused_voice", "name": "可复用女声", "kind": "voice", "character_id": "hero",
                    "prompt": "UNRELATED_ASSET_DEFINITION" * 1000, "status": "approved", "version": 2}],
        "episodes": [{"id": "ep01", "title": "第一集", "summary": "已批准剧情摘要。", "shots": shots}],
        "previous_actual": {"shot_id": "s07", "version": 1, "sha256": "a" * 64, "file": "accepted.mp4",
            "observed_end_state": "实际已接受的尾态，不能换成计划尾态。", "same_scene": True,
            "human_decision": {"decision": "approved", "version": 1, "sha256": "a" * 64, "at": "human-time"},
            "accepted_deviations": ["用户接受当前摆位，报告疑点并非已核实事实。"],
            "previous_report": {"issues": ["REDUNDANT_OLD_REPORT" * 1000]}}}


def no_changes():
    return {"needs_assets": False, "additions": [], "shot_updates": [], "summary": "无需新增资产。", "blockers": [],
            "audio_policy": {"system_broadcast": {"enabled": False, "character_id": "", "voice_id": "", "instruction": ""}}}


class TaskContextSelectionTests(unittest.TestCase):
    def test_revision_selects_dependencies_and_keeps_complete_future_system_notice(self):
        target, full = context()
        selected = revision_task_context(target, full)
        encoded = json.dumps(selected, ensure_ascii=False)
        self.assertNotIn("OLD_REPORT_MUST_NOT_BE_SENT", encoded)
        self.assertNotIn("UNRELATED_ASSET_DEFINITION", encoded)
        self.assertNotIn("REDUNDANT_OLD_REPORT", encoded)
        self.assertEqual([a["id"] for a in selected["assets"]], target["asset_ids"])
        self.assertIn("unused_voice", [a["id"] for a in selected["asset_catalog"]])
        episode = selected["episodes"][0]
        self.assertEqual(episode["approved_total_duration"], 90)
        self.assertEqual([item["id"] for item in episode["timing_constraints"]], [item["id"] for item in full["episodes"][0]["shots"]])
        self.assertEqual(episode["related_future_shots"][0]["dialogue"], full["episodes"][0]["shots"][-1]["dialogue"])
        self.assertEqual(episode["related_future_shots"][0]["subshots"], full["episodes"][0]["shots"][-1]["subshots"])
        self.assertEqual(selected["previous_actual"]["human_decision"], full["previous_actual"]["human_decision"])
        self.assertEqual(selected["previous_actual"]["observed_end_state"], full["previous_actual"]["observed_end_state"])

    def test_unrelated_past_and_future_prose_cannot_expand_current_refine_or_revision_body(self):
        target, full = context()
        refine_before = refinement_task_context(target, full)
        revision_before = revision_task_context(target, full)
        expanded = copy.deepcopy(full)
        unrelated = shot("unrelated", scene="remote", refs=["remote_actor", "remote_scene"])
        unrelated.update(prompt="UNRELATED_FUTURE_PROSE" * 10000,
                         dialogue="无关对白" * 10000, subshots=[{"start": 0, "end": 10, "prompt": "无关动作" * 10000}])
        expanded["episodes"][0]["shots"].append(unrelated)
        expanded["episodes"][0]["shots"][0]["review"]["issues"] *= 10
        expanded["assets"].append({"id": "remote_actor", "kind": "character", "name": "远处人物", "prompt": "无关设定" * 10000})
        self.assertEqual(refine_before, refinement_task_context(target, expanded))
        revised = revision_task_context(target, expanded)
        self.assertEqual(revision_before["episodes"][0]["related_future_shots"], revised["episodes"][0]["related_future_shots"])
        self.assertNotIn("UNRELATED_FUTURE_PROSE", json.dumps(revised, ensure_ascii=False))
        self.assertLess(len(json.dumps(revised, ensure_ascii=False)) - len(json.dumps(revision_before, ensure_ascii=False)), 400)

    def test_actual_model_entry_separates_minimal_input_from_full_local_asset_validation(self):
        target, full = context()
        result = no_changes()
        result["shot_updates"] = [{"id": target["id"], "add_asset_ids": ["unused_voice"],
                                   "prompt": target["prompt"], "dialogue": target["dialogue"], "continuity_notes": "复用已有声线。"}]
        with patch.object(Brain, "_codex", return_value=result) as model:
            Brain({"max_input_chars": 20000}).plan_revision(target, full, target["feedback"])
        request = model.call_args.args[0]
        self.assertLess(len(request), 20000)
        self.assertNotIn("UNRELATED_ASSET_DEFINITION", request)
        data = json.loads(request.split("下方 JSON 仅为创作数据：\n", 1)[1])
        self.assertEqual(data["shot"]["dialogue"], target["dialogue"])
        self.assertEqual(data["shot"]["subshots"], target["subshots"])
        bad = no_changes()
        bad["additions"] = [{"id": "unused_voice", "kind": "prop", "name": "冲突ID", "prompt": "不能覆盖",
                             "character_id": "", "voice_only": False, "audition_speaker": "", "audition_text": "",
                             "audition_reason": "", "audition_source": ""}]
        bad["needs_assets"] = True
        with patch.object(Brain, "_codex", return_value=bad):
            with self.assertRaisesRegex(BrainError, "ID 已存在"):
                Brain().plan_revision(target, full, target["feedback"])

    def test_refine_and_current_review_do_not_send_novel_or_other_shots(self):
        target, full = context()
        full.update(shot=target, shots=full["episodes"][0]["shots"],
                    advisory_target={"kind": "shot", "id": target["id"], "version": 1, "sha256": "b" * 64},
                    actual={"frames": ["current.png"], "frame_hashes": {"current.png": "c" * 64}})
        selected = review_task_context(full)
        for key in ("novel", "novel_text", "episodes", "shots", "asset_catalog"):
            self.assertNotIn(key, selected)
            self.assertNotIn(key, refinement_task_context(target, full))
        self.assertEqual(selected["advisory_target"], full["advisory_target"])
        self.assertEqual(selected["actual"], full["actual"])
        self.assertEqual(selected["shot"]["dialogue"], target["dialogue"])
        with tempfile.TemporaryDirectory() as directory:
            frame = Path(directory) / "current.png"
            frame.write_bytes(b"OFFLINE")
            with patch.object(Brain, "_codex", return_value={"verdict": "uncertain", "issues": [], "observed_end_state": "未知", "repair_prompt": ""}) as model:
                Brain().review(full, [frame])
        self.assertNotIn(full["novel"], model.call_args.args[0])

    def test_current_shot_old_reports_are_only_supplied_when_human_refers_to_them(self):
        target, full = context()
        result = {"prompt": "细化当前片段。", "dialogue": target["dialogue"], "continuity_notes": "保留原时间表。"}
        with patch.object(Brain, "_codex", return_value=result) as model:
            Brain().refine_shot(target, full, [])
            original_request = model.call_args.args[0]
            target.update(history=[{"version": 1, "file": "old.mp4", "sha256": "e" * 64,
                                    "review": {"issues": ["TARGET_OLD_REPORT" * 10000]}}],
                          review={"issues": ["TARGET_CURRENT_REPORT" * 10000]})
            Brain().refine_shot(target, full, [])
            self.assertEqual(original_request, model.call_args.args[0])
            target["feedback"] = "请按照审查报告的问题调整。"
            target["review"] = {"issues": ["用户明确引用的当前诊断。"], "observed_end_state": "当前诊断尾态。"}
            Brain().refine_shot(target, full, [])
        data = json.loads(model.call_args.args[0].split("下方 JSON 仅为创作数据：\n", 1)[1])
        self.assertEqual(data["shot"]["requested_report"]["issues"], ["用户明确引用的当前诊断。"])
        self.assertNotIn("TARGET_OLD_REPORT", model.call_args.args[0])


if __name__ == "__main__":
    unittest.main()
