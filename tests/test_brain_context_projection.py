"""Offline task-context bounds: keep creative facts, exclude nested runtime logs."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from norlan import core, pipeline
from norlan.brain import (Brain, task_asset_summary, task_context_summary,
                          task_shot_summary)


RUNTIME = "SYNTHETIC_RUNTIME_PAYLOAD_DO_NOT_SEND"


def base_shot(identifier="s01"):
    return {"id": identifier, "status": "planned", "version": 0, "revision": 0, "duration": 10,
            "prompt": "固定机位，保持柜体位置和动作顺序。", "dialogue": "系统：完整保留本次通知。",
            "scene_id": "room", "continuity_notes": "保留原切点。", "asset_ids": ["hero", "room"],
            "feedback": "用户最新完整反馈：" + "不要加速或改写台词。" * 20 + "反馈末尾标记。",
            "subshots": [{"start": 0, "end": 5, "prompt": "柜体落地。"}, {"start": 5, "end": 10, "prompt": "通知出现。"}]}


def bloated_shot():
    shot = base_shot()
    history = {"version": 1, "file": "media/s01-v1.mp4", "sha256": "a" * 64,
               "review": {"verdict": "uncertain", "issues": ["[待核实] 旧版本诊断。"],
                          "observed_end_state": "旧版尾态。", "repair_prompt": "仅为建议。"}}
    for _ in range(6):
        history = {**history, "history": [copy.deepcopy(history)],
                   "reference_manifest": {"context": {"shot": copy.deepcopy(history)}}}
    shot.update(history=[history], reference_manifest={"context": {"state": RUNTIME * 20000}},
                prompt_candidate={"prompt": RUNTIME * 20000}, prompt_history=[{"prompt": RUNTIME * 20000}],
                revision_analysis_history=[{"summary": RUNTIME * 20000}],
                human_decisions=[{"feedback": RUNTIME * 20000}])
    return shot


def bloated_context():
    current = bloated_shot()
    assets = [{"id": "hero", "kind": "character", "name": "主角", "prompt": "漫画角色，服装固定。",
               "status": "approved", "version": 2, "file": "media/hero-v2.png", "sha256": "1" * 64,
               "approved_sha256": "1" * 64, "approved_at": "synthetic-approval",
               "feedback": "当前角色反馈完整保留。", "history": [{"raw": RUNTIME * 20000}]},
              {"id": "room", "kind": "location", "name": "车间", "prompt": "漫画车间，窗在左侧。",
               "status": "approved", "version": 1, "sha256": "2" * 64}]
    decision = {"decision": "approved", "version": 2, "sha256": "3" * 64, "at": "synthetic-human-time"}
    return {"novel": "合成小说全文，不可截断。" * 300 + "小说末尾标记。", "assets": assets,
            "episodes": [{"id": "ep01", "title": "合成第一集", "summary": "完整当前分集摘要。", "status": "generating",
                          "shots": [current, base_shot("s02")], "rough_cuts": [{"context": RUNTIME * 20000}],
                          "reviews": [{"context": RUNTIME * 20000}]}],
            "previous_actual": {"shot_id": "previous", "sha256": "3" * 64, "same_scene": True,
                "reference_mode": "official_tail", "observed_end_state": "人物站在柜体左侧。", "human_decision": decision,
                "accepted_deviations": ["[细节] 服装颜色轻微变化，用户接受当前版本。"],
                "previous_report": {"verdict": "fail", "issues": ["[细节] 已接受的疑点。"]}},
            "image_references": [{"index": 1, "role": "previous_official_tail", "shot_id": "previous", "sha256": "4" * 64}],
            "audio_references": [{"index": 1, "asset_id": "voice", "character_id": "hero", "sha256": "5" * 64,
                                  "source_sha256": "5" * 64, "trim_start": 0, "trim_duration": 6.08, "derived": False}],
            "video_references": [{"index": 1, "role": "previous_original_video", "shot_id": "previous", "sha256": "3" * 64,
                                  "file": "media/previous-v2.mp4", "task_id": "synthetic-task", "duration": 10}],
            "observation_images": [{"index": 2, "role": "previous_observation_tail", "shot_id": "previous", "sha256": "6" * 64,
                                    "sent_to_video_engine": False}], "audio_policy": {},
            "history": [{"raw": RUNTIME * 20000}]}


class BrainContextProjectionTests(unittest.TestCase):
    def test_projection_removes_recursive_execution_snapshots_without_clipping_creative_data(self):
        original = bloated_context()
        before = copy.deepcopy(original)
        compact = task_context_summary(original)
        serialized = json.dumps(compact, ensure_ascii=False)
        self.assertNotIn(RUNTIME, serialized)
        self.assertLess(len(serialized), 20000)
        self.assertEqual(compact["novel"], original["novel"])
        self.assertTrue(compact["novel"].endswith("小说末尾标记。"))
        current = compact["episodes"][0]["shots"][0]
        for key in ("prompt", "dialogue", "feedback", "asset_ids", "duration", "subshots", "scene_id", "continuity_notes"):
            self.assertEqual(current[key], original["episodes"][0]["shots"][0][key])
        self.assertTrue(current["feedback"].endswith("反馈末尾标记。"))
        self.assertEqual(compact["previous_actual"], original["previous_actual"])
        for key in ("image_references", "audio_references", "video_references", "observation_images"):
            self.assertEqual(compact[key], original[key])
        self.assertEqual(current["previous_candidate"]["file"], "media/s01-v1.mp4")
        self.assertEqual(current["previous_candidate"]["sha256"], "a" * 64)
        self.assertEqual(original, before)
        self.assertEqual(task_context_summary(compact), compact)

    def test_repeated_history_growth_does_not_grow_the_next_model_payload(self):
        original = bloated_context()
        sizes = []
        for _ in range(5):
            sizes.append(len(json.dumps(task_context_summary(original), ensure_ascii=False)))
            current = original["episodes"][0]["shots"][0]
            current["history"].append({"version": 1, "file": "media/s01-v1.mp4", "sha256": "a" * 64,
                "review": current["history"][-1]["review"], "reference_manifest": copy.deepcopy(current.get("reference_manifest")),
                "history": copy.deepcopy(current["history"])})
        self.assertEqual(len(set(sizes)), 1)

    def test_plan_revision_direct_entry_uses_stable_task_summary_before_size_validation(self):
        context = bloated_context()
        current = context["episodes"][0]["shots"][0]
        response = {"needs_assets": False, "additions": [], "shot_updates": [], "summary": "仅改原有安排，无需新资产。", "blockers": [],
                    "audio_policy": {"system_broadcast": {"enabled": False, "character_id": "", "voice_id": "", "instruction": ""}}}
        with patch.object(Brain, "_codex", return_value=response) as model:
            Brain({"max_input_chars": 30000}).plan_revision(current, context, current["feedback"])
        request = model.call_args.args[0]
        self.assertLess(len(request), 30000)
        self.assertNotIn(RUNTIME, request)
        data = json.loads(request.split("下方 JSON 仅为创作数据：\n", 1)[1])
        self.assertEqual(data["context"]["novel"], context["novel"])
        self.assertEqual(data["feedback"], current["feedback"])

    def test_refinement_and_sampled_review_keep_exact_approved_reference_evidence(self):
        context = bloated_context()
        current = context["episodes"][0]["shots"][0]
        refined = {"prompt": "合成最终提示词。", "dialogue": current["dialogue"], "continuity_notes": "保留实际前段。"}
        with patch.object(Brain, "_codex", return_value=refined) as model:
            Brain({"max_input_chars": 30000}).refine_shot(current, context, [])
        request = model.call_args.args[0]
        self.assertNotIn(RUNTIME, request)
        data = json.loads(request.split("下方 JSON 仅为创作数据：\n", 1)[1])
        self.assertEqual(data["context"]["previous_actual"]["human_decision"], context["previous_actual"]["human_decision"])
        self.assertEqual(data["shot"]["subshots"], current["subshots"])
        with tempfile.TemporaryDirectory() as directory:
            frame = Path(directory) / "synthetic.png"
            frame.write_bytes(b"SYNTHETIC FRAME")
            review_context = {**context, "shot": current, "shots": [current], "advisory_target": {
                "kind": "shot", "id": current["id"], "version": 1, "sha256": "9" * 64,
                "file": "media/s01-v1.mp4", "video_job_id": "synthetic-video"}}
            response = {"verdict": "uncertain", "issues": [], "observed_end_state": "未知", "repair_prompt": ""}
            with patch.object(Brain, "_codex", return_value=response) as model:
                Brain({"max_input_chars": 30000}).review(review_context, [frame])
        request = model.call_args.args[0]
        self.assertNotIn(RUNTIME, request)
        data = json.loads(request.split("下方 JSON 仅为创作数据：\n", 1)[1])
        self.assertEqual(data["advisory_target"], review_context["advisory_target"])
        self.assertEqual(data["video_references"], context["video_references"])

    def test_asset_summary_retains_current_human_preferences_without_full_history(self):
        asset = bloated_context()["assets"][0]
        asset["creation_input"] = {"mode": "custom", "description": "用户指定的完整衣领样式。", "confirmed": True, "revision": 1,
                                   "references": [{"id": "ref1", "file": "inputs/ref.png", "sha256": "7" * 64, "mime": "image/png", "bytes": 12}]}
        summary = task_asset_summary(asset)
        self.assertEqual(summary["creation_input"], asset["creation_input"])
        self.assertEqual(summary["feedback"], asset["feedback"])
        self.assertEqual(summary["approved_sha256"], asset["approved_sha256"])
        self.assertNotIn("history", summary)

    def test_pipeline_revision_builds_compact_context_before_passing_it_to_the_brain(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            novel = "合成小说全文。" * 300 + "不可丢失末尾。"
            core.create_project(root, novel, "live")
            core.install_plan(root, pipeline.demo_plan())
            core.approve(root, "plan", "plan", 1)
            with core.transaction(root) as state:
                state["assets"][0]["history"] = [{"runtime": RUNTIME * 20000}]
                saved = state["episodes"][0]["shots"][0]
                saved.update(feedback="当前人工作品调整意见完整保留。", history=[{"file": "media/old.mp4", "version": 1, "sha256": "8" * 64,
                    "reference_manifest": {"context": {"raw": RUNTIME * 20000}}}], prompt_history=[{"raw": RUNTIME * 20000}])
            shot = core.load_project(root)["episodes"][0]["shots"][0]
            with patch.object(pipeline, "_brain", return_value={"needs_assets": False, "additions": [], "shot_updates": [],
                    "blockers": [], "summary": "普通调整。"}) as brain:
                self.assertTrue(pipeline._revision_analysis(root, shot))
            _, method, sent_shot, sent_context, sent_feedback = brain.call_args.args
            self.assertEqual(method, "plan_revision")
            self.assertNotIn(RUNTIME, json.dumps([sent_shot, sent_context], ensure_ascii=False))
            self.assertEqual(sent_context["novel"], novel)
            self.assertEqual(sent_feedback, shot["feedback"])
            self.assertEqual(sent_shot["duration"], shot["duration"])


if __name__ == "__main__":
    unittest.main()
