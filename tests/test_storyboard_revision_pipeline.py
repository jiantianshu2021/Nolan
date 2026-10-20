"""Synthetic offline local redesign: real gates, fake media/providers only."""
import copy
import unittest
from unittest.mock import patch

from norlan import core, pipeline, prompt_gate, storyboard_revision
import test_prompt_gate as fixtures
from test_pipeline import run_confirming_asset_prompts


def redesign(duration=10):
    return {"summary": "按本次反馈重排为两个连续摄影镜头。", "duration": duration,
            "subshots": [{"start": 0, "end": duration / 2, "prompt": "中景，人物在桌旁抬头。"},
                         {"start": duration / 2, "end": duration, "prompt": "切近景，人物平静地转头。"}]}


class StoryboardRevisionPipelineTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.PromptGateTests(methodName="runTest")
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.root = self.fixture.root
        old_evidence = pipeline.media.extract_evidence.side_effect
        self.enterContext(patch.object(pipeline.media, "extract_evidence",
                                      side_effect=lambda video, directory, **kwargs: old_evidence(video, directory)))
        self.result = {"needs_assets": False, "additions": [], "shot_updates": [],
                       "summary": "调整目标片段内部镜头。", "blockers": [], "storyboard_revision": redesign()}
        self.inputs = []
        self.fixture.brain.side_effect = self.answer

    def answer(self, root, method, *args, **kwargs):
        self.inputs.append((method, copy.deepcopy(args)))
        if method == "plan_revision":
            return copy.deepcopy(self.result)
        return self.fixture.answer(root, method, *args, **kwargs)

    def shot(self):
        return self.fixture.shot()

    def feedback(self, text="把这个片段重新设计为两个镜头，时长由你安排。"):
        pipeline.run(self.root)
        current = self.shot()
        candidate = current["prompt_candidate"]
        core.reject_prompt(self.root, current["id"], candidate["id"], candidate["request_hash"], text)

    def test_redesign_reloads_shot_and_waits_for_one_exact_prompt_confirmation(self):
        before = core.load_project(self.root)
        self.feedback()
        self.assertEqual(pipeline.run(self.root)["status"], "awaiting_prompt_approval")
        current = self.shot()
        self.assertEqual((current["duration"], len(current["subshots"])), (10, 2))
        self.assertEqual(current["prompt_candidate"]["storyboard_revision"], current["storyboard_revision"])
        self.assertEqual(current["storyboard_revision"]["before_duration"], 4)
        self.assertEqual(current["storyboard_revision"]["before_count"], 1)
        refined = next(args[0] for method, args in reversed(self.inputs) if method == "refine_shot")
        self.assertEqual(refined["subshots"], redesign()["subshots"])
        self.assertEqual(refined["duration"], 10)
        for value in redesign()["subshots"]:
            self.assertIn(value["prompt"], current["prompt_candidate"]["prompt"])
        self.fixture.api.submit_video.assert_not_called()
        for _ in range(3):
            pipeline.run(self.root)
        self.assertEqual(sum(method == "plan_revision" for method, _ in self.inputs), 1)
        with patch.object(pipeline, "_brain", side_effect=AssertionError("cached redesign must survive resume")):
            self.assertTrue(pipeline._revision_analysis(self.root, self.shot()))
        self.fixture.confirm()
        self.assertEqual(pipeline.run(self.root)["status"], "review_required")
        self.fixture.api.submit_video.assert_called_once()
        self.assertEqual(self.fixture.api.submit_video.call_args.args[1], 10)
        job = core.load_project(self.root)["jobs"][self.shot()["job_id"]]
        self.assertEqual(job["duration"], 10)
        # This synthetic model has no public price: obsolete per-second caps
        # must not become either a charge or a prerequisite for the redesign.
        self.assertEqual(job["amount"], 0)
        self.assertTrue(job["cost_track"])
        self.assertIsNone(job.get("billing", {}).get("actual_cost"))
        self.assertEqual(core.load_project(self.root)["episodes"][0]["shots"][1], before["episodes"][0]["shots"][1])
        self.assertEqual(core.load_project(self.root)["assets"], before["assets"])

    def test_prompt_rejection_discards_only_unsubmitted_draft_and_archives_it(self):
        self.feedback()
        pipeline.run(self.root)
        old = self.shot()["prompt_candidate"]
        core.reject_prompt(self.root, self.shot()["id"], old["id"], old["request_hash"], "取消刚才的重分镜，保持原镜头。")
        self.assertEqual(self.shot()["duration"], 4)
        self.assertNotIn("subshots", self.shot())
        self.assertNotIn("storyboard_revision", self.shot())
        self.assertEqual(self.shot()["storyboard_revision_history"][-1]["status"], "superseded_draft")
        self.result["storyboard_revision"] = None
        pipeline.run(self.root)
        self.assertNotIn("storyboard_revision", self.shot()["prompt_candidate"])
        with self.assertRaises(core.GateError):
            core.approve_prompt(self.root, self.shot()["id"], old["id"], old["request_hash"])
        self.fixture.api.submit_video.assert_not_called()

    def test_submitted_version_definition_is_kept_and_old_media_is_immutable(self):
        self.feedback()
        pipeline.run(self.root)
        self.fixture.confirm()
        pipeline.run(self.root)
        current = self.shot()
        original = (self.root / current["file"]).read_bytes()
        core.reject(self.root, "shot", current["id"], current["version"], "第二镜头动作稍慢，其余保留。")
        queued = self.shot()
        self.assertEqual((queued["duration"], queued["subshots"]), (10, redesign()["subshots"]))
        archive = queued["history"][-1]
        for key in ("duration", "subshots", "prompt", "refined_prompt", "reference_manifest", "prompt_candidate", "job_id"):
            self.assertEqual(archive[key], current[key])
        self.assertEqual((self.root / current["file"]).read_bytes(), original)
        self.result["storyboard_revision"] = None
        pipeline.run(self.root)
        self.assertEqual(self.shot()["duration"], 10)
        self.fixture.api.submit_video.assert_called_once()

    def test_queued_returned_segment_accepts_new_feedback_through_same_reject(self):
        pipeline.run(self.root)
        self.fixture.confirm()
        pipeline.run(self.root)
        old = self.shot()
        core.reject(self.root, "shot", old["id"], old["version"], "先稍微改一下动作。")
        queued = self.shot()
        self.assertTrue(core.can_edit_shot_revision(core.load_project(self.root), queued))
        core.reject(self.root, "shot", queued["id"], queued["version"], "改成两个镜头，由你决定新时长。")
        self.assertEqual(self.shot()["revision"], queued["revision"] + 1)
        self.assertEqual(len(self.shot()["history"]), 1)
        pipeline.run(self.root)
        self.assertEqual(self.shot()["duration"], 10)
        self.fixture.api.submit_video.assert_called_once()

    def test_asset_and_storyboard_request_preserves_cached_plan_until_real_asset_approval(self):
        self.result = {**self.fixture.amendment(), "storyboard_revision": redesign()}
        self.feedback("增加女性声音资产，并重新规划两个镜头和时长。")
        self.assertEqual(pipeline.run(self.root)["status"], "awaiting_asset_amendment")
        self.assertNotIn("subshots", self.shot())
        self.assertEqual(self.shot()["duration"], 4)
        state = core.load_project(self.root)
        amendment = state["asset_amendments"][-1]
        core.apply_asset_amendment(self.root, amendment["id"], "approve")
        self.assertEqual(pipeline.run(self.root)["status"], "awaiting_asset_input")
        fixtures.choose_gpt_assets(self.root)
        self.assertEqual(pipeline.run(self.root)["status"], "awaiting_asset_prompt_approval")
        self.fixture.api.submit_video.assert_not_called()
        self.assertNotIn("subshots", self.shot())
        self.assertEqual(run_confirming_asset_prompts(self.root)["status"], "awaiting_assets")
        self.assertNotIn("subshots", self.shot())
        voice = core._find(core.load_project(self.root), "asset", "system-voice")
        core.approve(self.root, "asset", voice["id"], voice["version"])
        self.assertEqual(pipeline.run(self.root)["status"], "awaiting_prompt_approval")
        self.assertEqual(self.shot()["subshots"], redesign()["subshots"])
        self.assertEqual(self.shot()["duration"], 10)
        self.assertEqual(sum(method == "plan_revision" for method, _ in self.inputs), 1)
        self.fixture.api.submit_video.assert_called_once()  # only the approved voice audition
        self.assertIn("system-voice", self.shot()["asset_ids"])
        pipeline.run(self.root)
        self.fixture.api.submit_video.assert_called_once()

    def test_stale_camera_summary_duration_or_schedule_cannot_use_previous_consent(self):
        self.feedback()
        pipeline.run(self.root)
        original = self.shot()
        candidate = original["prompt_candidate"]
        for mutate in (lambda sh: sh["storyboard_revision"].update(summary="篡改说明"),
                       lambda sh: sh.update(duration=12),
                       lambda sh: sh["subshots"][0].update(prompt="篡改摄影内容"),
                       lambda sh: sh["prompt_candidate"]["storyboard_revision"].update(before_duration=5)):
            with self.subTest(mutate=mutate):
                with core.transaction(self.root) as state:
                    sh = core._find(state, "shot", original["id"])
                    sh.clear()
                    sh.update(copy.deepcopy(original))
                    mutate(sh)
                with self.assertRaises(core.GateError):
                    core.approve_prompt(self.root, original["id"], candidate["id"], candidate["request_hash"])
        self.fixture.api.submit_video.assert_not_called()

    def test_bad_redesign_is_cached_as_blocker_without_repeated_planner(self):
        self.result["storyboard_revision"]["subshots"][0]["end"] = 8
        self.feedback()
        self.assertEqual(pipeline.run(self.root)["status"], "blocked")
        self.assertEqual(pipeline.run(self.root)["status"], "blocked")
        self.assertEqual(sum(method == "plan_revision" for method, _ in self.inputs), 1)
        self.fixture.api.submit_video.assert_not_called()

    def test_plan_rejects_extra_fields_or_outside_single_task_limit(self):
        source = self.shot()
        for proposal in ({**redesign(), "scene_id": "another-room"},
                         {**redesign(), "duration": 16}, {**redesign(), "duration": True},
                         {**redesign(), "dialogue": "改了人物原文"}):
            with self.subTest(proposal=proposal), self.assertRaises(ValueError):
                storyboard_revision.validate_plan(source, proposal)

    def test_planning_block_feedback_uses_the_same_redesign_and_final_prompt_gate(self):
        self.result["blockers"] = ["本次要求需要重新安排动作因果。"]
        self.feedback()
        self.assertEqual(pipeline.run(self.root)["status"], "blocked")
        current = self.shot()
        signature = current["revision_analysis"]["signature"]
        core.submit_planning_feedback(self.root, current["id"], "先抬头再转身，按自然节奏重新设计。",
                                      current["version"], signature)
        self.result.update(blockers=[], storyboard_revision=redesign(12))
        self.assertEqual(pipeline.run(self.root)["status"], "awaiting_prompt_approval")
        self.assertEqual(self.shot()["duration"], 12)
        self.assertEqual(sum(method == "plan_revision" for method, _ in self.inputs), 2)
        self.fixture.api.submit_video.assert_not_called()

    def test_stale_baseline_never_applies_and_cannot_bypass_an_asset_gate(self):
        source = self.shot()
        with core.transaction(self.root) as state:
            sh = core._find(state, "shot", source["id"])
            sh.update(feedback="合成重分镜请求", revision=1)
        current = self.shot()
        record = {"signature": storyboard_revision.feedback_signature(current), "storyboard_revision": redesign(),
                  "storyboard_baseline": storyboard_revision.baseline(current)}
        with core.transaction(self.root) as state:
            core._find(state, "shot", source["id"])["dialogue"] = "测试角色：新的原文。"
        with self.assertRaisesRegex(core.GateError, "过期"):
            storyboard_revision.apply(self.root, source["id"], record)
        self.assertEqual(self.shot()["duration"], 4)
        self.assertFalse(core.can_edit_shot_revision(core.load_project(self.root), self.shot()))
        self.fixture.api.submit_video.assert_not_called()


if __name__ == "__main__":
    unittest.main()
