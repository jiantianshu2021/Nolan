"""Synthetic end-to-end planning recovery; real gates and no paid providers."""
import copy
import unittest
from unittest.mock import patch

from norlan import core, future_continuity, pipeline, storyboard_revision
import test_prompt_gate as fixtures


REAL_BRAIN = pipeline._brain


def redesign():
    return {"summary": "按最新反馈改为三镜，父亲快速坠落、绝望特写、被接住后摇晃站稳，待完整提示词确认。",
            "duration": 10,
            "subshots": [{"start": 0, "end": 3, "prompt": "中景，父亲快速坠落。"},
                         {"start": 3, "end": 5, "prompt": "切父亲绝望的脸部近景。"},
                         {"start": 5, "end": 10, "prompt": "切回飞板接住父亲，他摇晃后站稳。"}]}


class FutureContinuityPipelineTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.PromptGateTests(methodName="runTest")
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.root = self.fixture.root
        with core.transaction(self.root) as state:
            target, neighbor = state["episodes"][0]["shots"]
            target.update(duration=6, version=3, revision=2, status="planned",
                prompt="合成旧动作：父亲坠落，飞板接近。", dialogue="",
                subshots=[{"start": 0, "end": 3, "prompt": "父亲坠落。"},
                          {"start": 3, "end": 6, "prompt": "飞板接近父亲。"}],
                feedback="保持上面的不变，先快速坠落，切父亲绝望表情，再切回飞板接住，摇晃后站稳。")
            neighbor.update(duration=8, prompt="飞板初托，父亲趴稳，随后查看飞板。", dialogue="父亲：这是什么？",
                continuity_notes="原计划的初托到趴稳动作。",
                subshots=[{"start": 0, "end": 4, "prompt": "飞板初托父亲，父亲趴稳。"},
                          {"start": 4, "end": 8, "prompt": "父亲观察飞板并说原对白。"}])
        self.before = core.load_project(self.root)
        self.target_id = self.before["episodes"][0]["shots"][0]["id"]
        self.neighbor_id = self.before["episodes"][0]["shots"][1]["id"]
        self.result = {"needs_assets": False, "additions": [], "shot_updates": [], "blockers": [],
            "summary": "按最新动作安排当前段，并协调下一段去除重复初托，完整提示词仍待用户确认。",
            "storyboard_revision": redesign(),
            "future_continuity_updates": [{"id": self.neighbor_id,
                "summary": "下一段从父亲已站稳继续，去掉重复初托和趴稳，保留观察飞板剧情与对白，仍待完整提示词确认。",
                "prompt": "承接父亲已经站稳的结尾，他保持站姿查看飞板，不重演初托与趴稳。",
                "continuity_notes": "从前段站稳姿态接续，后续剧情与原对白不变。",
                "subshots": [{"start": 0, "end": 4, "prompt": "父亲已经站稳，低头查看飞板。"},
                             {"start": 4, "end": 8, "prompt": "父亲观察飞板并说原对白。"}]}]}
        self.calls = []
        self.fixture.brain.side_effect = self.answer

    def answer(self, root, method, *args, **kwargs):
        self.calls.append((method, copy.deepcopy(args), copy.deepcopy(kwargs)))
        if method == "plan_revision":
            return copy.deepcopy(self.result)
        return self.fixture.answer(root, method, *args, **kwargs)

    def shot(self, neighbor=False):
        return core._find(core.load_project(self.root), "shot", self.neighbor_id if neighbor else self.target_id)

    def seed_legacy_block(self):
        with core.transaction(self.root) as state:
            target = core._find(state, "shot", self.target_id)
            target["revision_analysis"] = {"signature": storyboard_revision.feedback_signature(target),
                "summary": "当前草稿10秒三镜，但后续仍要求初托趴稳。",
                "blockers": ["下一段固定动作为趴稳，与当前要求站稳冲突，请补充意见。"],
                "storyboard_revision": redesign(), "storyboard_baseline": storyboard_revision.baseline(target)}
            state.update(status="blocked", last_error="合成旧规划冲突。")

    def planner_count(self):
        return sum(method == "plan_revision" for method, _, _ in self.calls)

    def test_legacy_block_migrates_once_to_complete_prompt_without_video_submission(self):
        self.seed_legacy_block()
        self.assertEqual(pipeline.run(self.root)["status"], "awaiting_prompt_approval")
        target = self.shot()
        self.assertEqual(target["revision_analysis"]["planning_policy_version"], future_continuity.PLANNING_POLICY_VERSION)
        self.assertEqual(target["revision_analysis"]["blockers"], [])
        self.assertEqual(target["duration"], 10)
        self.assertEqual(target["prompt_candidate"]["version"], 4)
        self.assertIn("摇晃后站稳", target["prompt_candidate"]["prompt"])
        self.assertEqual(self.planner_count(), 1)
        revision_args = next(args for method, args, _ in self.calls if method == "plan_revision")
        self.assertEqual(revision_args[-1], self.before["episodes"][0]["shots"][0]["feedback"])
        self.assertIn("blocker_repair", revision_args[1])
        self.assertEqual(core.load_project(self.root)["budget"], self.before["budget"])
        self.fixture.api.submit_video.assert_not_called()

    def test_future_plan_keeps_canonical_fields_and_does_not_claim_approval(self):
        self.assertEqual(pipeline.run(self.root)["status"], "awaiting_prompt_approval")
        before = self.before["episodes"][0]["shots"][1]
        target, neighbor = self.shot(), self.shot(neighbor=True)
        for key in ("id", "duration", "scene_id", "dialogue", "asset_ids", "version", "revision", "feedback", "status"):
            self.assertEqual(neighbor.get(key), before.get(key), key)
        self.assertEqual([(s["start"], s["end"]) for s in neighbor["subshots"]],
                         [(s["start"], s["end"]) for s in before["subshots"]])
        self.assertEqual(neighbor["prompt"], self.result["future_continuity_updates"][0]["prompt"])
        self.assertEqual(neighbor["subshots"], self.result["future_continuity_updates"][0]["subshots"])
        self.assertNotIn("prompt_candidate", neighbor)
        self.assertNotIn("approved_at", neighbor)
        self.assertIn("future_continuity_updates", target["prompt_candidate"])
        self.assertEqual(target["prompt_candidate"]["future_continuity_updates"][0]["id"], self.neighbor_id)
        self.assertEqual(core.load_project(self.root)["assets"], self.before["assets"])
        self.fixture.api.submit_video.assert_not_called()

    def test_continuing_and_restarting_reuse_current_draft_without_new_planner_call(self):
        self.seed_legacy_block()
        pipeline.run(self.root)
        before = core.load_project(self.root)
        for _ in range(3):
            self.assertEqual(pipeline.run(self.root)["status"], "awaiting_prompt_approval")
        with patch.object(pipeline, "_brain", side_effect=AssertionError("completed plan must survive restart")):
            self.assertTrue(pipeline._revision_analysis(self.root, self.shot()))
        self.assertEqual(core.load_project(self.root)["episodes"][0]["shots"], before["episodes"][0]["shots"])
        self.assertEqual(self.planner_count(), 1)
        self.fixture.api.submit_video.assert_not_called()

    def test_rejecting_current_prompt_rolls_back_only_unsubmitted_future_draft(self):
        pipeline.run(self.root)
        target = self.shot()
        candidate = target["prompt_candidate"]
        core.reject_prompt(self.root, self.target_id, candidate["id"], candidate["request_hash"], "保持原有结尾，不再改下一段接续。")
        self.assertEqual(self.shot(neighbor=True), self.before["episodes"][0]["shots"][1])
        self.assertEqual(self.shot()["duration"], 6)
        self.assertNotIn("future_continuity_binding", self.shot())
        self.assertTrue(self.shot().get("future_continuity_history"))
        self.fixture.api.submit_video.assert_not_called()

    def test_continuity_only_draft_is_also_idempotent_and_reversible(self):
        self.result["storyboard_revision"] = None
        self.assertEqual(pipeline.run(self.root)["status"], "awaiting_prompt_approval")
        target = self.shot()
        self.assertEqual(target["duration"], 6)
        self.assertNotIn("storyboard_revision", target)
        with patch.object(pipeline, "_brain", side_effect=AssertionError("reuse continuity-only plan")):
            self.assertTrue(pipeline._revision_analysis(self.root, target))
        candidate = target["prompt_candidate"]
        core.reject_prompt(self.root, self.target_id, candidate["id"], candidate["request_hash"], "取消这次接续修改。")
        self.assertEqual(self.shot(neighbor=True), self.before["episodes"][0]["shots"][1])
        self.fixture.api.submit_video.assert_not_called()

    def test_started_or_user_edited_neighbor_cannot_be_rewritten(self):
        for changes in ({"version": 1}, {"attempts": [{"job_id": "historical-task"}]},
                        {"prompt_candidate": {"id": "existing-prompt"}},
                        {"feedback": "保留用户指定的开头。", "revision": 1}):
            with self.subTest(changes=changes):
                with core.transaction(self.root) as state:
                    target = core._find(state, "shot", self.target_id)
                    target.clear()
                    target.update(copy.deepcopy(self.before["episodes"][0]["shots"][0]))
                    neighbor = core._find(state, "shot", self.neighbor_id)
                    neighbor.clear()
                    neighbor.update(copy.deepcopy(self.before["episodes"][0]["shots"][1]))
                    neighbor.update(changes)
                    state.update(status="ready", last_error=None)
                frozen = copy.deepcopy(self.shot(neighbor=True))
                state = pipeline.run(self.root)
                self.assertEqual(state["status"], "blocked")
                self.assertEqual(self.shot(neighbor=True), frozen)
                self.assertEqual(self.shot()["duration"], 6)
        self.fixture.api.submit_video.assert_not_called()

    def test_neighbor_tampering_invalidates_existing_current_prompt_consent(self):
        pipeline.run(self.root)
        candidate = self.shot()["prompt_candidate"]
        with core.transaction(self.root) as state:
            neighbor = core._find(state, "shot", self.neighbor_id)
            neighbor["subshots"][0]["prompt"] = "合成篡改：改回趴稳。"
        with self.assertRaises(core.GateError):
            core.approve_prompt(self.root, self.target_id, candidate["id"], candidate["request_hash"])
        self.assertNotIn("prompt_approval", self.shot())
        self.fixture.api.submit_video.assert_not_called()

    def test_asset_gate_preserves_same_future_plan_until_actual_asset_approval(self):
        self.result = {**self.fixture.amendment(), "storyboard_revision": redesign(),
                       "future_continuity_updates": self.result["future_continuity_updates"]}
        self.assertEqual(pipeline.run(self.root)["status"], "awaiting_asset_amendment")
        self.assertEqual(self.shot(neighbor=True), self.before["episodes"][0]["shots"][1])
        amendment = core.load_project(self.root)["asset_amendments"][-1]
        core.apply_asset_amendment(self.root, amendment["id"], "approve")
        self.assertEqual(pipeline.run(self.root)["status"], "awaiting_asset_input")
        fixtures.choose_gpt_assets(self.root)
        self.assertEqual(pipeline.run(self.root)["status"], "awaiting_assets")
        self.assertEqual(self.shot(neighbor=True), self.before["episodes"][0]["shots"][1])
        voice = core._find(core.load_project(self.root), "asset", "system-voice")
        core.approve(self.root, "asset", voice["id"], voice["version"])
        self.assertEqual(pipeline.run(self.root)["status"], "awaiting_prompt_approval")
        self.assertEqual(self.shot(neighbor=True)["prompt"], self.result["future_continuity_updates"][0]["prompt"])
        self.assertEqual(self.shot()["duration"], 10)
        self.assertIn("system-voice", self.shot()["asset_ids"])
        self.assertEqual(self.planner_count(), 1)
        self.assertEqual(self.fixture.api.submit_video.call_count, 1)  # approved synthetic voice audition only
        pipeline.run(self.root)
        self.assertEqual(self.fixture.api.submit_video.call_count, 1)

    def test_completed_revision_result_is_saved_and_reused_without_second_model_submission(self):
        def submit(client, *args, **kwargs):
            client.before_submit()
            return copy.deepcopy(self.result)

        config = {"provider": "codex", "billing": "subscription", "model": "synthetic-model"}
        with patch("norlan.connection.production_brain_config", return_value=config), \
                patch("norlan.brain.Brain.plan_revision", autospec=True, side_effect=submit) as planner, \
                patch("norlan.brain.subprocess.run", side_effect=AssertionError("synthetic transport only")):
            first = REAL_BRAIN(self.root, "plan_revision", self.shot(), {}, self.shot()["feedback"],
                               _cache_key="synthetic-revision-cache")
            jobs = core.load_project(self.root)["jobs"]
            job_id, job = next((identifier, value) for identifier, value in jobs.items()
                               if value.get("item_id") == "plan_revision")
            self.assertEqual((job["state"], job["accounting"]), ("completed", "spent"))
            self.assertEqual(job["post_result_hash"], core._json_digest(first))
            llm_calls = core.load_project(self.root)["llm_calls"]
            second = REAL_BRAIN(self.root, "plan_revision", self.shot(), {}, self.shot()["feedback"],
                                _cache_key="synthetic-revision-cache")
            self.assertEqual(second, first)
            first["future_continuity_updates"][0]["prompt"] = "调用者改动不能污染缓存。"
            self.assertEqual(core.load_project(self.root)["jobs"][job_id]["post_result"], self.result)
            self.assertEqual(core.load_project(self.root)["llm_calls"], llm_calls)
            self.assertEqual(planner.call_count, 1)
            with core.transaction(self.root) as state:
                state["jobs"][job_id]["post_result"]["summary"] = "合成缓存被篡改。"
            with self.assertRaises(core.GateError):
                REAL_BRAIN(self.root, "plan_revision", self.shot(), {}, self.shot()["feedback"],
                           _cache_key="synthetic-revision-cache")
            self.assertEqual(planner.call_count, 1)
        self.fixture.api.submit_video.assert_not_called()


if __name__ == "__main__":
    unittest.main()
