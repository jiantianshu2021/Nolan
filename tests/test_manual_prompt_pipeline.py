"""Synthetic exact-text manual prompt consent; providers and GPT stay mocked."""
import copy
import json
import unittest
from unittest.mock import patch

from norlan import core, pipeline
import test_prompt_gate as prompt_fixture


class ManualPromptPipelineTests(unittest.TestCase):
    # Reuse the offline assets/provider fixture without inheriting its test cases.
    setUp = prompt_fixture.PromptGateTests.setUp
    answer = prompt_fixture.PromptGateTests.answer
    amendment = prompt_fixture.PromptGateTests.amendment
    shot = prompt_fixture.PromptGateTests.shot
    confirm = prompt_fixture.PromptGateTests.confirm

    def prepare(self):
        self.assertEqual(pipeline.run(self.root)["status"], "awaiting_prompt_approval")
        return copy.deepcopy(self.shot()["prompt_candidate"])

    def edit(self, text, *, shot=None, candidate=None):
        shot = shot or self.shot()
        candidate = candidate or shot["prompt_candidate"]
        return core.edit_prompt_and_approve(self.root, shot["id"], candidate["id"],
                                            candidate["request_hash"], text)

    def forbid_gpt(self):
        self.brain.reset_mock()
        self.brain.side_effect = AssertionError("Manual exact-text submission must not call GPT")

    def test_manual_exact_text_submits_once_without_gpt_and_survives_resume(self):
        original = self.prepare()
        raw = "  手动定稿：快速下坠。\n切父亲绝望特写，再切飞行板接住父亲；踉跄后站稳。\n保留参考图1。  "
        self.forbid_gpt()
        result = self.edit(raw)
        self.assertEqual(result["status"], "ready")
        current = self.shot()
        self.assertEqual(current["status"], "prompt_approved")
        self.assertEqual(current["prompt_candidate"]["prompt"], raw)
        self.assertNotEqual(current["prompt_candidate"]["id"], original["id"])
        self.api.submit_video.assert_not_called()

        self.assertEqual(pipeline.run(self.root)["status"], "review_required")
        prompt, duration, references = self.api.submit_video.call_args.args
        self.assertEqual(prompt, raw)
        self.assertEqual(duration, self.shot()["duration"])
        self.assertEqual([str(path.relative_to(self.root)) for path in references],
                         original["reference_manifest"]["images"])
        self.assertEqual(self.shot()["version"], original["version"])
        for _ in range(3):
            self.assertEqual(pipeline.run(self.root)["status"], "review_required")
        self.api.submit_video.assert_called_once()
        self.brain.assert_not_called()
        self.assertFalse(core.human_approved_shot(self.shot()))

    def test_manual_route_does_not_reinterpret_feedback_or_add_voice_references(self):
        # The existing feedback was already interpreted when GPT prepared the
        # candidate. Editing that candidate is final text, not new instructions.
        with core.transaction(self.root) as state:
            state["episodes"][0]["shots"][0]["feedback"] = "仅调整人物动作，保留现有资产。"
        self.prepare()
        self.forbid_gpt()
        self.edit("手动确认的完整最终提示词。")
        with patch.object(pipeline, "_revision_analysis", side_effect=AssertionError("Do not re-plan")) as planner, \
             patch.object(pipeline, "_system_voice_reference", side_effect=AssertionError("Do not mutate references")) as voices, \
             patch.object(core, "resolve_completed_asset_feedback", side_effect=AssertionError("Do not rewrite feedback")) as feedback:
            self.assertEqual(pipeline.run(self.root)["status"], "review_required")
        planner.assert_not_called()
        voices.assert_not_called()
        feedback.assert_not_called()
        self.brain.assert_not_called()

    def test_interrupted_manual_submission_without_task_id_never_rewrites_or_resubmits(self):
        self.prepare()
        raw = "中断前用户手动确认的原文。"
        self.edit(raw)
        candidate = copy.deepcopy(self.shot()["prompt_candidate"])
        with core.transaction(self.root) as state:
            shot = state["episodes"][0]["shots"][0]
            shot.update(status="generating", version=candidate["version"])
        self.forbid_gpt()
        for _ in range(3):
            self.assertEqual(pipeline.run(self.root)["status"], "blocked")
            self.assertEqual(self.shot()["prompt_candidate"]["prompt"], raw)
            self.assertEqual(self.shot()["prompt_candidate"]["request_hash"], candidate["request_hash"])
        self.brain.assert_not_called()
        self.api.submit_video.assert_not_called()

    def test_source_candidate_is_preserved_and_new_consent_is_bound_to_exact_text(self):
        original = self.prepare()
        assets = copy.deepcopy(core.load_project(self.root)["assets"])
        self.edit("用户定稿；不能恢复为旧的GPT文字。")
        current = self.shot()
        saved_sources = current.get("prompt_history", [])
        self.assertTrue(any(item.get("id") == original["id"] and item.get("prompt") == original["prompt"]
                            for item in saved_sources))
        manual = current["prompt_candidate"]
        self.assertNotEqual(manual["request_hash"], original["request_hash"])
        self.assertEqual(manual["reference_manifest"], original["reference_manifest"])
        self.assertEqual(current["prompt_approval"]["request_hash"], manual["request_hash"])
        self.assertEqual(core.load_project(self.root)["assets"], assets)
        self.api.submit_video.assert_not_called()

    def test_stale_or_duplicate_manual_submit_cannot_approve_another_request(self):
        original = self.prepare()
        state = core.load_project(self.root)
        with self.assertRaises(core.GateError):
            core.edit_prompt_and_approve(self.root, self.shot()["id"], original["id"], "0" * 64, "手工文本")
        self.assertEqual(core.load_project(self.root), state)
        self.edit("人工直接定稿。")
        saved = copy.deepcopy(self.shot())
        with self.assertRaises(core.GateError):
            self.edit("不能由重复请求替换这次定稿。", candidate=original)
        self.assertEqual(self.shot(), saved)
        self.assertEqual(pipeline.run(self.root)["status"], "review_required")
        self.api.submit_video.assert_called_once()

    def test_invalid_or_paused_manual_edit_does_not_discard_source_candidate(self):
        original = self.prepare()
        for raw in (None, False, {}, "", "  \n\t", "bad\x00text"):
            with self.subTest(raw=raw):
                with self.assertRaises((ValueError, core.GateError)):
                    self.edit(raw)
                self.assertEqual(self.shot()["prompt_candidate"], original)
        core.set_paused(self.root, True)
        with self.assertRaises(core.GateError):
            self.edit("人工文本")
        self.assertEqual(self.shot()["prompt_candidate"], original)
        self.api.submit_video.assert_not_called()

    def test_asset_changed_before_manual_submit_requires_new_candidate(self):
        original = self.prepare()
        with core.transaction(self.root) as state:
            state["assets"][0]["version"] += 1
        with self.assertRaises(core.GateError):
            self.edit("不能沿用素材变化前的文字确认。")
        self.api.submit_video.assert_not_called()
        self.assertEqual(self.shot()["prompt_candidate"]["id"], original["id"])

    def test_provider_changed_after_manual_approval_blocks_before_submission(self):
        self.prepare()
        raw = "用户确认的当前模型请求。"
        self.edit(raw)
        approved = copy.deepcopy(self.shot()["prompt_candidate"])
        self.forbid_gpt()
        core.configure(self.root, {"ark": {"resolution": "480p"}})
        for _ in range(3):
            self.assertEqual(pipeline.run(self.root)["status"], "blocked")
            self.assertEqual(self.shot()["prompt_candidate"]["prompt"], raw)
            self.assertEqual(self.shot()["prompt_candidate"]["request_hash"], approved["request_hash"])
            self.assertEqual(self.shot()["prompt_approval"]["request_hash"], approved["request_hash"])
        self.api.submit_video.assert_not_called()
        self.brain.assert_not_called()

    def test_rebuilt_reference_manifest_mismatch_preserves_manual_text_without_gpt(self):
        self.prepare()
        raw = "人工已定稿：发生参考变化时也不能退回GPT擅自改写。"
        self.edit(raw)
        approved = copy.deepcopy(self.shot()["prompt_candidate"])
        self.forbid_gpt()
        original_context = pipeline._shot_context

        def changed_context(*args, **kwargs):
            context, images, voices = original_context(*args, **kwargs)
            context["synthetic_dependency_change"] = "SYNTHETIC DIFFERENT CONTEXT"
            return context, images, voices

        with patch.object(pipeline, "_shot_context", side_effect=changed_context):
            for _ in range(3):
                self.assertEqual(pipeline.run(self.root)["status"], "blocked")
                self.assertEqual(self.shot()["prompt_candidate"], approved)
        self.api.submit_video.assert_not_called()
        self.brain.assert_not_called()

    def test_previous_approved_video_change_invalidates_manual_consent(self):
        self.prepare()
        self.confirm()
        self.assertEqual(pipeline.run(self.root)["status"], "review_required")
        first = self.shot()
        core.approve(self.root, "shot", first["id"], first["version"])
        self.assertEqual(pipeline.run(self.root)["status"], "awaiting_prompt_approval")
        second = copy.deepcopy(core.load_project(self.root)["episodes"][0]["shots"][1])
        self.edit("人工确认承接前段最后一帧。", shot=second)
        self.forbid_gpt()
        with core.transaction(self.root) as state:
            state["episodes"][0]["shots"][0]["human_decision"]["decision"] = "revision_requested"
        with self.assertRaises(core.GateError):
            pipeline._shot(self.root, "E001", second["id"], self.shot())
        self.api.submit_video.assert_called_once()
        self.brain.assert_not_called()

    def test_manual_edit_keeps_budget_gate_before_any_paid_post(self):
        self.prepare()
        raw = "人工已确认，仍受预算控制。"
        self.edit(raw)
        approved = copy.deepcopy(self.shot()["prompt_candidate"])
        self.forbid_gpt()
        state = core.load_project(self.root)
        occupied = sum(state["budget"][key] for key in ("spent", "reserved", "unknown"))
        core.configure(self.root, limit=occupied)
        before_jobs = copy.deepcopy(core.load_project(self.root)["jobs"])
        before_llm_calls = core.load_project(self.root)["llm_calls"]
        for _ in range(3):
            self.assertEqual(pipeline.run(self.root)["status"], "blocked")
            current = self.shot()
            self.assertEqual(current["prompt_candidate"], approved)
            self.assertEqual(current["status"], "prompt_approved")
            self.assertEqual(current["version"], approved["version"] - 1)
            self.assertEqual(core.load_project(self.root)["llm_calls"], before_llm_calls)
        self.api.submit_video.assert_not_called()
        self.brain.assert_not_called()
        self.assertEqual(core.load_project(self.root)["jobs"], before_jobs)
        core.configure(self.root, limit=100)
        self.assertEqual(pipeline.run(self.root)["status"], "review_required")
        self.assertEqual(self.api.submit_video.call_args.args[0], raw)
        self.assertEqual(self.shot()["version"], approved["version"])
        self.api.submit_video.assert_called_once()
        self.assertEqual(core.load_project(self.root)["llm_calls"], before_llm_calls)
        self.brain.assert_not_called()

    def test_uncertain_manual_submission_never_retries(self):
        self.prepare()
        self.edit("人工最终定稿，单次请求。")
        self.forbid_gpt()
        self.api.submit_video.side_effect = TimeoutError("SYNTHETIC uncertain submission")
        self.assertEqual(pipeline.run(self.root)["status"], "blocked")
        current = self.shot()
        job = core.load_project(self.root)["jobs"][current["job_id"]]
        self.assertEqual(job["accounting"], "unknown")
        for _ in range(3):
            self.assertEqual(pipeline.run(self.root)["status"], "blocked")
        self.api.submit_video.assert_called_once()
        self.brain.assert_not_called()

    def test_provider_preflight_failure_retains_exact_manual_consent_until_retry(self):
        self.prepare()
        raw = "人工已定稿：预检失败后继续仍使用这份文本。"
        self.edit(raw)
        approved = copy.deepcopy(self.shot()["prompt_candidate"])
        before = core.load_project(self.root)
        self.forbid_gpt()
        self.api.preflight.side_effect = RuntimeError("SYNTHETIC unavailable credentials")
        for _ in range(3):
            self.assertEqual(pipeline.run(self.root)["status"], "blocked")
            current = self.shot()
            self.assertEqual(current["prompt_candidate"], approved)
            self.assertEqual(current["status"], "prompt_approved")
            self.assertEqual(current["version"], approved["version"] - 1)
            state = core.load_project(self.root)
            self.assertEqual(state["jobs"], before["jobs"])
            self.assertEqual(state["llm_calls"], before["llm_calls"])
        self.api.submit_video.assert_not_called()
        self.api.preflight.side_effect = None
        self.assertEqual(pipeline.run(self.root)["status"], "review_required")
        self.assertEqual(self.api.submit_video.call_args.args[0], raw)
        self.assertEqual(self.shot()["version"], approved["version"])
        self.api.submit_video.assert_called_once()
        self.brain.assert_not_called()

    def test_submitted_manual_job_recovers_original_task_without_gpt_or_new_post(self):
        self.prepare()
        raw = "人工最终文本：保留首尾换行。\n"
        self.edit(raw)
        self.forbid_gpt()
        with patch.object(pipeline, "_poll_video", side_effect=core.GateError("SYNTHETIC interrupted polling")):
            self.assertEqual(pipeline.run(self.root)["status"], "blocked")
        first_job = self.shot()["job_id"]
        self.assertEqual(core.load_project(self.root)["jobs"][first_job]["task_id"], "synthetic-1")
        self.assertEqual(pipeline.run(self.root)["status"], "review_required")
        self.assertEqual(self.shot()["job_id"], first_job)
        self.assertEqual(self.api.submit_video.call_args.args[0], raw)
        self.api.submit_video.assert_called_once()
        self.api.poll_video.assert_called_once_with("synthetic-1")
        self.brain.assert_not_called()

    def test_manual_prompt_rework_keeps_old_video_bytes_and_history(self):
        self.prepare()
        self.confirm()
        pipeline.run(self.root)
        old = copy.deepcopy(self.shot())
        old_bytes = (self.root / old["file"]).read_bytes()
        core.reject(self.root, "shot", old["id"], old["version"], "父亲接住后先踉跄再站稳。")
        self.assertEqual(pipeline.run(self.root)["status"], "awaiting_prompt_approval")
        self.forbid_gpt()
        self.edit("手动完整返修文本，保持父亲先踉跄再站稳。")
        self.assertEqual(pipeline.run(self.root)["status"], "review_required")
        current = self.shot()
        self.assertEqual(current["version"], old["version"] + 1)
        self.assertNotEqual(current["file"], old["file"])
        self.assertEqual((self.root / old["file"]).read_bytes(), old_bytes)
        self.assertTrue(any(item.get("file") == old["file"] and item.get("sha256") == old["sha256"]
                            for item in current["history"]))
        self.assertEqual(self.api.submit_video.call_count, 2)
        self.brain.assert_not_called()

    def test_manual_prompt_can_replace_timeline_text_without_rewriting_canonical_shots(self):
        subshots = [{"start": 0, "end": 2, "prompt": "旧中景动作。"},
                    {"start": 2, "end": 4, "prompt": "旧近景动作。"}]
        with core.transaction(self.root) as state:
            state["episodes"][0]["shots"][0]["subshots"] = copy.deepcopy(subshots)
        evidence = pipeline.media.extract_evidence.side_effect
        self.enterContext(patch.object(pipeline.media, "extract_evidence",
            side_effect=lambda video, directory, **kwargs: evidence(video, directory)))
        original = self.prepare()
        self.assertIn("旧中景动作", original["prompt"])
        raw = "人工重新写好的完整内容：先快速下坠，再切绝望表情，最后接住站稳。"
        self.edit(raw)
        self.forbid_gpt()
        self.assertEqual(pipeline.run(self.root)["status"], "review_required")
        self.assertEqual(self.api.submit_video.call_args.args[0], raw)
        self.assertEqual(self.api.submit_video.call_args.args[1], 4)
        self.assertEqual(self.shot()["subshots"], subshots)
        self.brain.assert_not_called()

    def test_manual_review_target_uses_actual_text_without_superseded_gpt_timing_claims(self):
        subshots = [{"start": 0, "end": 2, "prompt": "旧镜头一的过时检查目标。"},
                    {"start": 2, "end": 4, "prompt": "旧镜头二的过时检查目标。"}]
        with core.transaction(self.root) as state:
            shot = state["episodes"][0]["shots"][0]
            shot.update(subshots=copy.deepcopy(subshots), dialogue="系统：旧全文播报。")
        core.configure(self.root, {"require_voice_reference": False})
        evidence = pipeline.media.extract_evidence.side_effect
        self.enterContext(patch.object(pipeline.media, "extract_evidence",
            side_effect=lambda video, directory, **kwargs: evidence(video, directory)))
        self.prepare()
        raw = "人工只读关键内容，不照旧时间表。\n系统画外声：救援成功。"
        self.edit(raw)
        manual = self.shot()["prompt_candidate"]
        self.assertTrue(manual["manual_prompt"])
        self.assertEqual(manual["dialogue"], "")
        self.assertNotIn("timing_proposal", manual)
        self.assertNotIn("broadcast_proposal", manual)
        self.assertIn("人工", manual["continuity_notes"])
        self.assertIn("未由 GPT", manual["continuity_notes"])
        self.forbid_gpt()
        self.assertEqual(pipeline.run(self.root)["status"], "review_required")
        target = pipeline._review_shot_summary(self.shot())
        self.assertEqual(target["prompt"], raw)
        self.assertEqual(target.get("dialogue", ""), "")
        target_text = json.dumps(target, ensure_ascii=False)
        self.assertNotIn("旧镜头一的过时检查目标", target_text)
        self.assertNotIn("旧全文播报", target_text)
        self.brain.assert_not_called()


if __name__ == "__main__":
    unittest.main()
