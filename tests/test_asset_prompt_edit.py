"""Synthetic direct asset prompt edits: no GPT and no real provider calls."""
import copy
import unittest
from unittest.mock import patch

from norlan import asset_prompt_edit as edit, asset_prompt_gate as gate, core, pipeline
from norlan.providers import ProviderError
from tests import test_asset_prompt_pipeline as fixtures


class AssetPromptEditTests(unittest.TestCase):
    # Reuse only the synthetic fixture helpers, not the base test methods.
    setUp = fixtures.AssetPromptPipelineTests.setUp
    provider_for = fixtures.AssetPromptPipelineTests.provider_for
    generate_image = fixtures.AssetPromptPipelineTests.generate_image
    submit_video = fixtures.AssetPromptPipelineTests.submit_video
    evidence = staticmethod(fixtures.AssetPromptPipelineTests.evidence)
    asset = fixtures.AssetPromptPipelineTests.asset
    approve_prompt = fixtures.AssetPromptPipelineTests.approve_prompt
    materialize_approved_visuals = fixtures.AssetPromptPipelineTests.materialize_approved_visuals
    add_voice = fixtures.AssetPromptPipelineTests.add_voice

    def generated_image(self):
        pipeline.run(self.root)
        self.approve_prompt(prompt="  第一版完整线稿提示词。\n完整结尾。  ")
        pipeline.run(self.root)
        return self.asset()

    def snapshot(self, identifier="hero"):
        return edit.editor_snapshot(self.root, identifier, self.asset(identifier)["version"])

    def submit(self, snapshot, prompt="  人工直接修改的完整文本。\n保留缩进和结尾。\n  ", **kwargs):
        return edit.edit_and_approve(self.root, snapshot["id"], snapshot["version"],
                                     snapshot["sha256"], prompt, **kwargs)

    def test_editor_read_preserves_state_and_prefers_exact_wire_prompt(self):
        old = self.generated_image()
        with core.transaction(self.root) as state:
            core._find(state, "asset", "hero")["prompt"] = "计划中的短摘要，不是实际生成提示词。"
        before = (self.root / "project.json").read_bytes()
        with patch.object(pipeline, "_ark", side_effect=AssertionError("read-only editor")), \
             patch.object(pipeline, "_brain", side_effect=AssertionError("read-only editor")):
            snapshot = self.snapshot()
            again = self.snapshot()
        self.assertEqual(snapshot, again)
        self.assertEqual((self.root / "project.json").read_bytes(), before)
        self.assertEqual(snapshot["prompt"], old["generation_request"]["prompt"])
        self.assertEqual(snapshot["prompt_source"], "generation_request")
        self.assertEqual(snapshot["target_version"], 2)
        self.assertTrue(snapshot["models"])
        self.assertEqual(snapshot["reference_manifest"]["images"], [])
        self.assertNotIn("configuration", snapshot)
        self.assertNotIn("source", snapshot)

    def test_submit_adopts_exact_text_and_selected_model_generates_once_without_new_gate(self):
        old = self.generated_image()
        path = core.safe_path(self.root, old["file"])
        original = path.read_bytes()
        snapshot = self.snapshot()
        exact = "  原样保留这份人工提示词。\n\n不增加其他模板内容。  "
        before_jobs = copy.deepcopy(core.load_project(self.root)["jobs"])
        selected = "doubao-seedream-4-5-251128"
        self.submit(snapshot, exact, model=selected)
        approved = gate.approved(self.root, "hero")
        self.assertEqual(approved["prompt"], exact)
        self.assertEqual(approved["original_prompt"], snapshot["prompt"])
        self.assertEqual(approved["target_version"], 2)
        self.assertEqual(core.load_project(self.root)["jobs"], before_jobs)
        self.assertEqual(self.asset()["version"], 1)
        with patch.object(gate, "prepare", side_effect=AssertionError("no second prompt gate")), \
             patch.object(pipeline, "_select_audition", side_effect=AssertionError("no GPT")):
            for _ in range(3):
                state = pipeline.run(self.root)
        self.assertEqual(self.images[-1]["prompt"], exact)
        self.assertEqual(self.images[-1]["config"]["image_model"], selected)
        self.assertEqual(len(self.images), 2)
        self.assertEqual(len(state["jobs"]), 2)
        self.assertEqual(self.asset()["version"], 2)
        self.assertEqual(self.asset()["status"], "candidate")
        self.assertEqual(path.read_bytes(), original)
        self.assertTrue(any(item["file"] == old["file"] for item in self.asset()["history"]))
        self.brain.assert_not_called()

    def test_duplicate_submit_cannot_authorize_two_versions(self):
        self.generated_image()
        snapshot = self.snapshot()
        self.submit(snapshot)
        before = core.load_project(self.root)
        with self.assertRaises(core.GateError):
            self.submit(snapshot)
        self.assertEqual(core.load_project(self.root), before)
        self.assertEqual(self.asset()["version"], 1)

    def test_stale_script_style_config_or_artifact_rejects_without_discarding_old_candidate(self):
        mutators = [
            lambda state: state["plan"].update(style="全新合成风格"),
            lambda state: state["settings"]["ark"].update(image_size="4K"),
            lambda state: core._find(state, "asset", "hero").update(prompt="变化后的计划摘要"),
        ]
        self.generated_image()
        for mutate in mutators:
            snapshot = self.snapshot()
            with core.transaction(self.root) as state:
                mutate(state)
            before = core.load_project(self.root)
            with self.assertRaises(core.GateError):
                self.submit(snapshot)
            self.assertEqual(core.load_project(self.root), before)
        snapshot = self.snapshot()
        (self.root / "source/novel.txt").write_text("合成不同原文。")
        with self.assertRaises(core.GateError):
            self.submit(snapshot)
        snapshot = self.snapshot()
        core.safe_path(self.root, self.asset()["file"]).write_bytes(b"SYNTHETIC ALTERED FILE")
        with self.assertRaises(core.GateError):
            self.submit(snapshot)
        self.assertEqual(self.asset()["status"], "candidate")

    def test_pause_busy_and_unknown_call_leave_original_asset_untouched(self):
        self.generated_image()
        snapshot = self.snapshot()
        core.set_paused(self.root, True)
        with self.assertRaises(core.GateError):
            self.submit(snapshot)
        core.set_paused(self.root, False)
        with core.run_lock(self.root), self.assertRaises(core.GateError):
            self.submit(snapshot)
        core.reserve(self.root, "synthetic-unknown", "llm", "synthetic", 0)
        core.update_job(self.root, "synthetic-unknown", state="uncertain", accounting="unknown")
        before = core.load_project(self.root)
        with self.assertRaises(core.GateError):
            self.submit(snapshot)
        self.assertEqual(core.load_project(self.root), before)

    def test_invalid_text_or_model_is_atomic(self):
        self.generated_image()
        snapshot = self.snapshot()
        before = core.load_project(self.root)
        for prompt, model in [("", None), ("  ", None), ("valid", "not-an-open-model")]:
            with self.assertRaises(core.GateError):
                self.submit(snapshot, prompt, model=model)
            self.assertEqual(core.load_project(self.root), before)

    def test_initial_prompt_and_approved_asset_do_not_enter_completed_asset_editor(self):
        pipeline.run(self.root)
        with self.assertRaises(core.GateError):
            self.snapshot()
        self.approve_prompt()
        pipeline.run(self.root)
        core.approve(self.root, "asset", "hero", 1)
        before = core.load_project(self.root)
        with self.assertRaises(core.GateError):
            self.snapshot()
        self.assertEqual(core.load_project(self.root), before)

    def test_existing_dependent_video_is_never_retired_by_direct_asset_edit(self):
        self.generated_image()
        snapshot = self.snapshot()
        with core.transaction(self.root) as state:
            shot = state["episodes"][0]["shots"][0]
            shot.update(version=1, status="accepted", file="media/synthetic-kept.mp4", sha256="synthetic")
        before = core.load_project(self.root)
        with self.assertRaisesRegex(core.GateError, "现有视频已保留"):
            self.submit(snapshot)
        self.assertEqual(core.load_project(self.root), before)

    def test_known_failure_direct_edit_retains_failed_job_and_authorizes_only_one_retry(self):
        pipeline.run(self.root)
        self.approve_prompt()
        self.provider.generate_image.side_effect = ProviderError("合成明确失败", uncertain=False, stage="submission")
        pipeline.run(self.root)
        self.assertEqual(self.asset()["status"], "generation_failed")
        old = self.asset()
        failed_job = copy.deepcopy(core.load_project(self.root)["jobs"][old["job_id"]])
        snapshot = self.snapshot()
        exact = "合成明确失败后的人工修正提示词。"
        self.submit(snapshot, exact)
        self.provider.generate_image.side_effect = self.generate_image
        for _ in range(3):
            state = pipeline.run(self.root)
        self.assertEqual(self.images[-1]["prompt"], exact)
        self.assertEqual(self.provider.generate_image.call_count, 2)
        self.assertEqual(self.asset()["version"], 2)
        self.assertEqual(state["jobs"][old["job_id"]], failed_job)
        self.assertEqual(self.asset()["failure_history"][0]["job_id"], old["job_id"])
        self.brain.assert_not_called()

    def test_unverified_failed_job_cannot_be_turned_into_new_consent(self):
        self.generated_image()
        with core.transaction(self.root) as state:
            asset = core._find(state, "asset", "hero")
            asset.update(status="generation_failed", generation_error={"message": "合成"})
            state["jobs"][asset["job_id"]].update(state="failed", accounting="unknown")
        before = core.load_project(self.root)
        with self.assertRaises(core.GateError):
            self.snapshot()
        self.assertEqual(core.load_project(self.root), before)

    def test_failed_replacement_does_not_relabel_previous_file_as_failed_version(self):
        original = self.generated_image()
        self.submit(self.snapshot(), "合成第二版完整提示词。")
        self.provider.generate_image.side_effect = ProviderError("合成第二版明确失败",
                                                                 uncertain=False, stage="submission")
        pipeline.run(self.root)
        self.assertEqual(self.asset()["version"], 2)
        self.assertEqual(self.asset()["file"], original["file"])
        self.submit(self.snapshot(), "合成第三版人工修正。")
        self.provider.generate_image.side_effect = self.generate_image
        pipeline.run(self.root)
        current = self.asset()
        self.assertEqual(current["version"], 3)
        matching = [item for item in current["history"] if item.get("file") == original["file"]]
        self.assertEqual(len(matching), 1)
        self.assertEqual(matching[0]["version"], 1)
        self.assertEqual(current["failure_history"][0]["version"], 2)
        self.assertEqual(self.images[-1]["prompt"], "合成第三版人工修正。")

    def test_voice_edit_uses_saved_complete_text_duration_references_without_gpt_selection(self):
        self.materialize_approved_visuals()
        identifier = self.add_voice()
        pipeline.run(self.root)
        self.approve_prompt(identifier, model=fixtures.VOICE_CHOICE)
        pipeline.run(self.root)
        old = self.asset(identifier)
        snapshot = self.snapshot(identifier)
        exact = "  人工完整试音提示词，中性语气。\n" + fixtures.LINE + "\n"
        self.submit(snapshot, exact, model=fixtures.VOICE_CHOICE)
        with patch.object(pipeline, "_select_audition", side_effect=AssertionError("must not reselect voice")), \
             patch.object(gate, "prepare", side_effect=AssertionError("must not reprepare prompt")):
            for _ in range(3):
                pipeline.run(self.root)
        self.assertEqual(len(self.videos), 2)
        self.assertEqual(self.videos[-1]["prompt"], exact)
        self.assertEqual(self.videos[-1]["duration"], old["generation_request"]["duration"])
        self.assertEqual(self.videos[-1]["references"], self.videos[0]["references"])
        self.assertEqual(self.asset(identifier)["status"], "candidate")
        self.assertEqual(self.asset(identifier)["version"], 2)
        self.assertEqual(self.asset(identifier)["history"][0]["audio_file"], old["audio_file"])
        self.brain.assert_not_called()

    def test_legacy_image_uses_labeled_deterministic_template_and_never_calls_gpt(self):
        self.generated_image()
        with core.transaction(self.root) as state:
            asset = core._find(state, "asset", "hero")
            asset.pop("generation_request")
            asset.pop("generation_prompt")
            state["jobs"][asset["job_id"]].pop("asset_generation")
        before = (self.root / "project.json").read_bytes()
        snapshot = self.snapshot()
        self.assertEqual(snapshot["prompt_source"], "legacy_template")
        self.assertIn("整体画风", snapshot["prompt"])
        self.assertIn("未保存", snapshot["source_notice"])
        self.assertEqual((self.root / "project.json").read_bytes(), before)
        exact = snapshot["prompt"] + "\n人工修改这份模板。"
        self.submit(snapshot, exact)
        pipeline.run(self.root)
        self.assertEqual(self.images[-1]["prompt"], exact)
        self.brain.assert_not_called()

    def test_legacy_voice_uses_original_top_level_job_duration_before_estimate(self):
        self.materialize_approved_visuals()
        identifier = self.add_voice()
        pipeline.run(self.root)
        self.approve_prompt(identifier, model=fixtures.VOICE_CHOICE)
        pipeline.run(self.root)
        with core.transaction(self.root) as state:
            asset = core._find(state, "asset", identifier)
            asset.pop("generation_request")
            asset.pop("generation_prompt")
            job = state["jobs"][asset["job_id"]]
            job.pop("asset_generation")
            job["duration"] = 12
            # Top-level duration is the pipeline's actual saved request.
            # This deliberately contradictory legacy location must not win.
            job["video_request"]["duration"] = 4
        before = (self.root / "project.json").read_bytes()
        snapshot = self.snapshot(identifier)
        self.assertEqual(snapshot["prompt_source"], "audition_prompt")
        self.assertEqual(snapshot["duration"], 12)
        self.assertEqual((self.root / "project.json").read_bytes(), before)
        exact = snapshot["prompt"] + "\n人工修改合成试听要求。"
        self.submit(snapshot, exact, model=fixtures.VOICE_CHOICE)
        with patch.object(pipeline, "_select_audition", side_effect=AssertionError("no GPT selection")):
            pipeline.run(self.root)
        self.assertEqual(self.videos[-1]["duration"], 12)
        self.assertEqual(self.videos[-1]["prompt"], exact)
        self.brain.assert_not_called()


if __name__ == "__main__":
    unittest.main()
