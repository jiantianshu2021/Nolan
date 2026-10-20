"""Synthetic, offline exact asset consent fixtures; never credentials/providers."""
import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from norlan import asset_inputs, asset_models, asset_prompt_gate as gate, core, pipeline


class AssetPromptGateTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        core.create_project(self.root, "合成测试剧本：林舟说，设备已经检查完毕，现在可以按原来的计划继续操作。")
        core.install_plan(self.root, pipeline.demo_plan())
        core.approve(self.root, "plan", "plan", 1)
        core.set_remaining_asset_inputs_gpt(self.root, 1)

    def state(self):
        return core.load_project(self.root)

    def asset(self, identifier="hero"):
        return core._find(self.state(), "asset", identifier)

    def prepare(self, identifier="hero", prompt="合成最终资产提示词。", refs=None, duration=None):
        return gate.prepare(self.root, identifier, prompt, [] if refs is None else refs, duration)

    def confirm(self, candidate, identifier="hero", **kwargs):
        return gate.approve(self.root, identifier, candidate["id"], candidate["request_hash"], **kwargs)

    def visual(self):
        path = self.root / "media/assets/hero-synthetic.png"
        path.parent.mkdir(exist_ok=True)
        path.write_bytes(b"SYNTHETIC IMAGE FIXTURE")
        with core.transaction(self.root) as state:
            hero = core._find(state, "asset", "hero")
            hero.update(status="approved", version=1, file="media/assets/hero-synthetic.png", sha256=core.digest(path))
        return path

    def voice(self, offscreen=False):
        self.visual()
        with core.transaction(self.root) as state:
            character = "hero"
            if offscreen:
                source = {"id": "system_source", "kind": "character", "name": "合成系统声源",
                          "prompt": "仅画外声源。", "voice_only": True, "version": 0, "status": "planned"}
                core._register_voice_source(source)
                state["assets"].append(source)
                character = source["id"]
            state["assets"].append({"id": "voice", "kind": "voice", "name": "合成测试声线",
                "prompt": "成年男性，中音区，清晰自然。", "character_id": character,
                "version": 0, "status": "planned", "feedback": "", "history": [],
                "creation_input": {**asset_inputs.empty_input(), "mode": "gpt", "confirmed": True},
                "audition_text": "设备已经检查完毕，现在可以按原来的计划继续操作。",
                "audition_speaker": "林舟", "audition_source": "林舟说，设备已经检查完毕，现在可以按原来的计划继续操作。",
                "audition_reason": "合成原文中稳定的完整陈述。"})

    def test_prepare_is_local_idempotent_and_never_approves_or_increments_version(self):
        before = self.state()
        with patch("norlan.credentials.resolve_ark_key", side_effect=AssertionError("no key access")), \
             patch("norlan.providers.ArkProvider", side_effect=AssertionError("no provider")):
            candidate = self.prepare()
            again = self.prepare(prompt="重复继续不应覆写已准备文本。")
        self.assertEqual(candidate, again)
        state = self.state()
        self.assertEqual(state["status"], "awaiting_asset_prompt_approval")
        self.assertEqual(self.asset()["status"], "awaiting_asset_prompt_approval")
        self.assertEqual(self.asset()["version"], 0)
        self.assertEqual(candidate["target_version"], 1)
        self.assertEqual(candidate["sha256"], candidate["request_hash"])
        self.assertEqual(state["jobs"], before["jobs"])
        self.assertEqual(state["budget"], before["budget"])
        self.assertEqual(len(state["events"]), len(before["events"]) + 1)
        self.assertIsNone(gate.approved(self.root, "hero"))

    def test_confirm_binds_model_without_changing_defaults_and_only_one_version(self):
        candidate = self.prepare()
        defaults = copy.deepcopy(self.state()["settings"])
        model = "doubao-seedream-4-0-250828"
        self.confirm(candidate, model=model)
        approved = gate.approved(self.root, "hero")
        self.assertEqual(approved["prompt"], candidate["prompt"])
        self.assertEqual(approved["model"], model)
        self.assertEqual(approved["generation_config"]["image_model"], model)
        self.assertEqual(self.state()["settings"], defaults)
        self.assertEqual(self.asset()["version"], 0)
        with self.assertRaises(core.GateError):
            self.confirm(candidate, model=model)
        with core.transaction(self.root) as state:
            asset = core._find(state, "asset", "hero")
            asset.update(version=1, status="generating", generation_request=approved)
        self.assertIsNone(gate.approved(self.root, "hero"))
        self.assertEqual(self.state()["jobs"], {})

    def test_manual_exact_text_survives_repeated_prepare_without_gpt_rewrite(self):
        candidate = self.prepare()
        edited = "  用户原样文本\n不要附加任何旧提示词。  "
        self.confirm(candidate, prompt=edited)
        approved = gate.approved(self.root, "hero")
        self.assertEqual(approved["prompt"], edited)
        self.assertTrue(approved["manual_prompt"])
        self.assertEqual(self.prepare(prompt="旧自动提示词。"), approved)
        self.assertEqual(self.asset()["generation_prompt_history"][0]["prompt"], candidate["prompt"])
        self.assertEqual(self.asset()["generation_prompt_approval"]["source"], "human_manual_edit")

    def test_stale_id_hash_pause_and_unsupported_model_reject_without_mutation(self):
        candidate = self.prepare()
        before = self.state()
        mutations = [lambda: gate.approve(self.root, "hero", "stale", candidate["sha256"]),
                     lambda: gate.approve(self.root, "hero", candidate["id"], "0" * 64),
                     lambda: self.confirm(candidate, model="unknown-unconfigured-model"),
                     lambda: self.confirm(candidate, prompt="  ")]
        for mutate in mutations:
            with self.assertRaises(core.GateError):
                mutate()
            self.assertEqual(self.state(), before)
        core.set_paused(self.root, True)
        with self.assertRaises(core.GateError):
            self.confirm(candidate)
        self.assertEqual(self.state()["jobs"], {})

    def test_new_input_feedback_and_provider_configuration_require_fresh_candidate(self):
        for change in (lambda asset, state: asset.update(feedback="本次合成返修意见"),
                       lambda asset, state: asset["creation_input"].update(revision=9),
                       lambda asset, state: state["plan"].update(style="不同的合成画风"),
                       lambda asset, state: state["settings"]["ark"].update(image_size="1K")):
            candidate = self.prepare()
            self.confirm(candidate)
            with core.transaction(self.root) as state:
                change(core._find(state, "asset", "hero"), state)
            self.assertIsNone(gate.approved(self.root, "hero"))
            self.assertIsNone(gate.current(self.root, "hero"))
            updated = self.prepare(prompt="变更后的合成提示词。")
            self.assertNotEqual(updated["id"], candidate["id"])
            self.assertEqual(updated["status"], "candidate")
            self.assertNotIn("generation_prompt_approval", self.asset())

    def test_ref_manifest_contains_actual_ordered_hashes_and_rejects_missing_or_changed_files(self):
        directory = self.root / "inputs/assets/hero"
        directory.mkdir(parents=True)
        paths = [directory / "a.png", directory / "b.png"]
        for index, path in enumerate(paths):
            path.write_bytes(b"SYNTHETIC REF " + str(index).encode())
        with core.transaction(self.root) as state:
            asset = core._find(state, "asset", "hero")
            asset["creation_input"] = {**asset_inputs.empty_input(), "mode": "custom", "confirmed": True,
                "description": "合成参考图。", "references": [{"id": str(index), "mime": "image/png",
                    "file": str(path.relative_to(self.root)), "sha256": core.digest(path), "bytes": path.stat().st_size}
                   for index, path in enumerate(paths)]}
        candidate = self.prepare(refs=paths)
        self.assertEqual(candidate["reference_manifest"]["images"], [str(path.relative_to(self.root)) for path in paths])
        self.confirm(candidate)
        paths[0].write_bytes(b"SYNTHETIC CHANGED REF")
        with self.assertRaises(core.GateError):
            gate.approved(self.root, "hero")
        self.assertEqual(self.asset()["generation_prompt"]["prompt"], candidate["prompt"])

    def test_no_reference_omission_injection_or_external_path(self):
        path = self.root / "media/reference.png"
        path.write_bytes(b"SYNTHETIC UNAPPROVED REFERENCE")
        for references in ([path], ["missing.png"], ["../outside.png"]):
            with self.assertRaises(core.GateError):
                self.prepare(refs=references)
        self.assertNotIn("generation_prompt", self.asset())

    def test_approved_and_existing_candidate_media_are_never_retroactively_gated(self):
        self.visual()
        before = copy.deepcopy(self.asset())
        with self.assertRaises(core.GateError):
            self.prepare()
        self.assertEqual(self.asset(), before)
        with core.transaction(self.root) as state:
            core._find(state, "asset", "hero")["status"] = "candidate"
        with self.assertRaises(core.GateError):
            self.prepare()

    def test_rework_prepares_new_version_preserving_original_media(self):
        path = self.visual()
        core.reject(self.root, "asset", "hero", 1, "换成灰色工作服。")
        candidate = self.prepare(prompt="合成返修后的完整提示词。")
        self.assertEqual(candidate["target_version"], 2)
        self.assertEqual(self.asset()["version"], 1)
        self.assertEqual(path.read_bytes(), b"SYNTHETIC IMAGE FIXTURE")
        self.confirm(candidate)
        self.assertEqual(gate.approved(self.root, "hero")["target_version"], 2)

    def test_submitted_download_pending_completed_and_unknown_jobs_cannot_new_submit(self):
        for status in ("submitted", "download_pending", "completed", "uncertain", "submitting"):
            with core.transaction(self.root) as state:
                asset = core._find(state, "asset", "hero")
                asset.update(status="generating", version=1, job_id="synthetic-job")
                state["jobs"]["synthetic-job"] = {"id": "synthetic-job", "kind": "image", "item_id": "hero",
                    "state": status, "accounting": "spent", "amount": 0, "task_id": "synthetic-existing-task"}
            before = self.state()
            with self.assertRaises(core.GateError):
                self.prepare()
            self.assertEqual(self.state(), before)

    def test_voice_binds_actual_approved_character_and_duration(self):
        path = self.root / "media/assets/hero-synthetic.png"
        self.voice()
        candidate = self.prepare("voice", "合成完整试音提示词。", [path], 8)
        self.assertEqual(candidate["duration"], 8)
        self.assertEqual(candidate["reference_manifest"]["images"], ["media/assets/hero-synthetic.png"])
        self.assertEqual(candidate["source_manifest"]["character"]["sha256"], core.digest(path))
        self.confirm(candidate, "voice", model="doubao-seedance-2-0-fast-260128")
        approved = gate.approved(self.root, "voice")
        self.assertTrue(approved["generation_config"]["generate_audio"])
        self.assertEqual(approved["duration"], 8)
        with core.transaction(self.root) as state:
            core._find(state, "asset", "hero")["version"] = 2
        self.assertIsNone(gate.approved(self.root, "voice"))

    def test_offscreen_role_is_hidden_but_real_voice_always_requires_consent(self):
        self.voice(offscreen=True)
        with self.assertRaises(core.GateError):
            self.prepare("system_source")
        candidate = self.prepare("voice", "合成画外试听，深灰静止画面，不出现人物。", [], 8)
        self.assertEqual(candidate["reference_manifest"]["images"], [])
        self.assertIsNone(gate.approved(self.root, "voice"))
        self.confirm(candidate, "voice")
        self.assertIsNotNone(gate.approved(self.root, "voice"))

    def test_tampered_duration_prompt_or_config_cannot_consume_consent(self):
        candidate = self.prepare()
        self.confirm(candidate)
        with core.transaction(self.root) as state:
            core._find(state, "asset", "hero")["generation_prompt"]["generation_config"]["image_size"] = "4K"
        self.assertIsNone(gate.approved(self.root, "hero"))
        with core.transaction(self.root) as state:
            prompt = core._find(state, "asset", "hero")["generation_prompt"]
            prompt["generation_config"]["image_size"] = "2K"
            prompt["prompt"] = "篡改后的合成提示词。"
        self.assertIsNone(gate.approved(self.root, "hero"))

    def test_added_episode_asset_binds_its_actual_source_instead_of_first_episode(self):
        source = self.root / "source/ep02.txt"
        source.write_text("合成第二集：新的场景和角色。", encoding="utf-8")
        with core.transaction(self.root) as state:
            core._find(state, "asset", "hero").update(source_file="source/ep02.txt",
                source_sha256=core.digest(source), source_episode_id="ep02")
        candidate = self.prepare()
        self.assertEqual(candidate["source_manifest"]["source"],
                         {"file": "source/ep02.txt", "sha256": core.digest(source)})
        self.confirm(candidate)
        source.write_text("合成第二集已被替换。", encoding="utf-8")
        with self.assertRaisesRegex(core.GateError, "分集原文校验"):
            gate.approved(self.root, "hero")

    def test_missing_human_record_requires_fresh_gate_instead_of_accepting_approved_status(self):
        candidate = self.prepare()
        self.confirm(candidate)
        with core.transaction(self.root) as state:
            core._find(state, "asset", "hero").pop("generation_prompt_approval")
        self.assertIsNone(gate.approved(self.root, "hero"))
        self.assertIsNone(gate.current(self.root, "hero"))
        replacement = self.prepare()
        self.assertEqual(replacement["status"], "candidate")
        self.assertNotEqual(replacement["id"], candidate["id"])
        self.assertEqual(self.asset()["status"], "awaiting_asset_prompt_approval")


if __name__ == "__main__":
    unittest.main()
