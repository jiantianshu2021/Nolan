"""Synthetic offline internal voice-source registration and real voice gates."""
import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from norlan import core, pipeline
from test_pipeline import choose_gpt_assets, fake_media, run_confirming_asset_prompts, run_with_shot_approvals, write


class VoiceSourceRoleTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.enterContext(fake_media())
        self.line = "当前系统已经完成初始化，请按照操作说明查看本次任务信息。"
        self.source = "系统：" + self.line
        core.create_project(self.root, self.source)
        self.source_definition = {"id": "system-source", "kind": "character", "name": "画外系统声源",
                                  "prompt": "仅作为系统播报声源，不出现在画面中", "voice_only": True}
        self.voice_definition = {"id": "system-voice", "kind": "voice", "name": "系统女声", "prompt": "成年女声，中音，清澈",
            "character_id": "system-source", "audition_speaker": "系统", "audition_text": self.line,
            "audition_source": self.source, "audition_reason": "系统原文中的完整中性陈述，适合辨认音色"}

    def install(self, source=True):
        plan = pipeline.demo_plan()
        if source:
            plan["assets"] += [copy.deepcopy(self.source_definition), copy.deepcopy(self.voice_definition)]
            plan["episodes"][0]["shots"][0]["asset_ids"] += ["system-source", "system-voice"]
        core.install_plan(self.root, plan)
        core.approve(self.root, "plan", "plan", 1)
        return core.load_project(self.root)

    def materialize_visible_assets(self):
        choose_gpt_assets(self.root)
        pipeline.run(self.root)
        for asset in core.load_project(self.root)["assets"]:
            if asset["kind"] != "voice" and not core.is_internal_voice_source(asset):
                core.approve(self.root, "asset", asset["id"], asset["version"])

    def legacy_source(self, status="candidate", **updates):
        with core.transaction(self.root) as state:
            asset = core._find(state, "asset", "system-source")
            asset.pop("internal_role", None)
            asset.pop("internal_role_history", None)
            asset.update(status=status, **updates)

    def test_initial_role_is_internal_without_human_approval_or_portrait(self):
        state = self.install()
        source = core._find(state, "asset", "system-source")
        self.assertEqual(source["status"], "internal")
        self.assertIs(source["internal_role"], True)
        self.assertTrue(core.asset_ready(source))
        self.assertEqual(source["sha256"], core.voice_only_digest(source))
        self.assertFalse({"file", "approved_at", "approved_sha256", "human_decision"}.intersection(source))
        self.assertFalse(any(event.get("action") == "human_approved" and event.get("id") == source["id"] for event in state["events"]))
        self.assertFalse(core.asset_ready(core._find(state, "asset", "system-voice")))

    def test_source_has_no_manual_asset_input_or_approval_action(self):
        self.install()
        for operation in (
            lambda: core.approve(self.root, "asset", "system-source", 1),
            lambda: core.reject(self.root, "asset", "system-source", 1, "无须独立审批声源"),
            lambda: core.set_asset_input(self.root, "system-source", 1, 0, "gpt", ""),
        ):
            with self.assertRaises(core.GateError):
                operation()
        with core.transaction(self.root) as state:
            role = core._find(state, "asset", "system-source")
            role.update(status="planned", version=0)
        state = core.set_remaining_asset_inputs_gpt(self.root, 1)
        self.assertFalse(core._find(state, "asset", "system-source")["creation_input"]["confirmed"])
        self.assertTrue(core._find(state, "asset", "system-voice")["creation_input"]["confirmed"])

    def test_only_voice_is_auditioned_and_reviewed_before_segment_prompt(self):
        self.install()
        self.materialize_visible_assets()
        with core.transaction(self.root) as state:
            state["mode"] = "live"
        core.configure(self.root, {"caps": {"video_second": 1}}, limit=100)
        api = Mock()
        api.submit_video.return_value = {"id": "synthetic-voice-id"}
        api.poll_video.return_value = {"status": "completed", "url": "https://example.test/synthetic.mp4"}
        api.download.side_effect = lambda _url, output: write(output, b"SYNTHETIC VOICE")
        with patch.object(pipeline, "_ark", return_value=api), patch.object(pipeline, "_brain", return_value={
                "prompt": "合成画面，由系统女声播报。", "dialogue": self.line, "continuity_notes": "合成说明"}):
            state = pipeline.run(self.root)
            self.assertEqual(state["status"], "awaiting_asset_prompt_approval")
            api.submit_video.assert_not_called()
            state = run_confirming_asset_prompts(self.root)
            self.assertEqual(state["status"], "awaiting_assets")
            self.assertEqual(core._find(state, "asset", "system-voice")["status"], "candidate")
            self.assertEqual(core._find(state, "asset", "system-source")["status"], "internal")
            api.submit_video.assert_called_once()
            self.assertEqual(api.submit_video.call_args.args[2], [])
            self.assertFalse(any(job["kind"] == "video" for job in state["jobs"].values()))
            core.approve(self.root, "asset", "system-voice", 1)
            self.assertEqual(pipeline.run(self.root)["status"], "awaiting_prompt_approval")
            api.submit_video.assert_called_once()
            api.generate_image.assert_not_called()
            state = core.load_project(self.root)
            candidate = state["episodes"][0]["shots"][0]["prompt_candidate"]
            self.assertEqual([item["asset_id"] for item in candidate["reference_manifest"]["context"]["image_references"]], ["hero", "station"])
            self.assertEqual(candidate["reference_manifest"]["context"]["audio_references"][0]["asset_id"], "system-voice")

    def test_episode_approval_accepts_internal_source_but_still_uses_approved_voice(self):
        self.install()
        self.materialize_visible_assets()
        state = pipeline.run(self.root)
        voice = core._find(state, "asset", "system-voice")
        core.approve(self.root, "asset", voice["id"], voice["version"])
        state = run_with_shot_approvals(self.root)
        episode = state["episodes"][0]
        self.assertEqual(episode["status"], "awaiting_review")
        core.approve(self.root, "episode", episode["id"], episode["version"])
        state = core.load_project(self.root)
        self.assertEqual(state["episodes"][0]["status"], "approved")
        self.assertEqual(core._find(state, "asset", "system-source")["status"], "internal")

    def test_migration_invalidates_only_unsubmitted_source_bound_prompt_consent(self):
        self.install()
        self.legacy_source("candidate")
        with core.transaction(self.root) as state:
            first, second = state["episodes"][0]["shots"]
            first.update(status="prompt_approved", version=0, prompt_candidate={"id": "source-bound", "status": "approved"},
                         prompt_approval={"candidate_id": "source-bound"})
            second.update(status="awaiting_prompt_approval", version=0, prompt_candidate={"id": "unrelated", "status": "candidate"})
            state["status"] = "awaiting_prompt_approval"
        before = core.load_project(self.root)
        state = core.normalize_voice_sources(self.root)
        first, second = state["episodes"][0]["shots"]
        self.assertEqual(first["status"], "planned")
        self.assertFalse({"prompt_candidate", "prompt_approval"}.intersection(first))
        self.assertEqual(first["prompt_history"][-1]["status"], "invalidated")
        self.assertEqual(second, before["episodes"][0]["shots"][1])
        self.assertEqual(state["jobs"], before["jobs"])

    def test_migration_keeps_submitted_video_task_and_its_recovery_record(self):
        self.install()
        self.legacy_source("candidate")
        with core.transaction(self.root) as state:
            shot = state["episodes"][0]["shots"][0]
            shot.update(status="prompt_approved", version=1, job_id="submitted-video", prompt_candidate={"id": "already-sent", "status": "approved"},
                        prompt_approval={"candidate_id": "already-sent"})
            state["jobs"]["submitted-video"] = {"id": "submitted-video", "kind": "video", "item_id": shot["id"],
                "state": "submitted", "accounting": "reserved", "amount": 1, "task_id": "existing-video-task"}
        before = core.load_project(self.root)
        state = core.normalize_voice_sources(self.root)
        self.assertEqual(state["episodes"], before["episodes"])
        self.assertEqual(state["jobs"], before["jobs"])

    def test_legacy_candidate_and_approved_sources_migrate_without_touching_real_assets(self):
        for status in ("candidate", "approved"):
            with self.subTest(status=status):
                if not (self.root / "plans/plan-v1.json").exists():
                    self.install()
                self.legacy_source(status, approved_at="legacy-time", approved_sha256="legacy-audit-hash")
                before = core.load_project(self.root)
                after = core.normalize_voice_sources(self.root)
                role = core._find(after, "asset", "system-source")
                self.assertTrue(core.asset_ready(role))
                self.assertEqual(role["status"], "internal")
                self.assertNotIn("approved_at", role)
                self.assertNotIn("approved_sha256", role)
                self.assertEqual(role["internal_role_history"][-1]["approved_at"], "legacy-time")
                self.assertEqual(role["internal_role_history"][-1]["status"], status)
                for original in before["assets"]:
                    if original["id"] != "system-source":
                        self.assertEqual(core._find(after, "asset", original["id"]), original)
                for key in ("episodes", "jobs", "budget"):
                    self.assertEqual(after[key], before[key])
                second = core.normalize_voice_sources(self.root)
                self.assertEqual(second["assets"], after["assets"])
                self.assertEqual(second["events"], after["events"])

    def test_visible_character_is_never_registered_or_automatically_approved(self):
        state = self.install()
        original = copy.deepcopy(core._find(state, "asset", "hero"))
        with core.transaction(self.root) as state:
            core._find(state, "asset", "hero")["internal_role"] = True
        after = core.normalize_voice_sources(self.root)
        hero = core._find(after, "asset", "hero")
        self.assertFalse(core.is_internal_voice_source(hero))
        self.assertFalse(core.asset_ready(hero))
        self.assertEqual(hero["status"], original["status"])
        self.assertEqual(hero["version"], original["version"])
        self.assertNotIn("approved_at", hero)

    def test_changed_definition_hash_is_not_silently_trusted(self):
        self.install()
        self.legacy_source("candidate")
        with core.transaction(self.root) as state:
            core._find(state, "asset", "system-source")["prompt"] += "未登记变动"
        before = (self.root / "project.json").read_bytes()
        with self.assertRaises(core.GateError):
            core.normalize_voice_sources(self.root)
        self.assertEqual((self.root / "project.json").read_bytes(), before)

    def test_unknown_or_source_own_inflight_jobs_prevent_semantic_migration(self):
        self.install()
        self.legacy_source("candidate")
        for status, accounting in (("uncertain", "unknown"), ("submitted", "reserved")):
            with self.subTest(status=status):
                with core.transaction(self.root) as state:
                    state["jobs"]["synthetic-pending"] = {"id": "synthetic-pending", "kind": "voice", "item_id": "system-voice" if status == "uncertain" else "system-source",
                        "state": status, "accounting": accounting, "amount": 1}
                before = (self.root / "project.json").read_bytes()
                with self.assertRaises(core.GateError):
                    core.normalize_voice_sources(self.root)
                self.assertEqual((self.root / "project.json").read_bytes(), before)

    def test_legacy_source_migration_allows_recovery_of_known_submitted_voice_without_post(self):
        self.install()
        self.materialize_visible_assets()
        self.legacy_source("candidate")
        core.configure(self.root, {"caps": {"video_second": 1}}, limit=100)
        with core.transaction(self.root) as state:
            state["mode"] = "live"
            core._find(state, "asset", "system-voice").update(status="generating", version=1, job_id="existing-voice-job")
        core.reserve(self.root, "existing-voice-job", "voice", "system-voice", 5)
        core.update_job(self.root, "existing-voice-job", state="submitted", task_id="existing-provider-task")
        api = Mock()
        api.poll_video.return_value = {"status": "completed", "url": "https://example.test/synthetic.mp4"}
        api.download.side_effect = lambda _url, output: write(output, b"SYNTHETIC EXISTING VOICE")
        with patch.object(pipeline, "_ark", return_value=api), patch.object(pipeline, "_brain", side_effect=AssertionError("No model call during recovery")):
            state = pipeline.run(self.root)
        self.assertEqual(state["status"], "awaiting_assets")
        self.assertEqual(core._find(state, "asset", "system-source")["status"], "internal")
        self.assertEqual(core._find(state, "asset", "system-voice")["status"], "candidate")
        self.assertEqual(state["jobs"]["existing-voice-job"]["task_id"], "existing-provider-task")
        api.poll_video.assert_called_once_with("existing-provider-task")
        api.submit_video.assert_not_called()
        api.generate_image.assert_not_called()
        api.preflight.assert_not_called()

    def test_pending_amendment_baseline_survives_only_the_semantic_migration(self):
        self.install()
        self.legacy_source("approved", approved_at="legacy-time", approved_sha256=core.voice_only_digest(self.source_definition))
        target = core.load_project(self.root)["episodes"][0]["shots"][1]
        change = [{"id": target["id"], "add_asset_ids": ["system-source", "system-voice"]}]
        state = core.propose_asset_amendment(self.root, target["id"], 0, "复用女声", [], change)
        amendment = state["asset_amendments"][-1]
        old_hash = amendment["baseline_assets_sha256"]
        state = core.normalize_voice_sources(self.root)
        current = state["asset_amendments"][-1]
        self.assertNotEqual(current["baseline_assets_sha256"], old_hash)
        core.apply_asset_amendment(self.root, current["id"], "approve")
        self.assertFalse(core.activate_asset_amendment(self.root, current["id"]))
        self.assertNotEqual(core._find(core.load_project(self.root), "asset", "system-voice")["status"], "approved")

    def test_migration_does_not_refresh_already_stale_real_asset_baseline(self):
        self.install()
        self.legacy_source("candidate")
        target = core.load_project(self.root)["episodes"][0]["shots"][1]
        state = core.propose_asset_amendment(self.root, target["id"], 0, "复用女声", [],
            [{"id": target["id"], "add_asset_ids": ["system-source", "system-voice"]}])
        amendment = state["asset_amendments"][-1]
        with core.transaction(self.root) as state:
            core._find(state, "asset", "hero")["prompt"] += "另有真实角色修改"
        state = core.normalize_voice_sources(self.root)
        self.assertEqual(state["asset_amendments"][-1]["baseline_assets_sha256"], amendment["baseline_assets_sha256"])
        with self.assertRaises(core.GateError):
            core.apply_asset_amendment(self.root, amendment["id"], "approve")

    def test_dynamic_source_does_not_need_approval_but_new_voice_still_blocks_activation(self):
        self.install(source=False)
        self.materialize_visible_assets()
        before = core.load_project(self.root)
        target = before["episodes"][0]["shots"][0]
        state = core.propose_asset_amendment(self.root, target["id"], 0, "新增系统女声", [self.source_definition, self.voice_definition],
            [{"id": target["id"], "add_asset_ids": ["system-source", "system-voice"]}])
        amendment = state["asset_amendments"][-1]
        state = core.apply_asset_amendment(self.root, amendment["id"], "approve")
        self.assertTrue(core.asset_ready(core._find(state, "asset", "system-source")))
        self.assertFalse(core.activate_asset_amendment(self.root, amendment["id"]))
        choose_gpt_assets(self.root)
        state = pipeline.run(self.root)
        voice = core._find(state, "asset", "system-voice")
        self.assertEqual(voice["status"], "candidate")
        self.assertFalse(core.activate_asset_amendment(self.root, amendment["id"]))
        core.approve(self.root, "asset", "system-voice", voice["version"])
        self.assertTrue(core.activate_asset_amendment(self.root, amendment["id"]))
        state = core.load_project(self.root)
        self.assertEqual(state["assets"][:len(before["assets"])], before["assets"])
        self.assertFalse(any(event.get("action") == "human_approved" and event.get("id") == "system-source" for event in state["events"]))


if __name__ == "__main__":
    unittest.main()
