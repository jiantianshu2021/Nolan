"""Synthetic never-submitted legacy asset fixtures; no paid services or keys."""
import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from norlan import asset_preparation_recovery as recovery, core, pipeline


class AssetPreparationRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        core.create_project(self.root, "合成测试原文：设备已经检查完毕。", mode="live")
        core.install_plan(self.root, pipeline.demo_plan())
        core.approve(self.root, "plan", "plan", 1)
        core.set_remaining_asset_inputs_gpt(self.root, 1)
        core.reserve(self.root, "synthetic-plan", "llm", "plan", 0)
        core.update_job(self.root, "synthetic-plan", state="completed", accounting="spent")
        with core.transaction(self.root) as state:
            asset = core._find(state, "asset", "hero")
            asset.update(status="generating", version=1, feedback="合成用户原始意见。")
            state.update(status="blocked", last_error="合成本机预算拒绝。")
        self.before = self.state()

    def state(self):
        return core.load_project(self.root)

    def asset(self):
        return core._find(self.state(), "asset", "hero")

    def restore(self):
        with core.transaction(self.root) as state:
            state.clear()
            state.update(copy.deepcopy(self.before))

    def test_never_approved_budget_restores_only_local_preparation_and_audits_proof(self):
        with patch("norlan.providers.ArkProvider", side_effect=AssertionError("no provider")), \
             patch("norlan.credentials.resolve_ark_key", side_effect=AssertionError("no key")), \
             patch.object(core, "reserve", side_effect=AssertionError("no reservation")):
            self.assertEqual(recovery.recover(self.root), ["hero"])
        state, asset = self.state(), self.asset()
        self.assertEqual(asset["status"], "planned")
        self.assertEqual(asset["version"], 0)
        for field in ("prompt", "feedback", "creation_input", "history"):
            self.assertEqual(asset.get(field), core._find(self.before, "asset", "hero").get(field))
        for field in ("jobs", "budget", "last_error", "status", "episodes", "settings"):
            self.assertEqual(state[field], self.before[field])
        marker = asset["asset_preparation_recovery"]
        self.assertEqual((marker["original_status"], marker["original_version"]), ("generating", 1))
        self.assertIs(marker["proof"]["budget_approved"], False)
        self.assertEqual(marker["proof"]["media_ledger_jobs"], 0)
        self.assertTrue(marker["requires_human_prompt_confirmation"])
        self.assertNotIn("generation_prompt_approval", asset)
        self.assertEqual(state["events"][:-1], self.before["events"])
        self.assertEqual(state["events"][-1]["action"], "asset_preparation_recovered")
        for _ in range(3):
            self.assertEqual(recovery.recover(self.root), [])
        self.assertEqual(self.asset(), asset)
        self.assertEqual(self.state()["events"], state["events"])

    def test_budget_must_be_strictly_never_approved_and_zero(self):
        for updates in ({"approved": True}, {"approved": None}, {"approved": 0},
                        {"limit": 1}, {"limit": False}):
            with self.subTest(updates=updates):
                self.restore()
                with core.transaction(self.root) as state:
                    state["budget"].update(updates)
                self.assertEqual(recovery.recover(self.root), [])
                self.assertEqual(self.asset()["status"], "generating")
        self.restore()
        with core.transaction(self.root) as state:
            state["budget"].pop("approved")
        self.assertEqual(recovery.recover(self.root), [])

    def test_any_media_job_anywhere_prevents_recovery_even_without_remote_id(self):
        for kind in ("image", "voice", "video", "music", "unrecognized"):
            with self.subTest(kind=kind):
                self.restore()
                with core.transaction(self.root) as state:
                    state["jobs"]["synthetic-other-item"] = {"id": "synthetic-other-item", "kind": kind,
                        "item_id": "station", "state": "failed", "accounting": "spent", "amount": 0}
                before = self.state()
                self.assertEqual(recovery.recover(self.root), [])
                self.assertEqual(self.state()["jobs"], before["jobs"])
                self.assertEqual(self.asset()["version"], 1)

    def test_budget_history_or_media_reservation_blocks_even_if_job_is_missing(self):
        for row in ({"action": "configuration_updated", "budget_limit": 2000},
                    {"action": "call_reserved", "kind": "image", "id": "missing-job", "amount": 1},
                    {"action": "human_approved", "kind": "asset", "id": "hero", "version": 1},
                    {"action": "human_approved_asset_prompt", "id": "hero"}):
            with self.subTest(row=row):
                self.restore()
                with core.transaction(self.root) as state:
                    state["events"].append(row)
                self.assertEqual(recovery.recover(self.root), [])
                self.assertNotIn("asset_preparation_recovery", self.asset())

    def test_unknown_and_in_flight_llm_jobs_remain_blocking(self):
        for updates in ({"state": "uncertain", "accounting": "unknown"},
                        {"state": "submitted", "accounting": "reserved"}):
            with self.subTest(updates=updates):
                self.restore()
                with core.transaction(self.root) as state:
                    state["jobs"]["synthetic-plan"].update(updates)
                before = self.state()
                self.assertEqual(recovery.recover(self.root), [])
                self.assertEqual(self.state()["jobs"], before["jobs"])

    def test_any_asset_output_consent_attempt_or_remote_evidence_prevents_reset(self):
        cases = {"file": "media/old.png", "sha256": "synthetic-hash", "job_id": "missing-job",
                 "attempts": [{"job_id": "missing-job"}], "history": [{"version": 0}],
                 "evidence": {"frames": ["evidence/old.png"]}, "audio_file": "audio.wav",
                 "generation_prompt": {"id": "synthetic-prompt"},
                 "generation_request": {"prompt": "合成定稿"},
                 "generation_prompt_approval": {"decision": "generate"},
                 "generation_prompt_history": [{"id": "older-prompt"}],
                 "generation_model": "synthetic-model", "approval": {"version": 1},
                 "approved_at": "synthetic-timestamp", "request_id": "synthetic-request",
                 "task_id": "synthetic-task", "actual_cost": 0, "usage_cost": 0,
                 "generation_error": "合成结果未知，禁止重新提交。"}
        for field, value in cases.items():
            with self.subTest(field=field):
                self.restore()
                with core.transaction(self.root) as state:
                    core._find(state, "asset", "hero")[field] = value
                original = self.asset()
                self.assertEqual(recovery.recover(self.root), [])
                self.assertEqual(self.asset(), original)

    def test_versioned_file_directory_or_output_symlink_blocks_without_deletion(self):
        for relative, directory in (("media/assets/hero-v1.png", False),
                                    ("evidence/hero-v1", True),
                                    ("plans/hero-v2.json", False)):
            with self.subTest(relative=relative):
                path = self.root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.mkdir() if directory else path.write_bytes(b"SYNTHETIC MEDIA EVIDENCE")
                self.assertEqual(recovery.recover(self.root), [])
                self.assertTrue(path.exists())
                path.rmdir() if directory else path.unlink()
        outside = Path(self.temp.name).parent / (self.root.name + "-outside")
        outside.mkdir(exist_ok=True)
        self.addCleanup(outside.rmdir)
        link = self.root / "media/linked-output"
        link.symlink_to(outside, target_is_directory=True)
        self.assertEqual(recovery.recover(self.root), [])
        self.assertTrue(link.is_symlink())

    def test_only_first_generating_version_in_live_mode_is_eligible(self):
        for status, version in (("planned", 0), ("generating", 2), ("generating", True),
                                ("generation_failed", 1), ("candidate", 1), ("approved", 1)):
            with self.subTest(status=status, version=version):
                self.restore()
                with core.transaction(self.root) as state:
                    core._find(state, "asset", "hero").update(status=status, version=version)
                self.assertEqual(recovery.recover(self.root), [])
        self.restore()
        with core.transaction(self.root) as state:
            state["mode"] = "demo"
        self.assertEqual(recovery.recover(self.root), [])

    def test_caller_owned_run_lock_is_supported_without_acquiring_a_second_lock(self):
        with core.run_lock(self.root):
            self.assertEqual(recovery.recover(self.root, already_locked=True), ["hero"])

    def test_paused_project_stays_untouched_until_operator_resumes(self):
        core.set_paused(self.root, True)
        before = self.state()
        self.assertEqual(recovery.recover(self.root), [])
        for field in ("paused", "assets", "events", "jobs", "budget", "last_error", "status"):
            self.assertEqual(self.state()[field], before[field])

    def test_large_scan_is_bounded_and_fails_closed(self):
        with patch.object(recovery, "MAX_SCAN_ENTRIES", 2), \
             patch.object(recovery.os, "walk", return_value=[(str(self.root), [], ["one", "two", "three"])]):
            self.assertEqual(recovery.recover(self.root), [])
        self.assertEqual(self.asset()["version"], 1)

    def test_missing_job_is_not_sufficient_and_scan_errors_fail_closed(self):
        with patch.object(recovery, "_has_disk_evidence", return_value=True):
            self.assertEqual(recovery.recover(self.root), [])
        with patch.object(recovery.os, "walk", side_effect=OSError("synthetic unavailable directory")):
            self.assertEqual(recovery.recover(self.root), [])
        self.assertEqual(self.asset()["status"], "generating")


if __name__ == "__main__":
    unittest.main()
