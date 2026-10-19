"""Synthetic media only: version-bound human decisions and scoped rework."""
import copy
from pathlib import Path
import tempfile
import unittest

from norlan import core


class ReviewDecisionTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        core.create_project(self.root, "SYNTHETIC 审批测试，不调用生成服务")
        plan = {"title": "SYNTHETIC", "assets": [
            {"id": "hero", "kind": "character", "name": "测试角色", "prompt": "SYNTHETIC"}],
            "episodes": [{"id": f"e{index}", "title": "SYNTHETIC", "shots": [
                {"id": f"e{index}s{number}", "duration": 4, "prompt": "SYNTHETIC",
                 "scene_id": "b" if index == 1 and number == 3 else "a", "asset_ids": ["hero"]}
                for number in range(1, 5 if index == 1 else 3)]} for index in (1, 2)]}
        core.install_plan(self.root, plan)
        core.approve(self.root, "plan", "plan", 1)
        core.configure(self.root, limit=100)

    def artifact(self, path, body=None):
        full = self.root / path
        full.parent.mkdir(parents=True, exist_ok=True)
        full.write_bytes(body or ("SYNTHETIC " + path).encode())
        return {"file": path, "sha256": core.digest(full)}

    def shot(self, item_id="e1s2"):
        return core._find(core.load_project(self.root), "shot", item_id)

    def candidate(self, item_id="e1s2", version=1, verdict="fail"):
        artifact = self.artifact(f"media/{item_id}-v{version}.mp4")
        with core.transaction(self.root) as state:
            shot = core._find(state, "shot", item_id)
            shot.update(artifact, status="review_required", version=version,
                        review={"verdict": verdict, "issues": ["GPT 疑似服饰不一致"],
                                "sha256": artifact["sha256"], "job_id": "private-review-job"})
        return artifact

    def complete_project(self):
        state = core.load_project(self.root)
        for ep in state["episodes"]:
            for shot in ep["shots"]:
                self.candidate(shot["id"])
                core.approve(self.root, "shot", shot["id"], 1)
        hero = self.artifact("media/hero.png")
        with core.transaction(self.root) as state:
            state["assets"][0].update(hero, status="approved", version=1)
            for ep in state["episodes"]:
                ep.update(self.artifact(f"exports/{ep['id']}-v1.mp4"), status="approved", version=1,
                          approved_at=core.now(), render_manifest={"clips": [
                              {"id": shot["id"], "sha256": shot["sha256"]} for shot in ep["shots"]]},
                          verified_subtitles=[{"text": "旧字幕", "start": 0, "end": 1}],
                          subtitle_file=f"exports/{ep['id']}.srt", progress_stage={"stage": "rendering"},
                          progress_started_at=core.now())
                ep["approved_sha256"] = ep["sha256"]
            state["jobs"]["synthetic-paid"] = {"id": "synthetic-paid", "item_id": "e1s2", "kind": "video",
                                                  "amount": 2, "state": "completed", "accounting": "spent"}
        return core.load_project(self.root)

    def test_approval_keeps_report_and_records_exact_human_decision(self):
        for verdict in ("pass", "fail", "uncertain"):
            with self.subTest(verdict=verdict):
                self.candidate(verdict=verdict)
                before = core.load_project(self.root)
                core.approve(self.root, "shot", "e1s2", 1)
                shot = self.shot()
                self.assertTrue(core.human_approved_shot(shot))
                self.assertTrue(shot["review"]["human_override"])
                decision = shot["human_decision"]
                self.assertEqual(decision["decision"], "approved")
                self.assertEqual((decision["version"], decision["sha256"]), (1, shot["sha256"]))
                self.assertEqual(decision["report"], {"verdict": verdict, "issues": ["GPT 疑似服饰不一致"]})
                self.assertIn("未经用户逐条核实", decision["report_note"])
                self.assertNotIn("job_id", decision["report"])
                self.assertEqual(core.load_project(self.root)["jobs"], before["jobs"])
                with core.transaction(self.root) as state:
                    core._find(state, "shot", "e1s2")["review"]["issues"].append("后续检查")
                self.assertEqual(self.shot()["human_decisions"][-1]["report"]["issues"], ["GPT 疑似服饰不一致"])

    def test_accepted_status_or_unbound_legacy_override_is_not_human_approval(self):
        self.candidate()
        shot = self.shot()
        shot.update(status="accepted")
        self.assertFalse(core.human_approved_shot(shot))
        shot["review"]["human_override"] = True
        self.assertFalse(core.human_approved_shot(shot))
        shot.update(approved_at=core.now(), approved_sha256=shot["sha256"])
        self.assertTrue(core.human_approved_shot(shot))
        shot["approved_sha256"] = "a" * 64
        self.assertFalse(core.human_approved_shot(shot))

    def test_structured_decision_never_falls_back_to_legacy_when_stale(self):
        self.candidate()
        core.approve(self.root, "shot", "e1s2", 1)
        original = self.shot()
        for key, value in (("version", 2), ("version", True), ("sha256", "a" * 64),
                           ("decision", "revision_requested"), ("at", "")):
            with self.subTest(key=key, value=value):
                changed = copy.deepcopy(original)
                changed["human_decision"][key] = value
                self.assertFalse(core.human_approved_shot(changed))
        changed = copy.deepcopy(original)
        changed["approved_version"] = 2
        self.assertFalse(core.human_approved_shot(changed))

    def test_stale_version_or_tampered_media_blocks_both_decisions_atomically(self):
        artifact = self.candidate()
        for action in ("approve", "reject"):
            for version in (0, 2, True):
                before = core.load_project(self.root)
                with self.subTest(action=action, version=version), self.assertRaises(core.GateError):
                    if action == "approve":
                        core.approve(self.root, "shot", "e1s2", version)
                    else:
                        core.reject(self.root, "shot", "e1s2", version, "修改")
                self.assertEqual(core.load_project(self.root), before)
        (self.root / artifact["file"]).write_bytes(b"SYNTHETIC tampered")
        for action in ("approve", "reject"):
            before = core.load_project(self.root)
            with self.subTest(action=action), self.assertRaises(core.GateError):
                if action == "approve":
                    core.approve(self.root, "shot", "e1s2", 1)
                else:
                    core.reject(self.root, "shot", "e1s2", 1, "修改")
            self.assertEqual(core.load_project(self.root), before)

    def test_interrupted_new_version_cannot_approve_previous_version_file(self):
        self.candidate(version=2)
        with core.transaction(self.root) as state:
            core._find(state, "shot", "e1s2").update(version=3, job_id="failed-v3",
                generation_error="本次视频生成未得到可用结果。")
            state["jobs"]["failed-v3"] = {"id": "failed-v3", "item_id": "e1s2", "kind": "video",
                                           "amount": 2, "state": "failed", "accounting": "spent"}
        before = core.load_project(self.root)
        for action in ("approve", "reject"):
            with self.subTest(action=action), self.assertRaisesRegex(core.GateError, "不属于当前版本"):
                if action == "approve":
                    core.approve(self.root, "shot", "e1s2", 3)
                else:
                    core.reject(self.root, "shot", "e1s2", 3, "修改")
            self.assertEqual(core.load_project(self.root), before)
        legacy = self.shot()
        legacy.update(status="accepted", approved_at=core.now(), approved_sha256=legacy["sha256"])
        legacy["review"]["human_override"] = True
        self.assertFalse(core.human_approved_shot(legacy))

    def test_external_video_is_compatible_but_conflicting_job_output_is_rejected(self):
        original = self.artifact("media/imported.mp4")
        replacement = self.artifact("media/new-output.mp4")
        with core.transaction(self.root) as state:
            core._find(state, "shot", "e1s2").update(original, status="review_required", version=3)
        core.approve(self.root, "shot", "e1s2", 3)
        self.assertTrue(core.human_approved_shot(self.shot()))
        with core.transaction(self.root) as state:
            core._find(state, "shot", "e1s2").update(status="review_required", job_id="new-v3")
            state["jobs"]["new-v3"] = {"id": "new-v3", "item_id": "e1s2", "kind": "video", "amount": 2,
                                        "state": "completed", "accounting": "spent", "output": replacement["file"]}
        before = core.load_project(self.root)
        with self.assertRaisesRegex(core.GateError, "不属于当前版本"):
            core.approve(self.root, "shot", "e1s2", 3)
        self.assertEqual(core.load_project(self.root), before)

    def test_unbound_report_does_not_block_human_approval_or_enter_decision_report(self):
        for source in ("old_version", "old_batch", "missing_hash", "invalid_version"):
            with self.subTest(source=source):
                self.candidate(version=2)
                with core.transaction(self.root) as state:
                    review = core._find(state, "shot", "e1s2")["review"]
                    if source == "old_version":
                        review["version"] = 1
                    elif source == "invalid_version":
                        review["version"] = True
                    elif source == "old_batch":
                        review["sha256"] = "a" * 64
                        review["issues"] = ["粗剪检查意见，仅建议"]
                    else:
                        review.pop("sha256")
                original_review = self.shot()["review"]
                core.approve(self.root, "shot", "e1s2", 2)
                shot = self.shot()
                self.assertTrue(core.human_approved_shot(shot))
                self.assertEqual(shot["human_decision"]["report"], {})
                self.assertFalse(shot["human_decision"]["report_bound"])
                self.assertIn("未绑定当前视频版本", shot["human_decision"]["report_note"])
                self.assertEqual(shot["review"]["issues"], original_review["issues"])
                self.assertEqual(shot["review"].get("sha256"), original_review.get("sha256"))

    def test_reject_only_regenerates_named_clip_preserving_downstream_and_costs(self):
        before = self.complete_project()
        core.reject(self.root, "shot", "e1s2", 1, "用户要求修改服饰")
        state = core.load_project(self.root)
        self.assertEqual(state["jobs"], before["jobs"])
        self.assertEqual(state["budget"], before["budget"])
        self.assertEqual(state["assets"], before["assets"])
        selected = self.shot()
        self.assertEqual((selected["status"], selected["revision"]), ("planned", 1))
        self.assertNotIn("file", selected)
        self.assertEqual(selected["human_decision"]["decision"], "revision_requested")
        self.assertEqual([d["decision"] for d in selected["human_decisions"]], ["approved", "revision_requested"])
        self.assertEqual(selected["history"][-1]["sha256"], core._find(before, "shot", "e1s2")["sha256"])
        for item_id in ("e1s4", "e2s1", "e2s2"):
            original, current = core._find(before, "shot", item_id), self.shot(item_id)
            self.assertEqual(current["status"], "review_required")
            for key in ("file", "sha256", "version", "revision", "attempts", "review"):
                self.assertEqual(current[key], original[key])
            self.assertFalse(core.human_approved_shot(current))
            self.assertEqual(current["continuity_review"]["source_shots"][0]["id"], "e1s2")
        for item_id in ("e1s1", "e1s3"):
            self.assertEqual(self.shot(item_id), core._find(before, "shot", item_id))
        for ep, original in zip(state["episodes"], before["episodes"]):
            self.assertEqual(ep["status"], "revision_requested")
            for key in ("file", "sha256", "render_manifest", "verified_subtitles", "subtitle_file", "progress_stage", "progress_started_at"):
                self.assertNotIn(key, ep)
            self.assertEqual(ep["history"][-1]["file"], original["file"])
            self.assertEqual(core.digest(self.root / original["file"]), original["sha256"])
            with self.assertRaises(core.GateError):
                core.approve(self.root, "episode", ep["id"], 1)

    def test_downstream_can_be_kept_only_after_new_upstream_is_human_approved(self):
        before = self.complete_project()
        core.reject(self.root, "shot", "e1s2", 1, "修改")
        with self.assertRaises(core.GateError):
            core.approve(self.root, "shot", "e1s4", 1)
        self.candidate(version=2)
        with self.assertRaises(core.GateError):
            core.approve(self.root, "shot", "e1s4", 1)
        core.approve(self.root, "shot", "e1s2", 2)
        core.approve(self.root, "shot", "e1s4", 1)
        shot = self.shot("e1s4")
        self.assertTrue(core.human_approved_shot(shot))
        self.assertNotIn("continuity_review", shot)
        self.assertEqual(shot["human_decision"]["continuity_checked"], [
            {"id": "e1s2", "version": 2, "sha256": self.shot()["sha256"]}])
        self.assertEqual(shot["file"], core._find(before, "shot", "e1s4")["file"])
        self.assertEqual(core.load_project(self.root)["jobs"], before["jobs"])

    def test_episode_selected_rework_regenerates_only_explicit_targets(self):
        before = self.complete_project()
        core.reject(self.root, "episode", "e1", 1, "只修改第二段", ["e1s2"])
        self.assertEqual(self.shot("e1s2")["status"], "planned")
        self.assertEqual(self.shot("e1s2")["human_decision"]["decision"], "revision_requested")
        for item_id in ("e1s4", "e2s1", "e2s2"):
            shot = self.shot(item_id)
            self.assertEqual(shot["status"], "review_required")
            self.assertEqual(shot["file"], core._find(before, "shot", item_id)["file"])
            self.assertEqual((shot["version"], shot["revision"]), (1, 0))
        self.assertTrue(core.human_approved_shot(self.shot("e1s1")))
        self.assertTrue(core.human_approved_shot(self.shot("e1s3")))
        state = core.load_project(self.root)
        self.assertEqual(state["jobs"], before["jobs"])
        self.assertEqual(state["budget"], before["budget"])
        self.assertEqual(state["events"][-1]["shot_ids"], ["e1s2"])

    def test_episode_empty_selection_explicitly_reworks_whole_current_episode_only(self):
        before = self.complete_project()
        core.reject(self.root, "episode", "e1", 1, "本集全部修改", [])
        state = core.load_project(self.root)
        self.assertTrue(all(shot["status"] == "planned" for shot in state["episodes"][0]["shots"]))
        self.assertTrue(all(shot["revision"] == 1 for shot in state["episodes"][0]["shots"]))
        self.assertTrue(all(shot["human_decision"]["decision"] == "revision_requested" for shot in state["episodes"][0]["shots"]))
        for shot in state["episodes"][1]["shots"]:
            self.assertEqual(shot["status"], "review_required")
            self.assertEqual((shot["version"], shot["revision"]), (1, 0))
            self.assertEqual(shot["file"], core._find(before, "shot", shot["id"])["file"])
        self.assertEqual(state["jobs"], before["jobs"])
        self.assertEqual(state["budget"], before["budget"])

    def test_episode_repeated_request_does_not_authorize_another_attempt_for_queued_clip(self):
        self.complete_project()
        core.reject(self.root, "episode", "e1", 1, "第一次修改", ["e1s2"])
        first = self.shot()
        core.reject(self.root, "episode", "e1", 1, "补充修改意见", ["e1s2"])
        current = self.shot()
        self.assertEqual(current["feedback"], "补充修改意见")
        self.assertEqual(current["revision"], first["revision"])
        self.assertEqual(current["human_decisions"], first["human_decisions"])
        self.assertEqual(current["history"], first["history"])

    def test_episode_selected_rework_validates_every_target_before_mutating(self):
        self.complete_project()
        (self.root / self.shot("e1s4")["file"]).write_bytes(b"SYNTHETIC modified externally")
        before = core.load_project(self.root)
        with self.assertRaises(core.GateError):
            core.reject(self.root, "episode", "e1", 1, "修改两段", ["e1s2", "e1s4"])
        self.assertEqual(core.load_project(self.root), before)

    def test_invalid_episode_selection_never_becomes_full_rework(self):
        self.complete_project()
        before = core.load_project(self.root)
        for selected in ("", False, [None], ["unknown"], ["e2s1"]):
            with self.subTest(selected=selected), self.assertRaises(ValueError):
                core.reject(self.root, "episode", "e1", 1, "修改", selected)
            self.assertEqual(core.load_project(self.root), before)

    def test_episode_cannot_approve_machine_accepted_shot(self):
        self.complete_project()
        with core.transaction(self.root) as state:
            state["episodes"][0]["status"] = "awaiting_review"
            shot = state["episodes"][0]["shots"][0]
            shot.pop("human_decision")
            shot["review"].pop("human_override")
        with self.assertRaisesRegex(core.GateError, "未经人工批准"):
            core.approve(self.root, "episode", "e1", 1)

    def test_explicit_fourth_revision_is_allowed_and_old_decisions_immutable(self):
        for version in range(1, 5):
            self.candidate(version=version)
            core.approve(self.root, "shot", "e1s2", version)
            if version < 4:
                core.reject(self.root, "shot", "e1s2", version, f"修改第 {version} 版")
        shot = self.shot()
        self.assertTrue(core.human_approved_shot(shot))
        self.assertEqual(shot["version"], 4)
        self.assertEqual(shot["revision"], 3)
        self.assertEqual(len(shot["human_decisions"]), 7)
        self.assertEqual([entry["version"] for entry in shot["history"]], [1, 2, 3])
        self.assertEqual(shot["human_decisions"][0]["decision"], "approved")

    def test_spent_download_pending_cannot_be_orphaned_by_reject(self):
        self.candidate()
        with core.transaction(self.root) as state:
            state["jobs"]["pending"] = {"id": "pending", "item_id": "e2s1", "kind": "video",
                                         "amount": 2, "state": "download_pending", "accounting": "spent"}
        before = core.load_project(self.root)
        with self.assertRaises(core.GateError):
            core.reject(self.root, "shot", "e1s2", 1, "修改")
        self.assertEqual(core.load_project(self.root), before)

    def test_known_failed_job_requires_explicit_retry_but_cannot_be_approved(self):
        with core.transaction(self.root) as state:
            core._find(state, "shot", "e1s2").update(status="review_required", version=1,
                job_id="failed", generation_error="本次视频生成未得到可用结果。")
            state["jobs"]["failed"] = {"id": "failed", "item_id": "e1s2", "kind": "video",
                                        "amount": 2, "state": "failed", "accounting": "spent"}
        with self.assertRaises(core.GateError):
            core.approve(self.root, "shot", "e1s2", 1)
        core.reject(self.root, "shot", "e1s2", 1, "同意再试")
        shot = self.shot()
        self.assertEqual((shot["status"], shot["revision"]), ("planned", 1))
        self.assertNotIn("generation_error", shot)
        self.assertNotIn("job_id", shot)
        self.assertEqual(shot["human_decision"]["job_id"], "failed")
        self.assertIsNone(shot["human_decision"]["sha256"])
        self.assertEqual(shot["human_decision"]["outcome"], "generation_failed")
        self.assertEqual(core.load_project(self.root)["budget"]["spent"], 2)

    def test_missing_or_wrong_failed_job_cannot_authorize_retry(self):
        for job in ({}, {"id": "failed", "item_id": "other", "kind": "video", "amount": 1,
                         "state": "failed", "accounting": "spent"}):
            with core.transaction(self.root) as state:
                core._find(state, "shot", "e1s2").update(status="review_required", version=1,
                    job_id="failed", generation_error="本次视频生成未得到可用结果。")
                state["jobs"] = {"failed": job} if job else {}
            before = core.load_project(self.root)
            with self.assertRaisesRegex(core.GateError, "失败记录尚未核实"):
                core.reject(self.root, "shot", "e1s2", 1, "重试")
            self.assertEqual(core.load_project(self.root), before)


if __name__ == "__main__":
    unittest.main()
