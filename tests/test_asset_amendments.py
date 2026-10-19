"""Synthetic offline asset-amendment gates; no real media or provider calls."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from norlan import core, mcp_server, pipeline


class AssetAmendmentTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / "project"
        self.line = "危机响应已完成，降维民生科技系统绑定成功，请查看当前任务。"
        core.create_project(self.root, "系统：" + self.line)
        core.install_plan(self.root, pipeline.demo_plan())
        core.approve(self.root, "plan", "plan", 1)
        with core.transaction(self.root) as state:
            for asset in state["assets"]:
                asset.update(self.artifact("media/" + asset["id"] + ".png"), status="approved", version=1,
                             approved_at=core.now())
                asset["approved_sha256"] = asset["sha256"]
            # Deliberately retain an old, approved audition that would not meet
            # today's selection rule. An amendment must not reconstruct it.
            state["assets"].append({"id": "old-voice", "kind": "voice", "name": "旧声音", "prompt": "男声",
                "character_id": "hero", "audition_text": "", "audition_source": "", "audition_reason": "",
                "status": "approved", "version": 3, "history": [{"version": 2}]})
            ep = state["episodes"][0]
            for shot in ep["shots"]:
                shot.update(self.artifact("media/" + shot["id"] + "-v1.mp4"), version=1, status="review_required")
            future = copy.deepcopy(ep["shots"][-1])
            future.update(id="future-shot", status="planned", version=0, attempts=[], revision=0)
            for key in ("file", "sha256"):
                future.pop(key, None)
            ep["shots"].append(future)
        self.target = "E001-S002"
        self.additions = [
            {"id": "system-source", "kind": "character", "name": "系统播报声源", "prompt": "画外系统声音，不入画", "voice_only": True},
            {"id": "system-voice", "kind": "voice", "name": "女播报音色", "prompt": "成年女声，中音，清澈",
             "character_id": "system-source", "voice_only": False, "audition_speaker": "系统",
             "audition_text": self.line, "audition_source": "系统：" + self.line, "audition_reason": "原文系统完整陈述，平稳且足够辨音"},
        ]
        self.updates = [{"id": self.target, "add_asset_ids": ["system-source", "system-voice"],
                         "prompt": "保留画面，由画外系统女声完整播报任务文字。", "dialogue": self.line},
                        {"id": "future-shot", "add_asset_ids": ["system-source", "system-voice"], "continuity_notes": "任务框使用相同女声播报"}]
        self.policy = {"system_broadcast": {"enabled": True, "character_id": "system-source", "voice_id": "system-voice", "instruction": "每次弹出任务框都以相同女性声音播报"}}

    def artifact(self, relative, content=b"synthetic offline media"):
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        return {"file": relative, "sha256": core.digest(path)}

    def propose(self, **kwargs):
        values = dict(shot_id=self.target, version=1, feedback="新增女性系统播报员，再返修第二片段", additions=self.additions,
                      shot_updates=self.updates, audio_policy=self.policy)
        values.update(kwargs)
        return core.propose_asset_amendment(self.root, **values)["asset_amendments"][-1]

    def approve_new_assets(self):
        self.assertTrue(core.asset_ready(core._find(core.load_project(self.root), "asset", "system-source")))
        with core.transaction(self.root) as state:
            asset = core._find(state, "asset", "system-voice")
            asset.update(self.artifact("media/system-voice-v1.mp4"), status="candidate", version=1)
            audio = self.artifact("media/system-voice-v1.wav", b"synthetic audio")
            asset.update(audio_file=audio["file"], audio_sha256=audio["sha256"])
        core.approve(self.root, "asset", "system-voice", 1)

    def test_candidate_preserves_assets_old_clips_and_budget(self):
        before = core.load_project(self.root)
        proposal = self.propose()
        after = core.load_project(self.root)
        self.assertEqual(proposal["status"], "candidate")
        self.assertEqual(after["assets"], before["assets"])
        self.assertEqual(after["episodes"][0]["shots"][0], before["episodes"][0]["shots"][0])
        target = core._find(after, "shot", self.target)
        target.pop("pending_asset_amendment")
        self.assertEqual(target, core._find(before, "shot", self.target))
        self.assertEqual(after["jobs"], before["jobs"])
        self.assertEqual(after["budget"], before["budget"])

    def test_accepting_definition_adds_assets_without_changing_video_or_prompt(self):
        original = core._find(core.load_project(self.root), "shot", self.target)
        proposal = self.propose()
        state = core.apply_asset_amendment(self.root, proposal["id"], "approve")
        self.assertEqual(state["asset_amendments"][-1]["status"], "awaiting_assets")
        source = core._find(state, "asset", "system-source")
        self.assertEqual((source["status"], source["version"]), ("internal", 1))
        self.assertNotIn("file", source)
        self.assertEqual(core._find(state, "asset", "system-voice")["status"], "planned")
        current = core._find(state, "shot", self.target)
        for key in ("prompt", "file", "sha256", "version", "revision"):
            self.assertEqual(current.get(key), original.get(key))
        self.assertFalse(core.activate_asset_amendment(self.root, proposal["id"]))

    def test_confirmed_future_prompt_consumes_only_its_saved_feedback(self):
        with core.transaction(self.root) as state:
            future = core._find(state, "shot", "future-shot")
            future["feedback"] = "让加工柜更有未来科技感"
            future["revision_analysis"] = {"signature": "old", "summary": "历史分析"}
        updates = copy.deepcopy(self.updates)
        updates[1]["prompt"] = "复用批准的加工柜，用工程光层增强科技感。"
        proposal = self.propose(shot_updates=updates)
        core.apply_asset_amendment(self.root, proposal["id"], "approve")
        self.approve_new_assets()
        core.activate_asset_amendment(self.root, proposal["id"])
        future = core._find(core.load_project(self.root), "shot", "future-shot")
        self.assertEqual(future["feedback"], "让加工柜更有未来科技感")
        self.assertEqual(future["revision_analysis"]["amendment_id"], proposal["id"])
        self.assertEqual(future["revision_analysis_history"], [{"signature": "old", "summary": "历史分析"}])
        with patch.object(pipeline, "_brain", side_effect=AssertionError("Do not reanalyse consumed feedback")):
            self.assertTrue(pipeline._revision_analysis(self.root, future))
        # A later human request has a different signature and must be analysed.
        with core.transaction(self.root) as state:
            core._find(state, "shot", "future-shot")["feedback"] += "\n系统播报提前到第3秒"
            state["mode"] = "live"
        changed = core._find(core.load_project(self.root), "shot", "future-shot")
        with patch.object(pipeline, "_brain", side_effect=RuntimeError("New feedback needs analysis")):
            with self.assertRaisesRegex(RuntimeError, "New feedback"):
                pipeline._revision_analysis(self.root, changed)

    def test_reference_only_update_does_not_consume_unrelated_future_feedback(self):
        with core.transaction(self.root) as state:
            core._find(state, "shot", "future-shot")["feedback"] = "重新安排人物动作"
        proposal = self.propose()
        core.apply_asset_amendment(self.root, proposal["id"], "approve")
        self.approve_new_assets()
        core.activate_asset_amendment(self.root, proposal["id"])
        future = core._find(core.load_project(self.root), "shot", "future-shot")
        self.assertNotIn("revision_analysis", future)

    def test_initial_plan_voice_only_source_is_internal_without_image_generation(self):
        other = Path(self.temp.name) / "initial-voice-source"
        core.create_project(other, "系统：" + self.line)
        plan = pipeline.demo_plan()
        plan["assets"] += self.additions
        plan["episodes"][0]["shots"][0]["asset_ids"] += ["system-source", "system-voice"]
        core.install_plan(other, plan)
        source = core._find(core.load_project(other), "asset", "system-source")
        self.assertEqual((source["status"], source["version"]), ("internal", 1))
        self.assertEqual(source["sha256"], core.voice_only_digest(source))
        self.assertNotIn("file", source)
        core.approve(other, "plan", "plan", 1)
        with self.assertRaises(core.GateError):
            core.approve(other, "asset", "system-source", 1)

    def test_all_assets_must_be_approved_then_only_target_is_queued_once(self):
        before = core.load_project(self.root)
        proposal = self.propose()
        core.apply_asset_amendment(self.root, proposal["id"], "approve")
        self.assertTrue(core.asset_ready(core._find(core.load_project(self.root), "asset", "system-source")))
        self.assertFalse(core.activate_asset_amendment(self.root, proposal["id"]))
        self.approve_new_assets_without_source()
        self.assertTrue(core.activate_asset_amendment(self.root, proposal["id"]))
        state = core.load_project(self.root)
        target = core._find(state, "shot", self.target)
        self.assertEqual(target["status"], "planned")
        self.assertEqual(target["revision"], 1)
        self.assertEqual(target["history"][-1]["sha256"], core._find(before, "shot", self.target)["sha256"])
        self.assertNotIn("file", target)
        self.assertIn("system-voice", target["asset_ids"])
        self.assertEqual(state["episodes"][0]["shots"][0], before["episodes"][0]["shots"][0])
        self.assertEqual(state["assets"][:len(before["assets"])], before["assets"])
        self.assertEqual(state["audio_policy"], self.policy)
        self.assertEqual(target["revision_analysis"]["amendment_id"], proposal["id"])
        self.assertEqual(target["revision_analysis"]["signature"],
                         hashlib.sha256(json.dumps([target["revision"], target["feedback"]], ensure_ascii=False).encode()).hexdigest())
        self.assertFalse(core.activate_asset_amendment(self.root, proposal["id"]))
        self.assertEqual(core._find(core.load_project(self.root), "shot", self.target)["revision"], 1)
        self.assertEqual(state["jobs"], before["jobs"])

    def approve_new_assets_without_source(self):
        with core.transaction(self.root) as state:
            asset = core._find(state, "asset", "system-voice")
            asset.update(self.artifact("media/system-voice-v1.mp4"), status="candidate", version=1)
            audio = self.artifact("media/system-voice-v1.wav", b"synthetic audio")
            asset.update(audio_file=audio["file"], audio_sha256=audio["sha256"])
        core.approve(self.root, "asset", "system-voice", 1)

    def add_existing_approved_system_voice(self):
        with core.transaction(self.root) as state:
            source = {**copy.deepcopy(self.additions[0]), "status": "approved", "version": 2,
                      "history": [], "approved_at": core.now()}
            source["sha256"] = source["approved_sha256"] = core.voice_only_digest(source)
            voice = {**copy.deepcopy(self.additions[1]), "status": "approved", "version": 3,
                     "history": [], "approved_at": core.now(),
                     # Old source selections are retained, never revalidated.
                     "audition_text": "", "audition_source": "", "audition_reason": "",
                     **self.artifact("media/existing-system-voice-v3.mp4")}
            audio = self.artifact("media/existing-system-voice-v3.wav", b"synthetic approved existing voice")
            voice.update(audio_file=audio["file"], audio_sha256=audio["sha256"], approved_sha256=voice["sha256"])
            state["assets"].extend([source, voice])
        core.normalize_voice_sources(self.root)

    def test_existing_voice_reuse_requires_scheme_and_changes_no_asset_or_paid_job(self):
        self.add_existing_approved_system_voice()
        original = core.load_project(self.root)
        proposal = self.propose(additions=[])
        self.assertEqual(proposal["additions"], [])
        self.assertEqual(core.load_project(self.root)["assets"], original["assets"])
        state = core.apply_asset_amendment(self.root, proposal["id"], "approve")
        self.assertEqual(state["asset_amendments"][-1]["status"], "awaiting_assets")
        self.assertEqual(state["assets"], original["assets"])
        self.assertTrue(core.activate_asset_amendment(self.root, proposal["id"]))
        state = core.load_project(self.root)
        target = core._find(state, "shot", self.target)
        self.assertIn("system-voice", target["asset_ids"])
        self.assertIn("system-source", target["asset_ids"])
        self.assertEqual(state["audio_policy"], self.policy)
        self.assertEqual(state["assets"], original["assets"])
        self.assertEqual(state["jobs"], original["jobs"])
        self.assertEqual(state["budget"], original["budget"])
        self.assertEqual(state["episodes"][0]["shots"][0], original["episodes"][0]["shots"][0])
        self.assertEqual(target["history"][-1]["sha256"], core._find(original, "shot", self.target)["sha256"])

    def test_empty_asset_scheme_without_reference_or_policy_change_is_rejected(self):
        before = (self.root / "project.json").read_bytes()
        for update in ({"id": self.target, "add_asset_ids": []},
                       {"id": self.target, "add_asset_ids": ["hero"], "prompt": "仅改画面"}):
            with self.subTest(update=update):
                with self.assertRaises(ValueError):
                    self.propose(additions=[], shot_updates=[update], audio_policy=None)
                self.assertEqual((self.root / "project.json").read_bytes(), before)

    def test_existing_asset_policy_change_can_be_reviewed_without_new_references(self):
        self.add_existing_approved_system_voice()
        with core.transaction(self.root) as state:
            core._find(state, "shot", self.target)["asset_ids"] += ["system-source", "system-voice"]
        proposal = self.propose(additions=[], shot_updates=[{"id": self.target, "add_asset_ids": []}])
        core.apply_asset_amendment(self.root, proposal["id"], "approve")
        self.assertTrue(core.activate_asset_amendment(self.root, proposal["id"]))
        self.assertEqual(core.load_project(self.root)["audio_policy"], self.policy)

    def test_already_requested_revision_is_not_invalidated_again(self):
        core.reject(self.root, "shot", self.target, 1, "女性系统播报")
        original = core._find(core.load_project(self.root), "shot", self.target)
        proposal = self.propose()
        core.apply_asset_amendment(self.root, proposal["id"], "approve")
        self.approve_new_assets()
        core.activate_asset_amendment(self.root, proposal["id"])
        target = core._find(core.load_project(self.root), "shot", self.target)
        self.assertEqual(target["revision"], original["revision"])
        self.assertEqual(target["history"], original["history"])

    def test_rejecting_candidate_does_not_add_assets_or_retire_video(self):
        original = core.load_project(self.root)
        proposal = self.propose()
        state = core.apply_asset_amendment(self.root, proposal["id"], "reject")
        self.assertEqual(state["assets"], original["assets"])
        self.assertEqual(state["episodes"], original["episodes"])
        self.assertEqual(state["asset_amendments"][-1]["status"], "rejected")

    def test_stale_asset_definition_or_shot_version_blocks_apply(self):
        proposal = self.propose()
        with core.transaction(self.root) as state:
            state["assets"][0]["prompt"] += "变更"
        before = (self.root / "project.json").read_bytes()
        with self.assertRaises(core.GateError):
            core.apply_asset_amendment(self.root, proposal["id"], "approve")
        self.assertEqual((self.root / "project.json").read_bytes(), before)

    def test_stale_target_or_future_update_blocks_apply(self):
        proposal = self.propose()
        with core.transaction(self.root) as state:
            core._find(state, "shot", "future-shot")["revision"] += 1
        with self.assertRaises(core.GateError):
            core.apply_asset_amendment(self.root, proposal["id"], "approve")

    def test_stale_proposal_can_still_be_rejected_without_overwriting_newer_work(self):
        proposal = self.propose()
        with core.transaction(self.root) as state:
            core._find(state, "shot", self.target)["prompt"] += "由用户另行修改"
        current = core.load_project(self.root)
        state = core.apply_asset_amendment(self.root, proposal["id"], "reject")
        expected = core._find(current, "shot", self.target)
        expected.pop("pending_asset_amendment")
        self.assertEqual(core._find(state, "shot", self.target), expected)

    def test_source_definition_hash_prevents_stale_approval(self):
        proposal = self.propose()
        core.apply_asset_amendment(self.root, proposal["id"], "approve")
        with core.transaction(self.root) as state:
            core._find(state, "asset", "system-source")["prompt"] += "变更"
        with self.assertRaises(core.GateError):
            core.approve(self.root, "asset", "system-source", 1)

    def test_no_target_approval_or_parallel_rework_while_asset_gate_pending(self):
        self.propose()
        with self.assertRaises(core.GateError):
            core.approve(self.root, "shot", self.target, 1)
        with self.assertRaises(core.GateError):
            core.reject(self.root, "shot", self.target, 1, "另一返工")
        with self.assertRaises(core.GateError):
            self.propose()

    def test_revision_clears_old_prompt_authorization_and_keeps_candidate_history(self):
        old = {"id": "synthetic-old-candidate", "status": "approved", "request_hash": "a" * 64}
        with core.transaction(self.root) as state:
            core._find(state, "shot", self.target).update(prompt_candidate=copy.deepcopy(old),
                prompt_approval={"candidate_id": old["id"], "request_hash": old["request_hash"]},
                revision_analysis={"signature": "stale"})
        core.reject(self.root, "shot", self.target, 1, "调整原片段")
        shot = core._find(core.load_project(self.root), "shot", self.target)
        self.assertEqual(shot["prompt_history"][-1], old)
        self.assertFalse({"prompt_candidate", "prompt_approval", "revision_analysis"}.intersection(shot))

    def test_refuse_overwriting_assets_or_editing_earlier_generated_clip(self):
        cases = [dict(additions=[{**self.additions[0], "id": "hero"}]),
                 dict(shot_updates=[{**self.updates[0], "id": "E001-S001"}]),
                 dict(additions=[{**self.additions[0], "status": "approved"}]),
                 dict(shot_updates=[{**self.updates[0], "file": "media/old.mp4"}]),
                 dict(shot_updates=[{**self.updates[0], "add_asset_ids": ["system-voice"]}])]
        for values in cases:
            with self.subTest(values=values):
                before = (self.root / "project.json").read_bytes()
                with self.assertRaises((ValueError, core.GateError)):
                    self.propose(**values)
                self.assertEqual((self.root / "project.json").read_bytes(), before)

    def test_source_audition_uses_original_speaker_instead_of_display_name(self):
        proposal = self.propose()
        self.assertEqual(proposal["additions"][1]["audition_speaker"], "系统")
        bad = copy.deepcopy(self.additions)
        bad[1]["audition_speaker"] = "不存在的说话人"
        core.apply_asset_amendment(self.root, proposal["id"], "reject")
        with self.assertRaises(core.GateError):
            self.propose(additions=bad)

    def test_proposal_can_run_inside_existing_runner_lock(self):
        with core.run_lock(self.root):
            self.assertEqual(self.propose(already_locked=True)["status"], "candidate")

    def test_unknown_provider_outcome_blocks_proposal(self):
        with core.transaction(self.root) as state:
            state["jobs"]["unresolved"] = {"kind": "video", "item_id": self.target, "state": "uncertain", "accounting": "unknown", "amount": 1}
        with self.assertRaises(core.GateError):
            self.propose()

    def test_mcp_only_saves_candidate_opens_review_and_preserves_state_if_browser_fails(self):
        with patch.object(mcp_server, "_last_project", return_value=None):
            server = mcp_server.MCPServer(Path(self.temp.name) / "workspace", project=self.root)
        self.addCleanup(server.close)
        server.handle({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {
            "protocolVersion": "2025-06-18", "capabilities": {}, "clientInfo": {"name": "synthetic", "version": "1"}}})
        before = core.load_project(self.root)
        with patch.object(server, "_open_review", side_effect=RuntimeError("synthetic browser failure")) as opened, \
                patch.object(core, "install_plan") as install, patch.object(pipeline, "run") as run:
            result = server.call_tool("propose_asset_amendment", {
                "project": str(self.root), "shot_id": self.target, "version": 1, "feedback": "新增系统女声",
                "additions": self.additions, "shot_updates": self.updates, "audio_policy": self.policy})
        self.assertIn("候选已保存", result["message"])
        self.assertEqual(opened.call_count, 1)
        install.assert_not_called()
        run.assert_not_called()
        state = core.load_project(self.root)
        self.assertEqual(state["assets"], before["assets"])
        self.assertEqual(state["jobs"], before["jobs"])
        self.assertEqual(state["asset_amendments"][-1]["status"], "candidate")


class AssetAmendmentMCPTests(unittest.TestCase):
    def test_only_proposal_tool_is_exposed_with_closed_schemas(self):
        tool = mcp_server.TOOL_MAP["propose_asset_amendment"]
        self.assertFalse(tool["inputSchema"]["additionalProperties"])
        self.assertFalse(mcp_server.ASSET["additionalProperties"])
        self.assertFalse(mcp_server.AMENDMENT_SHOT_UPDATE["additionalProperties"])
        self.assertNotIn("apply_asset_amendment", mcp_server.TOOL_MAP)
        mcp_server._validate({"project": "/synthetic/project", "shot_id": "shot", "version": 1,
                             "feedback": "复用已有声音", "additions": [],
                             "shot_updates": [{"id": "shot", "add_asset_ids": ["existing-voice"]}]}, tool["inputSchema"])
        for forbidden in ("status", "approved", "file", "settings", "budget"):
            with self.subTest(forbidden=forbidden):
                arguments = {"project": "/synthetic/project", "shot_id": "shot", "version": 1, "feedback": "新增女声",
                             "additions": [{"id": "voice", "kind": "character", "name": "声源", "prompt": "画外声", forbidden: "forbidden"}],
                             "shot_updates": [{"id": "shot", "add_asset_ids": ["voice"]}]}
                with self.assertRaises(mcp_server.ArgumentValidationError):
                    mcp_server._validate(arguments, tool["inputSchema"])


if __name__ == "__main__":
    unittest.main()
