"""Synthetic source-selection tests; no real project or model is accessed."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from norlan import core, mcp_server, pipeline
from norlan.voice import audition_selection_key


TEXT = "我们先把这些事情说明白，再一起安排接下来的工作。"
SOURCE = "测试角色：" + TEXT
REASON = "合成原文中归属明确的连续完整陈述，适合自然稳定试音。"


class VoiceSelectionTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.workspace = Path(temporary.name).resolve() / "workspace"
        self.enterContext(patch.object(mcp_server, "_last_project", return_value=None))
        self.server = mcp_server.MCPServer(self.workspace)
        self.addCleanup(self.server.close)
        self.root = self.server.projects / "synthetic-story"
        self.server.handle({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {
            "protocolVersion": "2025-06-18", "capabilities": {}, "clientInfo": {"name": "synthetic-test", "version": "1"}}})
        core.create_project(self.root, "合成回归剧本\n" + SOURCE, mode="live")
        plan = pipeline.demo_plan()
        plan["assets"][0]["name"] = "测试角色"
        core.install_plan(self.root, plan)
        with core.transaction(self.root) as state:
            state["assets"].append({"id": "voice", "kind": "voice", "name": "测试角色声音", "character_id": "hero",
                                    "prompt": "中年男声", "feedback": "声线略低一点", "status": "planned", "version": 0,
                                    "history": [], "audition_selection_error": "试音来源校验未通过。",
                                    "audition_selection_key": "previous-selection-error"})

    def voice(self):
        return core._find(core.load_project(self.root), "asset", "voice")

    def set_selection(self):
        return core.set_voice_audition(self.root, "voice", TEXT, SOURCE, REASON)

    def call(self, name, **arguments):
        return self.server.handle({"jsonrpc": "2.0", "id": 2, "method": "tools/call", "params": {
            "name": name, "arguments": arguments}})["result"]

    def test_valid_selection_unlocks_metadata_without_approval_generation_or_version_change(self):
        before = core.load_project(self.root)
        with patch.object(pipeline, "run") as run, patch.object(core, "reserve", wraps=core.reserve) as reserve:
            after = self.set_selection()
        voice = core._find(after, "asset", "voice")
        self.assertEqual((voice["audition_text"], voice["audition_source"], voice["audition_reason"]), (TEXT, SOURCE, REASON))
        self.assertNotIn("audition_selection_error", voice)
        self.assertEqual(voice["audition_selection_key"], audition_selection_key(voice, "测试角色", core.digest(self.root / "source/novel.txt")))
        for key in ("status", "version", "history", "prompt", "feedback"):
            self.assertEqual(voice[key], core._find(before, "asset", "voice")[key])
        for key in ("plan", "episodes", "budget", "jobs", "paused", "status"):
            self.assertEqual(after[key], before[key])
        self.assertNotIn("approved_at", voice)
        run.assert_not_called()
        reserve.assert_not_called()

    def test_rejected_voice_retains_media_history_and_approval_decision(self):
        with core.transaction(self.root) as state:
            core._find(state, "asset", "voice").update(status="rejected", version=3,
                file="media/old.mp4", sha256="synthetic-old-hash", audio_file="media/old.wav",
                audio_sha256="synthetic-audio-hash", history=[{"version": 2, "file": "media/v2.mp4"}])
        before = copy.deepcopy(self.voice())
        self.set_selection()
        for key in ("status", "version", "file", "sha256", "audio_file", "audio_sha256", "history"):
            self.assertEqual(self.voice()[key], before[key])

    def test_candidate_approved_generating_and_nonvoice_are_rejected_without_changes(self):
        for status in ("candidate", "approved", "generating"):
            with core.transaction(self.root) as state:
                core._find(state, "asset", "voice")["status"] = status
            before = (self.root / "project.json").read_bytes()
            with self.subTest(status=status), self.assertRaises(core.GateError):
                self.set_selection()
            self.assertEqual((self.root / "project.json").read_bytes(), before)
        before = (self.root / "project.json").read_bytes()
        with self.assertRaisesRegex(core.GateError, "只能设置在声音资产"):
            core.set_voice_audition(self.root, "hero", TEXT, SOURCE, REASON)
        self.assertEqual((self.root / "project.json").read_bytes(), before)

    def test_busy_run_and_inflight_or_unknown_jobs_block_selection(self):
        before = (self.root / "project.json").read_bytes()
        with core.run_lock(self.root), self.assertRaisesRegex(core.GateError, "运行中"):
            self.set_selection()
        self.assertEqual((self.root / "project.json").read_bytes(), before)
        for status, accounting in (("submitting", "reserved"), ("submitted", "reserved"),
                                   ("download_pending", "spent"), ("failed", "unknown")):
            with core.transaction(self.root) as state:
                state["jobs"] = {"unlinked-voice-job": {"id": "unlinked-voice-job", "kind": "voice", "item_id": "voice",
                    "state": status, "accounting": accounting, "amount": 0, "task_id": "synthetic-task"}}
            before = (self.root / "project.json").read_bytes()
            with self.subTest(status=status), self.assertRaisesRegex(core.GateError, "在途或结果不明"):
                self.set_selection()
            self.assertEqual((self.root / "project.json").read_bytes(), before)

    def test_source_speaker_length_and_missing_fields_rejected_atomically(self):
        examples = [
            (TEXT, "另一角色：" + TEXT, REASON),
            ("我新编了一句听起来很平稳但并不在原文里的长台词。", SOURCE, REASON),
            (TEXT, SOURCE, ""),
            ("", "", "原文没有合适台词"),
            (TEXT, SOURCE, "x" * 4001),
        ]
        for text, source, reason in examples:
            before = (self.root / "project.json").read_bytes()
            with self.subTest(text=text), self.assertRaises(core.GateError):
                core.set_voice_audition(self.root, "voice", text, source, reason)
            self.assertEqual((self.root / "project.json").read_bytes(), before)

    def test_mcp_tool_exposes_selection_and_safe_errors_but_never_approves_or_launches(self):
        with patch.object(pipeline, "run") as run, patch.object(self.server, "_launch") as launch:
            result = self.call("set_voice_audition", project=str(self.root), id="voice",
                               audition_text=TEXT, audition_source=SOURCE, audition_reason=REASON)
        self.assertFalse(result["isError"], result)
        public = json.loads(result["content"][0]["text"])
        voice = next(asset for asset in public["assets"] if asset["id"] == "voice")
        self.assertEqual((voice["audition_text"], voice["audition_source"], voice["audition_reason"]), (TEXT, SOURCE, REASON))
        self.assertEqual((voice["status"], voice["version"]), ("planned", 0))
        self.assertEqual(public["plan"]["status"], "candidate")
        self.assertNotIn("audition_selection_error", voice)
        run.assert_not_called()
        launch.assert_not_called()
        with core.transaction(self.root) as state:
            core._find(state, "asset", "voice")["audition_selection_error"] = "试音 source mismatch; secret=synthetic-private-value"
        status = self.call("project_status", project=str(self.root))
        serialized = status["content"][0]["text"]
        self.assertNotIn("synthetic-private-value", serialized)
        self.assertIn("set_voice_audition", serialized)

    def test_mcp_cannot_set_approved_status_or_change_approved_asset(self):
        before = (self.root / "project.json").read_bytes()
        args = dict(project=str(self.root), id="voice", audition_text=TEXT,
                    audition_source=SOURCE, audition_reason=REASON)
        self.assertTrue(self.call("set_voice_audition", **args, status="approved")["isError"])
        self.assertEqual((self.root / "project.json").read_bytes(), before)
        with core.transaction(self.root) as state:
            core._find(state, "asset", "voice").update(status="approved", version=1)
        before = (self.root / "project.json").read_bytes()
        self.assertTrue(self.call("set_voice_audition", **args)["isError"])
        self.assertEqual((self.root / "project.json").read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
