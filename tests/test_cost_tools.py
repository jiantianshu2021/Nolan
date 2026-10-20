"""Synthetic MCP costs tests; synchronization is mocked, no external calls."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from norlan import core, costs, mcp_server, pipeline


class CostToolTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.enterContext(patch.object(mcp_server, "_last_project", return_value=None))
        self.server = mcp_server.MCPServer(Path(temporary.name).resolve() / "workspace")
        self.addCleanup(self.server.close)
        self.root = self.server.projects / "synthetic-costs"
        core.create_project(self.root, "合成测试小说", mode="live")
        self.server.handle({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {
            "protocolVersion": "2025-06-18", "capabilities": {}, "clientInfo": {"name": "synthetic-test", "version": "1"}}})

    def call(self, **arguments):
        return self.server.handle({"jsonrpc": "2.0", "id": 2, "method": "tools/call", "params": {
            "name": "get_costs", "arguments": {"project": str(self.root), **arguments}}})["result"]

    def test_default_reports_local_costs_without_sync_or_state_changes(self):
        before = (self.root / "project.json").read_bytes()
        with patch.object(costs, "sync_usage") as sync, patch.object(pipeline, "run") as run:
            result = self.call()
        self.assertFalse(result["isError"], result)
        report = json.loads(result["content"][0]["text"])
        self.assertEqual(report["calls"], [])
        self.assertIn("账单", report["basis"]["actual_cost"])
        self.assertIn("不等于账户扣款", report["basis"]["usage_cost"])
        for retired in ("limit", "available", "approved", "spent", "reserved", "unknown"):
            self.assertNotIn(retired, report["budget"])
        self.assertEqual(report["budget"]["inflight_jobs"], 0)
        self.assertEqual(report["budget"]["unknown_jobs"], 0)
        self.assertNotIn("budget_cap", report["basis"])
        self.assertNotIn("sync", report)
        self.assertEqual((self.root / "project.json").read_bytes(), before)
        sync.assert_not_called()
        run.assert_not_called()

    def test_sync_passes_explicit_history_confirmation_and_never_generates_or_imports_bills(self):
        for confirmation in (False, True):
            with self.subTest(confirmation=confirmation), patch.object(costs, "sync_usage", return_value={"synced": 1}) as sync, \
                    patch.object(pipeline, "run") as run, patch.object(core, "import_bill_costs") as import_bill:
                result = self.call(sync=True, confirm_no_video_input=confirmation)
            self.assertFalse(result["isError"], result)
            sync.assert_called_once_with(self.root.resolve(), confirm_no_video_input=confirmation)
            self.assertEqual(json.loads(result["content"][0]["text"])["sync"], {"synced": 1})
            run.assert_not_called()
            import_bill.assert_not_called()

    def test_confirmation_without_sync_and_nonboolean_options_are_rejected(self):
        for arguments in ({"confirm_no_video_input": True}, {"sync": "true"}, {"sync": 1},
                          {"sync": True, "confirm_no_video_input": "yes"}, {"import_bill": []}):
            with self.subTest(arguments=arguments), patch.object(costs, "sync_usage") as sync:
                self.assertTrue(self.call(**arguments)["isError"])
                sync.assert_not_called()


if __name__ == "__main__":
    unittest.main()
