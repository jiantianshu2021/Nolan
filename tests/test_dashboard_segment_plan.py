"""HTTP approval integration with real local state and no production worker."""
import copy
import http.client
import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

from norlan import core, dashboard, pipeline
from norlan.regroup import propose_segments
from test_pipeline import choose_gpt_assets, fake_media


class DashboardSegmentPlanTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        core.create_project(self.root, "HTTP审批合成测试小说。")
        plan = pipeline.demo_plan()
        for shot in plan["episodes"][0]["shots"]:
            shot["duration"] = 5
            shot["dialogue"] = "你好。"
        core.install_plan(self.root, plan)
        core.approve(self.root, "plan", "plan", 1)
        choose_gpt_assets(self.root)
        with fake_media():
            state = pipeline.run(self.root)
        for asset in state["assets"]:
            core.approve(self.root, "asset", asset["id"], asset["version"])
        core.set_paused(self.root, True)
        self.original = copy.deepcopy(core.load_project(self.root))
        state = propose_segments(self.root, "E001", [["E001-S001", "E001-S002"]])
        self.proposal = state["episodes"][0]["segment_proposal"]
        self.worker_started = threading.Event()

        def no_production(root):
            self.worker_started.set()
            return core.load_project(root)

        self.run = self.enterContext(patch.object(dashboard, "_run", side_effect=no_production))
        self.server = dashboard.make_server(self.root, port=0)
        self.port = self.server.server_address[1]
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.addCleanup(self.shutdown)

    def shutdown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)

    def request(self, method, path, payload=None, headers=None):
        connection = http.client.HTTPConnection("127.0.0.1", self.port, timeout=3)
        body = json.dumps(payload) if payload is not None else None
        request_headers = {}
        if body is not None:
            request_headers = {"Content-Type": "application/json", "X-Norlan-Token": self.server.context.token}
        request_headers.update(headers or {})
        connection.request(method, path, body=body, headers=request_headers)
        response = connection.getresponse()
        status, content = response.status, response.read()
        connection.close()
        return status, json.loads(content)

    def decision(self, approve=True, proposal_id=None):
        return {"action": "segment_plan", "id": "E001",
                "proposal_id": proposal_id or self.proposal["id"], "approve": approve}

    def test_proposal_public_fields_are_whitelisted_and_progress_waits(self):
        with core.transaction(self.root) as state:
            proposal = state["episodes"][0]["segment_proposal"]
            proposal["provider_raw"] = "SYNTHETIC_PRIVATE_PAYLOAD"
            proposal["shots"][0]["api_key"] = "SYNTHETIC_PRIVATE_PAYLOAD"
            proposal["shots"][0]["subshots"][0]["secret"] = "SYNTHETIC_PRIVATE_PAYLOAD"
        status, result = self.request("GET", "/api/state")
        self.assertEqual(status, 200)
        self.assertNotIn("SYNTHETIC_PRIVATE_PAYLOAD", json.dumps(result))
        proposal = result["episodes"][0]["segment_proposal"]
        self.assertEqual(proposal["id"], self.proposal["id"])
        self.assertEqual(proposal["shots"][0]["duration"], 10)
        self.assertEqual(len(proposal["shots"][0]["subshots"]), 2)
        self.assertNotIn("base_hash", proposal)
        self.assertNotIn("settings", result)
        progress = result["episodes"][0]["progress"]
        self.assertEqual(progress["stage"], "awaiting_segment_plan")
        self.assertTrue(progress["waiting_for_human"])
        self.assertFalse(progress["is_working"])
        self.assertEqual(progress["percent"], 0)

    def test_approval_applies_exact_proposal_then_explicit_run_can_continue(self):
        status, result = self.request("POST", "/api/action", self.decision())
        self.assertEqual(status, 200, result)
        state = core.load_project(self.root)
        self.assertTrue(state["paused"])
        self.assertEqual(state["episodes"][0]["shots"], self.proposal["shots"])
        self.assertEqual(state["episodes"][0]["shots"][0]["dialogue"], "你好。\n你好。")
        self.assertEqual(state["assets"], self.original["assets"])
        self.assertEqual(state["jobs"], self.original["jobs"])
        self.assertEqual(state["budget"], self.original["budget"])
        self.assertNotIn("segment_proposal", result["episodes"][0])
        self.run.assert_not_called()
        # The browser's explicit "approve and continue" makes this second call.
        status, _ = self.request("POST", "/api/action", {"action": "run"})
        self.assertEqual(status, 202)
        self.assertTrue(self.worker_started.wait(timeout=2))
        self.assertFalse(core.load_project(self.root)["paused"])
        self.run.assert_called_once()

    def test_cancel_preserves_original_shots_and_does_not_resume(self):
        status, result = self.request("POST", "/api/action", self.decision(False))
        self.assertEqual(status, 200, result)
        state = core.load_project(self.root)
        self.assertEqual(state["episodes"][0]["shots"], self.original["episodes"][0]["shots"])
        self.assertEqual(state["assets"], self.original["assets"])
        self.assertEqual(state["jobs"], self.original["jobs"])
        self.assertTrue(state["paused"])
        self.assertNotIn("segment_proposal", result["episodes"][0])
        self.run.assert_not_called()

    def test_stale_id_or_source_snapshot_cannot_be_approved(self):
        status, _ = self.request("POST", "/api/action", self.decision(proposal_id="old-id"))
        self.assertEqual(status, 409)
        with core.transaction(self.root) as state:
            state["episodes"][0]["shots"][0]["prompt"] += "合成测试：原方案被更改。"
        status, _ = self.request("POST", "/api/action", self.decision())
        self.assertEqual(status, 409)
        state = core.load_project(self.root)
        self.assertEqual(state["episodes"][0]["segment_proposal"]["status"], "candidate")
        self.assertEqual(len(state["episodes"][0]["shots"]), 2)
        self.assertTrue(state["paused"])
        self.run.assert_not_called()

    def test_csrf_origin_and_nonboolean_approval_are_rejected(self):
        for headers in ({"X-Norlan-Token": "incorrect"}, {"Origin": "https://example.test"}):
            with self.subTest(headers=headers):
                self.assertEqual(self.request("POST", "/api/action", self.decision(), headers)[0], 403)
        for value in ("true", 1, None):
            with self.subTest(value=value):
                self.assertEqual(self.request("POST", "/api/action", self.decision(value))[0], 409)
        state = core.load_project(self.root)
        self.assertEqual(state["episodes"][0]["segment_proposal"]["status"], "candidate")
        self.assertTrue(state["paused"])
        self.run.assert_not_called()

    def test_pending_proposal_blocks_http_run_without_unpausing_or_starting(self):
        status, result = self.request("POST", "/api/action", {"action": "run"})
        self.assertEqual(status, 409)
        self.assertIn("分段方案", result["error"])
        self.assertTrue(core.load_project(self.root)["paused"])
        self.assertFalse(self.server.context.running)
        self.run.assert_not_called()
        with patch.object(dashboard.threading, "Thread") as worker:
            with self.assertRaisesRegex(ValueError, "分段方案"):
                self.server.context.launch()
        worker.assert_not_called()
        self.assertTrue(core.load_project(self.root)["paused"])


if __name__ == "__main__":
    unittest.main()
