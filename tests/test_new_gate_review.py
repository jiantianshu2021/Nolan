"""Synthetic human gates open one reusable review page; no browser or model calls."""
import copy
import hashlib
import json
from contextlib import redirect_stdout
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from norlan import cli, core, mcp_server, pipeline


class NewGateReviewTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.server = mcp_server.MCPServer(Path(temporary.name) / "workspace")
        self.addCleanup(self.server.close)
        self.root = self.server.projects / "synthetic-story"
        core.create_project(self.root, "合成人工关卡测试。", "demo")
        core.install_plan(self.root, pipeline.demo_plan())
        self.browser = self.enterContext(patch("norlan.mcp_server.webbrowser.open", return_value=True))

    def state(self, status):
        state = core.load_project(self.root)
        state["status"] = status
        if status == "awaiting_prompt_approval":
            state["episodes"][0]["shots"][0].update(status=status, prompt_candidate={"id": "prompt-synthetic-a"})
        elif status == "awaiting_asset_amendment":
            state["asset_amendments"] = [{"id": "amend-synthetic-a", "version": 1, "status": "candidate"}]
        elif status == "asset_generation_failed":
            state["assets"][0].update(status="generation_failed", version=1)
        return state

    def launch(self, state):
        with patch("norlan.pipeline.run", return_value=state) as run:
            self.assertTrue(self.server._launch(self.root)["started"])
            self.server.workers[self.root].join(2)
            self.assertFalse(self.server.workers[self.root].is_alive())
            run.assert_called_once_with(self.root)

    def test_prompt_candidate_opens_once_and_next_candidate_reuses_the_server(self):
        state = self.state("awaiting_prompt_approval")
        self.launch(state)
        first = self.server.reviews[self.root][0]
        self.assertEqual(self.browser.call_count, 1)
        self.launch(state)
        self.assertEqual(self.browser.call_count, 1)
        state["episodes"][0]["shots"][0]["prompt_candidate"]["id"] = "prompt-synthetic-b"
        self.launch(state)
        self.assertEqual(self.browser.call_count, 2)
        self.assertIs(self.server.reviews[self.root][0], first)
        state["episodes"][0]["shots"][0]["prompt_candidate"]["id"] = "prompt-synthetic-a"
        self.launch(state)
        self.assertEqual(self.browser.call_count, 2)
        self.assertEqual(core.load_project(self.root)["jobs"], {})

    def test_each_amendment_opens_once_without_opening_old_candidates_again(self):
        state = self.state("awaiting_asset_amendment")
        self.launch(state)
        self.launch(state)
        self.assertEqual(self.browser.call_count, 1)
        state["asset_amendments"][0]["id"] = "amend-synthetic-b"
        self.launch(state)
        self.assertEqual(self.browser.call_count, 2)
        self.assertEqual(len(self.server.reviews), 1)

    def test_each_failed_asset_version_opens_once(self):
        state = self.state("asset_generation_failed")
        self.launch(state)
        self.launch(state)
        self.assertEqual(self.browser.call_count, 1)
        state["assets"][0]["generation_error"] = "原因文字变化不重复打开。"
        self.launch(state)
        self.assertEqual(self.browser.call_count, 1)
        state["assets"][0]["version"] = 2
        self.launch(state)
        self.assertEqual(self.browser.call_count, 2)

    def planning_state(self):
        state = self.state("blocked")
        state["plan"]["status"] = "approved"
        for asset in state["assets"]:
            asset["status"] = "approved"  # Synthetic projection fixture; no media calls.
        shot = state["episodes"][0]["shots"][0]
        shot.update(status="planned", version=0, feedback="增加系统播报", revision=0)
        signature = hashlib.sha256(json.dumps([0, shot["feedback"]], ensure_ascii=False).encode()).hexdigest()
        shot["revision_analysis"] = {"signature": signature, "blockers": ["请确认播报起点"], "summary": "等待时间安排"}
        return state

    def test_new_planning_conflict_opens_review_once(self):
        state = self.planning_state()
        self.launch(state)
        self.launch(state)
        self.assertEqual(self.browser.call_count, 1)
        self.assertEqual(self.server._review_gate_key(state)[0], "planning")

    def test_cached_planning_conflict_opens_feedback_instead_of_rerunning(self):
        state = self.planning_state()
        with core.transaction(self.root) as stored:
            stored.update(state)
        before = (self.root / "project.json").read_bytes()
        with patch.object(pipeline, "run", side_effect=AssertionError("Do not rerun cached blocker")):
            first = self.server._launch(self.root)
            second = self.server._launch(self.root)
        self.assertFalse(first["started"])
        self.assertIn("调整意见", first["reason"])
        self.assertEqual(first["review"]["url"], second["review"]["url"])
        self.assertEqual(self.browser.call_count, 1)
        self.assertEqual(self.server.workers, {})
        self.assertEqual((self.root / "project.json").read_bytes(), before)

    def test_review_server_failure_does_not_cache_gate_or_invalidate_the_stop(self):
        state = self.state("awaiting_prompt_approval")
        with patch("norlan.dashboard.make_server", side_effect=OSError("synthetic-private-failure")):
            self.launch(state)
        self.assertEqual(self.server.review_errors[self.root], mcp_server.REVIEW_ERROR_MESSAGE)
        self.assertNotIn(self.root, self.server.review_gate_keys)
        self.assertFalse(self.server.reviews)
        self.launch(state)
        self.assertNotIn(self.root, self.server.review_errors)
        self.assertEqual(self.browser.call_count, 1)
        self.assertNotIn(self.root, self.server.worker_errors)

    def test_manual_open_remains_available_after_automatic_open(self):
        self.launch(self.state("awaiting_prompt_approval"))
        first = self.server.reviews[self.root][0]
        result = self.server._open_review(self.root)
        self.assertTrue(result["browser_opened"])
        self.assertEqual(self.browser.call_count, 2)
        self.assertIs(self.server.reviews[self.root][0], first)

    def test_unrelated_status_does_not_open_or_change_tool_permissions(self):
        tools = copy.deepcopy(mcp_server.TOOLS)
        for status in ("complete", "blocked", "review_required", "awaiting_assets"):
            self.launch(self.state(status))
        self.browser.assert_not_called()
        self.assertEqual(mcp_server.TOOLS, tools)
        for tool in ("approve_prompt", "apply_asset_amendment", "approve", "configure"):
            self.assertNotIn(tool, mcp_server.TOOL_MAP)

    def test_cli_run_and_resume_keep_review_alive_at_each_new_human_gate(self):
        for status in ("awaiting_prompt_approval", "awaiting_asset_amendment", "asset_generation_failed"):
            state = self.state(status)
            for command in ("run", "resume"):
                with self.subTest(status=status, command=command), patch("norlan.cli.remember"), \
                        patch("norlan.pipeline.run", return_value=state) as run, \
                        patch("norlan.core.set_paused", return_value=state), \
                        patch("norlan.dashboard.serve") as serve, redirect_stdout(io.StringIO()) as output:
                    self.assertEqual(cli.main([command, str(self.root)]), 0)
                    serve.assert_called_once_with(self.root, port=0, open_browser=True)
                    self.assertEqual(run.call_count, 1 if command == "run" else 0)
                    self.assertIn(cli.STATUS[status], output.getvalue())


if __name__ == "__main__":
    unittest.main()
