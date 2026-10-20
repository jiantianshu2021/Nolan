"""Synthetic dashboard cost workers; no provider API calls or production runs."""

import copy
import http.client
import json
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

from norlan import dashboard
from tests import test_dashboard as fixtures


class DashboardCostsTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name).resolve()
        self.state = {"title": "Synthetic cost dashboard", "mode": "live", "status": "awaiting_assets", "paused": True,
                      "plan": {"status": "approved", "version": 1}, "assets": [], "episodes": [],
                      "budget": {"limit": 100, "spent": 5, "reserved": 0, "unknown": 0, "currency": "CNY"},
                      "jobs": {"video-synthetic": {"amount": 5, "accounting": "spent"}}}
        self.core = fixtures.FakeCore(self.state)
        self.core_patch = patch.object(dashboard, "_core", return_value=self.core)
        self.core_patch.start()
        self.server = dashboard.make_server(self.root, port=0)
        self.port = self.server.server_address[1]
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)
        self.core_patch.stop()
        self.temp.cleanup()

    def request(self, method, payload=None, headers=None, path="/api/action"):
        connection = http.client.HTTPConnection("127.0.0.1", self.port, timeout=3)
        body = json.dumps(payload) if payload is not None else None
        request_headers = {}
        if body is not None:
            request_headers.update({"Content-Type": "application/json", "X-Norlan-Token": self.server.context.token})
        request_headers.update(headers or {})
        connection.request(method, path, body=body, headers=request_headers)
        response = connection.getresponse()
        result = response.status, json.loads(response.read())
        connection.close()
        return result

    def wait_done(self):
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline:
            if not self.server.context.state()["dashboard"]["cost_sync"]["running"]:
                return
            threading.Event().wait(0.005)
        self.fail("Synthetic cost worker did not finish")

    def result(self, **values):
        return {"synced": 1, "skipped": 2, "unavailable": 3, "errors": [], **values}

    def run_mock_worker(self, sync_result=None, error=None):
        with patch.object(dashboard.threading, "Thread") as thread, patch("norlan.costs.sync_usage") as sync:
            if error is not None:
                sync.side_effect = error
            else:
                sync.return_value = sync_result if sync_result is not None else self.result()
            self.assertTrue(self.server.context.launch_cost_sync())
            thread.call_args.kwargs["target"]()
            sync.assert_called_once_with(self.root, confirm_no_video_input=False)
        return self.server.context.state()["dashboard"]["cost_sync"]

    def test_independent_worker_and_duplicate_start_do_not_run_or_approve(self):
        context = self.server.context
        context.running = True
        with patch.object(dashboard.threading, "Thread") as thread, patch("norlan.costs.sync_usage", return_value=self.result()) as sync, \
             patch.object(dashboard, "_run") as run, patch.object(self.core, "set_paused") as pause, \
             patch.object(self.core, "approve") as approve:
            self.assertTrue(context.launch_cost_sync())
            self.assertFalse(context.launch_cost_sync())
            self.assertEqual(thread.call_count, 1)
            thread.return_value.start.assert_called_once()
            sync.assert_not_called()
            self.assertTrue(context.state()["dashboard"]["cost_sync"]["running"])
            self.assertEqual(thread.call_args.kwargs["name"], "norlan-cost-sync")
            self.assertTrue(thread.call_args.kwargs["daemon"])
            thread.call_args.kwargs["target"]()
            sync.assert_called_once_with(self.root, confirm_no_video_input=False)
            run.assert_not_called()
            pause.assert_not_called()
            approve.assert_not_called()
        self.assertTrue(context.running)
        self.assertTrue(self.state["paused"])
        self.assertFalse(context.state()["dashboard"]["cost_sync"]["running"])

    def test_http_sync_runs_during_production_and_rejects_duplicate_launch(self):
        release, started = threading.Event(), threading.Event()

        def sync(_root, confirm_no_video_input):
            self.assertFalse(confirm_no_video_input)
            started.set()
            release.wait(timeout=2)
            return self.result()

        context = self.server.context
        context.running = True
        with patch("norlan.costs.sync_usage", side_effect=sync) as lookup, patch.object(dashboard, "_run") as run, \
             patch.object(self.core, "approve") as approve, patch.object(self.core, "set_paused") as pause:
            try:
                status, state = self.request("POST", {"action": "sync_costs", "confirm_no_video_input": True})
                self.assertEqual(status, 202)
                self.assertTrue(started.wait(timeout=1))
                self.assertTrue(state["dashboard"]["cost_sync"]["running"])
                self.assertEqual(self.request("POST", {"action": "sync_costs"})[0], 409)
                self.assertEqual(lookup.call_count, 1)
            finally:
                release.set()
                self.wait_done()
            run.assert_not_called()
            approve.assert_not_called()
            pause.assert_not_called()
        self.assertTrue(context.running)
        self.assertTrue(self.state["paused"])

    def test_sync_result_and_ledger_totals_reach_public_state(self):
        def sync(_root, confirm_no_video_input):
            self.state["jobs"]["video-synthetic"]["billing"] = {
                "currency": "CNY", "usage_cost": 3.7, "status": "usage_priced", "raw": "SECRET-RAW"}
            return self.result()

        with patch("norlan.costs.sync_usage", side_effect=sync):
            self.assertEqual(self.request("POST", {"action": "sync_costs"})[0], 202)
            self.wait_done()
        status, state = self.request("GET", path="/api/state")
        self.assertEqual(status, 200)
        self.assertEqual(state["budget"]["usage_cost"], 3.7)
        self.assertEqual(state["budget"]["usage_cost_jobs"], 1)
        self.assertIsNone(state["budget"]["actual_cost"])
        self.assertEqual(state["dashboard"]["cost_sync"]["result"]["synced"], 1)
        self.assertNotIn("SECRET-RAW", json.dumps(state))
        self.assertEqual(self.core.approvals, [])

    def test_cost_projection_ignores_legacy_budget_ceiling_and_tracks_unsettled_calls(self):
        self.state["budget"].update(approved=False, limit=0, available=-999)
        self.state["jobs"].update({
            "synthetic-active": {"amount": 0, "accounting": "reserved", "state": "submitted"},
            "synthetic-unknown": {"amount": 3, "accounting": "unknown", "state": "uncertain"},
        })
        before = copy.deepcopy(self.state)
        data = dashboard.public_state(self.state)
        for retired in ("limit", "available", "approved", "spent", "reserved", "unknown"):
            self.assertNotIn(retired, data["budget"])
        self.assertEqual(data["budget"]["inflight_jobs"], 1)
        self.assertEqual(data["budget"]["unknown_jobs"], 1)
        self.assertEqual(data["budget"]["cost_pending_jobs"], 2)
        self.assertIsNone(data["budget"]["actual_cost"])
        self.assertEqual(self.state, before)

    def test_csrf_and_origin_still_required_for_cost_sync(self):
        with patch.object(self.server.context, "launch_cost_sync") as launch:
            self.assertEqual(self.request("POST", {"action": "sync_costs"}, {"X-Norlan-Token": "wrong"})[0], 403)
            self.assertEqual(self.request("POST", {"action": "sync_costs"}, {"Origin": "https://evil.example"})[0], 403)
            self.assertEqual(self.request("POST", {"action": "sync_costs"}, {"Host": "evil.example"})[0], 403)
            self.assertEqual(self.request("POST", {"action": "sync_costs"}, {"Content-Type": "text/plain"})[0], 415)
            self.assertEqual(self.request("GET", path="/api/action?action=sync_costs")[0], 404)
            launch.assert_not_called()

    def test_result_whitelist_never_exposes_raw_error_fields(self):
        result = self.result(secret="SECRET-TOP", errors=[{"job_id": "video-synthetic", "error": "SECRET-ERROR", "response": "SECRET-RESPONSE"},
                                                         {"job_id": "https://example.test?key=SECRET-ID", "error": "SECRET-ERROR"}])
        state = self.run_mock_worker(result)
        self.assertIsNone(state["error"])
        self.assertEqual(state["result"]["error_count"], 2)
        self.assertEqual(state["result"]["errors"][0]["job_id"], "video-synthetic")
        self.assertEqual(state["result"]["errors"][1]["job_id"], "")
        self.assertNotIn("SECRET", json.dumps(state))
        many = dashboard._public_cost_sync_result(self.result(errors=[{"job_id": "video-synthetic"}] * 101))
        self.assertEqual(many["error_count"], 101)
        self.assertEqual(len(many["errors"]), 100)

    def test_worker_failure_is_safe_and_can_be_retried(self):
        self.server.context.worker_error = "Unrelated production error"
        state = self.run_mock_worker(error=RuntimeError("SECRET-EXCEPTION"))
        self.assertFalse(state["running"])
        self.assertIsNone(state["result"])
        self.assertIn("费用同步未完成", state["error"])
        self.assertNotIn("SECRET", json.dumps(state))
        self.assertEqual(self.server.context.worker_error, "Unrelated production error")
        state = self.run_mock_worker()
        self.assertIsNone(state["error"])
        self.assertEqual(state["result"]["synced"], 1)

    def test_invalid_summary_and_thread_start_failure_do_not_leak(self):
        state = self.run_mock_worker(self.result(synced=True, error="SECRET"))
        self.assertIsNone(state["result"])
        self.assertIn("费用同步未完成", state["error"])
        with patch.object(dashboard.threading, "Thread") as thread:
            thread.return_value.start.side_effect = RuntimeError("SECRET-THREAD")
            with self.assertRaises(RuntimeError) as caught:
                self.server.context.launch_cost_sync()
        self.assertNotIn("SECRET", str(caught.exception))
        state = self.server.context.state()["dashboard"]["cost_sync"]
        self.assertFalse(state["running"])
        self.assertIn("费用同步未能启动", state["error"])


@unittest.skipUnless(shutil.which("node"), "Node.js required for synthetic DOM checks")
class CostDisplayDomTests(unittest.TestCase):
    def test_zero_bill_pending_cost_and_unknown_calls_render_without_budget_panel(self):
        html = Path(dashboard.__file__).with_name("dashboard.html").read_text()
        self.assertIn("<h2>费用记录</h2>", html)
        self.assertNotIn("费用与预算", html)
        self.assertNotIn("预算占用", html)
        self.assertNotIn("剩余预算", html)
        sources = []
        for name in ("el", "budgetNumber", "money", "paintBudget"):
            start = re.search(r"function " + name + r"\(", html).start()
            following = re.search(r"\n(?:async )?function ", html[start + 1:])
            sources.append(html[start:start + 1 + following.start()])
        program = r'''
const assert=require('node:assert/strict'),vm=require('node:vm');
const source=JSON.parse(require('node:fs').readFileSync(0,'utf8')).join('\n');
class Element{
 constructor(tag='div'){this.tag=tag;this.children=[];this.textContent='';}
 append(...nodes){this.children.push(...nodes);}
 replaceChildren(...nodes){this.children=[...nodes];}
}
const nodes=new Map(),$=id=>{if(!nodes.has(id))nodes.set(id,new Element());return nodes.get(id);};
const context={assert,$,document:{createElement:tag=>new Element(tag)}};
vm.runInNewContext(source+`
const pending={currency:'CNY',cost_jobs:3,actual_cost_jobs:0,actual_cost:null,
 usage_cost_jobs:1,usage_cost:2.5,cost_pending_jobs:3,inflight_jobs:1,unknown_jobs:2,
 limit:2000,available:999,spent:400,reserved:100,unknown:77};
paintBudget(pending);
assert.equal($('budget-total').textContent,'待对账');
assert.equal(JSON.stringify($('budget').children.map(n=>n.children.map(x=>x.textContent))),JSON.stringify([
 ['按实际用量计费（未对账）','2.50 CNY'],['未核实调用数','3 次'],['进行中调用','1 次'],['结果待核实调用','2 次']]));
paintBudget({...pending,actual_cost_jobs:1,actual_cost:0});
assert.equal($('budget-total').textContent,'0.00 CNY');
assert.equal($('budget-status').textContent,'已核对 1 / 3 次调用');
`,context);
'''
        result = subprocess.run([shutil.which("node"), "-e", program], input=json.dumps(sources),
                                text=True, capture_output=True, timeout=15)
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
