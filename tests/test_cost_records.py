"""Synthetic projects and mocked GET responses; no real provider requests."""

from contextlib import redirect_stderr, redirect_stdout
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from norlan import cli, core
from norlan.billing import VIDEO_MODEL, price_snapshot
from norlan.costs import cost_report, format_report, sync_usage
from norlan.providers import ArkProvider, ProviderError


class CostRecordsTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / "synthetic-project"
        core.create_project(self.root, "Synthetic test novel, not a real production.", "live")
        self.config = {"video_model": VIDEO_MODEL, "resolution": "720p"}
        core.configure(self.root, {"ark": self.config}, limit=1000, currency="CNY")

    def job(self, job_id="job-1", *, historical=False, kind="video", accounting="spent", amount=5):
        pricing = None if historical else price_snapshot(self.config, kind)
        core.reserve(self.root, job_id, kind, "synthetic-item", amount, pricing=pricing)
        return core.update_job(self.root, job_id, accounting=accounting, state="completed", task_id="task-" + job_id)

    def bill(self, job_id="job-1", amount=2, **values):
        return {"job_id": job_id, "amount": amount, "currency": "CNY", "bill_reference": "synthetic-bill-" + job_id,
                "scope": "per_call", **values}

    def response(self, **values):
        return {"model": VIDEO_MODEL, "usage": {"completion_tokens": 108108}, "provider_status": "succeeded",
                "resolution": "720p", "service_tier": "default", "request_id": "synthetic-request", **values}

    def test_actual_zero_and_repeat_import_are_idempotent(self):
        self.job()
        first = core.import_bill_costs(self.root, [self.bill(amount=0)])
        second = core.import_bill_costs(self.root, [self.bill(amount=0)])
        self.assertEqual(first["budget"]["actual_cost"], 0)
        self.assertEqual(second["budget"]["actual_cost_jobs"], 1)
        self.assertEqual(second["budget"]["spent"], 5)
        self.assertEqual(len(first["events"]), len(second["events"]))
        self.assertEqual(second["budget"]["cost_pending_jobs"], 0)

    def test_wrong_currency_rolls_back_entire_import(self):
        self.job()
        self.job("job-2")
        before = (self.root / "project.json").read_bytes()
        with self.assertRaises(ValueError):
            core.import_bill_costs(self.root, [self.bill(), self.bill("job-2", currency="USD")])
        self.assertEqual((self.root / "project.json").read_bytes(), before)

    def test_conflicting_or_aggregate_bill_cannot_overwrite(self):
        self.job()
        core.import_bill_costs(self.root, [self.bill()])
        for row in (self.bill(amount=3), self.bill(bill_reference="different-bill"), self.bill(scope="daily_total")):
            with self.subTest(row=row), self.assertRaises((ValueError, core.GateError)):
                core.import_bill_costs(self.root, [row])
        self.assertEqual(core.load_project(self.root)["budget"]["actual_cost"], 2)

    def test_bill_cost_above_cap_increases_exposure_and_lower_bill_cannot_release_cap(self):
        self.job()
        self.job("job-2")
        state = core.import_bill_costs(self.root, [self.bill(amount=2), self.bill("job-2", amount=8)])
        self.assertEqual(state["budget"]["actual_cost"], 10)
        self.assertEqual(state["budget"]["spent"], 13)
        self.assertEqual(state["budget"]["available"], 987)

    def test_existing_task_sync_uses_get_only_and_skips_complete_usage(self):
        self.job()
        raw = {"model": VIDEO_MODEL, "status": "succeeded", "usage": {"completion_tokens": 108108},
               "resolution": "720p", "service_tier": "default", "prompt": "SECRET-RESPONSE"}
        before = core.load_project(self.root)
        with patch.dict(os.environ, {"ARK_API_KEY": "synthetic-key"}), \
             patch.object(ArkProvider, "_request", return_value=(raw, "synthetic-request")) as request, \
             patch.object(ArkProvider, "download") as download, \
             patch.object(ArkProvider, "submit_video") as submit:
            result = sync_usage(self.root)
            self.assertEqual(result, {"synced": 1, "skipped": 0, "unavailable": 0, "errors": []})
            request.assert_called_once_with("GET", "/contents/generations/tasks/task-job-1")
            download.assert_not_called()
            submit.assert_not_called()
            second = sync_usage(self.root)
            self.assertEqual(second["skipped"], 1)
            self.assertEqual(request.call_count, 1)
        after = core.load_project(self.root)
        self.assertEqual(after["budget"]["usage_cost"], 3.999996)
        self.assertIsNone(after["budget"]["actual_cost"])
        for key in ("assets", "episodes", "plan", "status", "paused"):
            self.assertEqual(before[key], after[key])
        self.assertNotIn("SECRET-RESPONSE", (self.root / "project.json").read_text())

    def test_historical_usage_is_saved_without_guessing_price_or_input_video(self):
        self.job(historical=True)
        with patch("norlan.costs.ArkProvider") as provider:
            provider.return_value.video_usage.return_value = self.response()
            self.assertEqual(sync_usage(self.root)["synced"], 1)
        state = core.load_project(self.root)
        billing = state["jobs"]["job-1"]["billing"]
        self.assertEqual(billing["usage"]["completion_tokens"], 108108)
        self.assertEqual(billing["status"], "pricing_missing")
        self.assertIsNone(state["budget"]["usage_cost"])
        self.assertFalse(billing.get("price"))

    def test_explicit_historical_confirmation_uses_reported_model_and_resolution(self):
        self.job(historical=True)
        with patch("norlan.costs.ArkProvider") as provider:
            provider.return_value.video_usage.return_value = self.response()
            result = sync_usage(self.root, confirm_no_video_input=True)
        self.assertEqual(result["synced"], 1)
        billing = core.load_project(self.root)["jobs"]["job-1"]["billing"]
        self.assertEqual(billing["usage_cost"], 3.999996)
        self.assertTrue(billing["price"]["historical_lookup"])
        self.assertEqual(billing["price"]["basis"], "current_public_price_for_historical_usage")
        self.assertIn("不是历史账单费用", format_report(cost_report(self.root)))

    def test_unknown_returned_model_or_resolution_does_not_get_historical_price(self):
        for values in ({"model": "unknown-model"}, {"resolution": None}, {"service_tier": "flex"}):
            with self.subTest(values=values):
                if "job-1" not in core.load_project(self.root)["jobs"]:
                    self.job(historical=True)
                with patch("norlan.costs.ArkProvider") as provider:
                    provider.return_value.video_usage.return_value = self.response(**values)
                    sync_usage(self.root, confirm_no_video_input=True)
                self.assertIsNone(core.load_project(self.root)["budget"]["usage_cost"])

    def test_pending_usage_and_unknown_accounting_are_not_settled(self):
        self.job(accounting="unknown")
        with patch("norlan.costs.ArkProvider") as provider:
            provider.return_value.video_usage.return_value = self.response(provider_status="running")
            sync_usage(self.root)
        state = core.load_project(self.root)
        self.assertEqual(state["jobs"]["job-1"]["accounting"], "unknown")
        self.assertEqual(state["jobs"]["job-1"]["state"], "completed")
        self.assertIsNone(state["budget"]["usage_cost"])
        self.assertEqual(state["budget"]["unknown"], 5)

    def test_image_history_reserved_and_demo_tasks_are_never_queried(self):
        self.job(historical=True, kind="image")
        self.job("job-reserved", accounting="reserved")
        with patch("norlan.costs.ArkProvider") as provider:
            result = sync_usage(self.root)
            provider.assert_not_called()
        self.assertEqual(result["unavailable"], 1)
        self.assertEqual(result["skipped"], 1)
        demo = Path(self.temp.name) / "synthetic-demo"
        core.create_project(demo, "Synthetic demo.", "demo")
        core.reserve(demo, "demo-call", "video", "synthetic", 0)
        core.update_job(demo, "demo-call", accounting="spent", task_id="fake-task")
        with patch("norlan.costs.ArkProvider") as provider:
            self.assertEqual(sync_usage(demo)["skipped"], 1)
            provider.assert_not_called()

    def test_missing_key_and_task_errors_have_fixed_safe_messages(self):
        self.job()
        with patch("norlan.costs.ArkProvider", side_effect=ProviderError("SECRET-KEY")):
            with self.assertRaises(ProviderError) as caught:
                sync_usage(self.root)
        self.assertNotIn("SECRET-KEY", str(caught.exception))
        self.assertIn("方舟 API Key", str(caught.exception))
        with patch("norlan.costs.ArkProvider") as provider:
            provider.return_value.video_usage.side_effect = ProviderError("SECRET-RESPONSE")
            result = sync_usage(self.root)
        self.assertEqual(result["errors"][0]["job_id"], "job-1")
        self.assertNotIn("SECRET-RESPONSE", str(result))

    def test_usage_sync_cannot_erase_verified_bill_currency_or_amount(self):
        self.job(historical=True)
        core.import_bill_costs(self.root, [self.bill(amount=0)])
        with patch("norlan.costs.ArkProvider") as provider:
            provider.return_value.video_usage.return_value = self.response()
            sync_usage(self.root)
        state = core.load_project(self.root)
        self.assertEqual(state["budget"]["actual_cost"], 0)
        self.assertEqual(state["budget"]["actual_cost_jobs"], 1)
        self.assertEqual(state["jobs"]["job-1"]["billing"]["currency"], "CNY")

    def test_cli_cost_json_is_safe_and_does_not_run_production(self):
        self.job()
        with core.transaction(self.root) as state:
            state["settings"]["ark"]["untrusted"] = "SECRET-CONFIG"
            state["jobs"]["job-1"]["raw_response"] = "SECRET-RESPONSE"
        output = io.StringIO()
        with redirect_stdout(output), patch("norlan.pipeline.run") as run, patch("norlan.costs.ArkProvider") as provider:
            code = cli.main(["costs", str(self.root), "--json"])
        self.assertEqual(code, 0)
        run.assert_not_called()
        provider.assert_not_called()
        self.assertNotIn("SECRET", output.getvalue())
        report = json.loads(output.getvalue())
        self.assertEqual(report["budget"]["cost_jobs"], 1)
        self.assertIsNone(report["budget"]["actual_cost"])

    def test_cli_import_json_and_confirmation_requires_sync(self):
        self.job()
        source = Path(self.temp.name) / "synthetic-bill.json"
        source.write_text(json.dumps([self.bill(amount=0)]))
        output, error = io.StringIO(), io.StringIO()
        with redirect_stdout(output), redirect_stderr(error):
            code = cli.main(["costs", str(self.root), "--import-bill", str(source), "--json"])
        self.assertEqual(code, 0)
        self.assertEqual(json.loads(output.getvalue())["budget"]["actual_cost"], 0)
        with redirect_stderr(error):
            code = cli.main(["costs", str(self.root), "--confirm-no-video-input"])
        self.assertEqual(code, 2)
        self.assertIn("必须与 --sync", error.getvalue())


if __name__ == "__main__":
    unittest.main()
