"""Synthetic video-review setting/API checks; no models or real projects."""
import copy
import http.client
import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

from norlan import core, dashboard, pipeline


class VideoReviewSettingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / "synthetic-project"
        core.create_project(self.root, "合成测试小说，不调用任何供应商。")
        core.install_plan(self.root, pipeline.demo_plan())

    def test_new_demo_and_live_projects_default_to_disabled(self):
        for mode in ("demo", "live"):
            with self.subTest(mode=mode):
                root = Path(self.temp.name) / ("synthetic-default-" + mode)
                state = core.create_project(root, "合成测试原文。", mode=mode)
                self.assertIs(state["settings"]["gpt_video_review_enabled"], False)
                self.assertIs(dashboard.public_state(state)["video_review"]["enabled"], False)

    def test_legacy_missing_setting_is_disabled_and_can_be_persisted(self):
        with core.transaction(self.root) as state:
            state["settings"].pop("gpt_video_review_enabled", None)
        legacy = core.load_project(self.root)
        self.assertIs(dashboard.public_state(legacy)["video_review"]["enabled"], False)
        enabled = core.set_video_review_enabled(self.root, True)
        self.assertIs(enabled["settings"]["gpt_video_review_enabled"], True)
        self.assertIs(dashboard.public_state(core.load_project(self.root))["video_review"]["enabled"], True)
        disabled = core.set_video_review_enabled(self.root, False)
        self.assertIs(disabled["settings"]["gpt_video_review_enabled"], False)
        self.assertIs(core.load_project(self.root)["settings"]["gpt_video_review_enabled"], False)

    def test_setter_rejects_non_boolean_without_mutating_state(self):
        for value in (None, 0, 1, "false", "true", [], {}, 0.0):
            with self.subTest(value=value):
                before = (self.root / "project.json").read_bytes()
                with self.assertRaises(ValueError):
                    core.set_video_review_enabled(self.root, value)
                self.assertEqual((self.root / "project.json").read_bytes(), before)

    def test_configure_accepts_only_boolean_and_rolls_back_invalid_values(self):
        core.configure(self.root, {"gpt_video_review_enabled": True})
        self.assertIs(core.load_project(self.root)["settings"]["gpt_video_review_enabled"], True)
        for value in (None, 0, 1, "false", {}, []):
            with self.subTest(value=value):
                before = (self.root / "project.json").read_bytes()
                with self.assertRaises(ValueError):
                    core.configure(self.root, {"gpt_video_review_enabled": value})
                self.assertEqual((self.root / "project.json").read_bytes(), before)
        core.configure(self.root, {"gpt_video_review_enabled": False})
        self.assertIs(core.load_project(self.root)["settings"]["gpt_video_review_enabled"], False)

    def test_toggle_during_run_preserves_reports_approvals_jobs_and_budget(self):
        core.configure(self.root, limit=2000)
        core.reserve(self.root, "synthetic-review", "llm", "review", 0)
        core.update_job(self.root, "synthetic-review", state="completed", accounting="spent")
        with core.transaction(self.root) as state:
            state.update(status="review_required", paused=True)
            episode = state["episodes"][0]
            episode["reviews"] = [{"key": "synthetic-cut", "issues": ["保留的旧拼接报告"]}]
            shot = episode["shots"][0]
            shot.update(status="accepted", version=2, approved_version=2,
                        approved_at="synthetic-human-approval", approved_sha256="a" * 64,
                        review={"issues": ["保留的旧片段报告"], "version": 2, "sha256": "a" * 64},
                        history=[{"version": 1, "review": {"issues": ["历史报告"]}}])
        before = core.load_project(self.root)
        with core.run_lock(self.root):
            core.set_video_review_enabled(self.root, True)
            core.set_video_review_enabled(self.root, False)
        after = core.load_project(self.root)
        for key in ("plan", "assets", "episodes", "jobs", "budget", "status", "paused", "llm_calls"):
            self.assertEqual(after[key], before[key], key)
        settings = copy.deepcopy(before["settings"])
        settings["gpt_video_review_enabled"] = False
        self.assertEqual(after["settings"], settings)
        self.assertIs(dashboard.public_state(after)["video_review"]["enabled"], False)

    def test_disabled_review_is_rejected_before_creating_a_job_or_using_a_call(self):
        core.set_video_review_enabled(self.root, False)
        before = (self.root / "project.json").read_bytes()
        with self.assertRaises(core.VideoReviewDisabled):
            core.reserve(self.root, "synthetic-blocked-review", "llm", "review", 0,
                         require_video_review=True)
        self.assertEqual((self.root / "project.json").read_bytes(), before)
        self.assertEqual(core.load_project(self.root)["jobs"], {})
        self.assertEqual(core.load_project(self.root)["llm_calls"], 0)

    def test_enabled_review_can_reserve_and_disabling_does_not_cancel_it(self):
        core.set_video_review_enabled(self.root, True)
        core.reserve(self.root, "synthetic-started-review", "llm", "review", 0,
                     require_video_review=True)
        started = core.load_project(self.root)["jobs"]["synthetic-started-review"]
        core.set_video_review_enabled(self.root, False)
        self.assertEqual(core.load_project(self.root)["jobs"]["synthetic-started-review"], started)
        core.update_job(self.root, "synthetic-started-review", state="completed", accounting="spent")
        finished = core.load_project(self.root)["jobs"]["synthetic-started-review"]
        self.assertEqual((finished["state"], finished["accounting"]), ("completed", "spent"))

    def test_disabled_review_does_not_block_other_gpt_work(self):
        core.set_video_review_enabled(self.root, False)
        core.reserve(self.root, "synthetic-planning", "llm", "refine_shot", 0)
        core.update_job(self.root, "synthetic-planning", state="completed", accounting="spent")
        state = core.load_project(self.root)
        self.assertEqual(state["llm_calls"], 1)
        self.assertEqual(state["jobs"]["synthetic-planning"]["state"], "completed")


class VideoReviewSettingApiTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / "synthetic-api-project"
        core.create_project(self.root, "合成 API 测试，不调用模型。")
        self.server = dashboard.make_server(self.root, port=0)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.addCleanup(self.close_server)

    def close_server(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)

    def request(self, payload, headers=None):
        port = self.server.server_address[1]
        connection = http.client.HTTPConnection("127.0.0.1", port, timeout=3)
        all_headers = {"Content-Type": "application/json", "X-Norlan-Token": self.server.context.token}
        all_headers.update(headers or {})
        connection.request("POST", "/api/action", json.dumps(payload), all_headers)
        response = connection.getresponse()
        status, body = response.status, json.loads(response.read())
        connection.close()
        return status, body

    def test_api_persists_setting_and_returns_public_state_without_starting_production(self):
        with patch.object(dashboard, "_run") as run:
            for enabled in (True, False):
                with self.subTest(enabled=enabled):
                    status, body = self.request({"action": "set_video_review", "enabled": enabled})
                    self.assertEqual(status, 200)
                    self.assertIs(body["video_review"]["enabled"], enabled)
                    self.assertIs(core.load_project(self.root)["settings"]["gpt_video_review_enabled"], enabled)
            run.assert_not_called()
        self.assertEqual(core.load_project(self.root)["jobs"], {})
        self.assertEqual(core.load_project(self.root)["llm_calls"], 0)

    def test_api_toggle_is_available_while_dashboard_worker_is_running(self):
        self.server.context.running = True
        with patch.object(dashboard, "_run") as run:
            status, body = self.request({"action": "set_video_review", "enabled": True})
            self.assertEqual(status, 200)
            self.assertIs(body["video_review"]["enabled"], True)
            self.assertTrue(self.server.context.running)
            run.assert_not_called()

    def test_api_toggle_is_available_while_external_production_holds_run_lock(self):
        with core.run_lock(self.root):
            status, body = self.request({"action": "set_video_review", "enabled": True})
            self.assertEqual(status, 200)
            self.assertIs(body["video_review"]["enabled"], True)
        self.assertEqual(core.load_project(self.root)["status"], "initialized")

    def test_api_rejects_missing_non_boolean_or_extra_fields_without_mutation(self):
        payloads = [{"action": "set_video_review"},
                    *({"action": "set_video_review", "enabled": value}
                      for value in (None, 0, 1, "false", [], {})),
                    {"action": "set_video_review", "enabled": True, "approve": True}]
        for payload in payloads:
            with self.subTest(payload=payload):
                before = (self.root / "project.json").read_bytes()
                status, _ = self.request(payload)
                self.assertEqual(status, 409)
                self.assertEqual((self.root / "project.json").read_bytes(), before)

    def test_api_toggle_retains_existing_csrf_and_origin_protection(self):
        payload = {"action": "set_video_review", "enabled": True}
        before = (self.root / "project.json").read_bytes()
        for headers in ({"X-Norlan-Token": "synthetic-wrong-token"}, {"Origin": "https://example.invalid"}):
            with self.subTest(headers=headers):
                status, _ = self.request(payload, headers)
                self.assertEqual(status, 403)
                self.assertEqual((self.root / "project.json").read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
