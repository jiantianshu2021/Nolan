"""Synthetic video model HTTP/projection checks; no paid providers or user data."""
import copy
import json
import unittest
from unittest.mock import patch

from norlan import core, dashboard
from tests import test_dashboard_prompt_gate as prompt_gate_fixture


FAST = "doubao-seedance-2-0-fast-260128"
PRO = "doubao-seedance-2-0-260128"
MINI = "doubao-seedance-2-0-mini-260615"
LATEST = "doubao-seedance-2-5-260628"


class DashboardVideoModelApiTests(unittest.TestCase):
    # Reuse setup helpers without inheriting/discovering the fixture's tests.
    setUp_base = prompt_gate_fixture.DashboardPromptGateTests.setUp
    tearDown = prompt_gate_fixture.DashboardPromptGateTests.tearDown
    request = prompt_gate_fixture.DashboardPromptGateTests.request

    def setUp(self):
        self.setUp_base()
        self.core.GateError = core.GateError
        self.dispatches = []
        self.fail_model = None

        def dispatch(name, *args, **kwargs):
            self.dispatches.append((name, args, kwargs))
            if kwargs.get("video_model") == self.fail_model and self.fail_model is not None:
                raise core.GateError("该视频模型不在已核实的模型列表中。")
            return self.state

        self.core.approve_prompt = lambda *args, **kwargs: dispatch("approve", *args, **kwargs)
        self.core.edit_prompt_and_approve = lambda *args, **kwargs: dispatch("edit", *args, **kwargs)
        self.state["settings"] = {"ark": {"video_model": FAST, "resolution": "720p",
                                             "api_key": "SECRET-ARK-KEY"}}

    def payload(self, manual=False, **changes):
        result = {"action": "edit_prompt_and_generate" if manual else "approve_prompt", "id": "S1",
                  "candidate_id": "candidate-one", "request_hash": "a" * 64}
        if manual:
            result["prompt"] = "  人工完整定稿\n保留空格、换行及最后空行。\n\n"
        result.update(changes)
        return result

    def submit(self, manual=False, payload=None, headers=None):
        return self.request("POST", "/api/prompt-edit" if manual else "/api/action",
                            self.payload(manual) if payload is None else payload, headers)

    def test_normal_and_manual_paths_dispatch_selected_model_once_without_running(self):
        before = copy.deepcopy(self.state)
        for manual in (False, True):
            with self.subTest(manual=manual):
                payload = self.payload(manual, video_model=PRO)
                self.assertEqual(self.submit(manual, payload)[0], 200)
                name, args, kwargs = self.dispatches[-1]
                self.assertEqual(name, "edit" if manual else "approve")
                expected = (self.root.resolve(), "S1", "candidate-one", "a" * 64)
                self.assertEqual(args, expected + ((payload["prompt"],) if manual else ()))
                self.assertEqual(kwargs, {"video_model": PRO})
        self.assertEqual(len(self.dispatches), 2)
        self.assertEqual(self.core.calls, [], "No GPT refinement/rejection path is dispatched")
        self.assertEqual(self.state, before)
        self.run.assert_not_called()

    def test_omitting_model_preserves_the_original_call_signature(self):
        for manual in (False, True):
            with self.subTest(manual=manual):
                self.assertEqual(self.submit(manual)[0], 200)
                self.assertEqual(self.dispatches[-1][2], {}, "Legacy clients do not send a new keyword")
                self.assertEqual(len(self.dispatches[-1][1]), 5 if manual else 4)
        self.assertEqual(len(self.dispatches), 2)

    def test_invalid_model_values_never_reach_either_gate(self):
        invalid = [None, True, 1, [], {}, "", " ", "api/model", "with space", "模型",
                   "bad\nmodel", "bad\x00model", "x" * 201, "https://private.invalid/key"]
        for manual in (False, True):
            for model in invalid:
                with self.subTest(manual=manual, model=repr(model)):
                    status, _ = self.submit(manual, self.payload(manual, video_model=model))
                    self.assertEqual(status, 409)
        self.assertEqual(self.dispatches, [])

    def test_unknown_well_formed_model_is_checked_by_core_and_cannot_start_work(self):
        self.fail_model = "synthetic-unknown-model"
        before = copy.deepcopy(self.state)
        for manual in (False, True):
            with self.subTest(manual=manual):
                status, result = self.submit(manual, self.payload(manual, video_model=self.fail_model))
                self.assertEqual(status, 409)
                self.assertTrue(result["error"])
                self.assertEqual(self.dispatches[-1][2], {"video_model": self.fail_model})
        self.assertEqual(len(self.dispatches), 2)
        self.assertEqual(self.state, before)
        self.run.assert_not_called()

    def test_dashboard_and_external_run_locks_block_both_confirmation_paths(self):
        for manual in (False, True):
            with self.subTest(manual=manual, lock="dashboard"):
                self.server.context.running = True
                self.assertEqual(self.submit(manual, self.payload(manual, video_model=PRO))[0], 409)
                self.server.context.running = False
            with self.subTest(manual=manual, lock="external"):
                with patch.object(self.server.context, "external_run_active", return_value=True):
                    self.assertEqual(self.submit(manual, self.payload(manual, video_model=PRO))[0], 409)
        self.assertEqual(self.dispatches, [])

    def test_manual_whitelist_accepts_only_model_addition_and_keeps_csrf_checks(self):
        for extra in ({"settings": {"video_model": PRO}}, {"reference_url": "https://private.invalid/SECRET"},
                      {"duration": 30}, {"approve": True}):
            with self.subTest(extra=next(iter(extra))):
                self.assertEqual(self.submit(True, self.payload(True, video_model=PRO, **extra))[0], 409)
        for manual in (False, True):
            for headers in ({"X-Norlan-Token": "invalid"}, {"Host": "foreign.invalid"},
                            {"Origin": "https://foreign.invalid"}):
                with self.subTest(manual=manual, headers=headers):
                    self.assertEqual(self.submit(manual, self.payload(manual, video_model=PRO), headers)[0], 403)
        self.assertEqual(self.dispatches, [])

    def test_state_returns_safe_candidate_options_and_does_not_write_settings(self):
        candidate = self.state["episodes"][0]["shots"][0]["prompt_candidate"]
        candidate["video_model"] = PRO
        before = copy.deepcopy(self.state)
        status, result = self.request("GET", "/api/state")
        self.assertEqual(status, 200)
        self.assertEqual(result["video_models"]["default_model"], FAST)
        self.assertTrue(any(item["id"] == LATEST for item in result["video_models"]["models"]))
        public = result["episodes"][0]["shots"][0]["prompt_candidate"]
        self.assertEqual(public["video_model"], PRO)
        self.assertTrue(any(option["id"] == PRO and option["available"] for option in public["model_options"]))
        self.assertTrue(any(not option["available"] and option["reason"] for option in public["model_options"]))
        self.assertEqual(public["prompt"], candidate["prompt"])
        text = json.dumps(result)
        self.assertNotIn("SECRET-", text)
        self.assertNotIn("https://private.invalid", text)
        self.assertNotIn("settings", result)
        self.assertNotIn("config", public)
        self.assertEqual(self.state, before)


class DashboardVideoModelProjectionTests(unittest.TestCase):
    def setUp(self):
        self.shot = {"id": "S1", "version": 3, "status": "awaiting_prompt_approval", "duration": 10,
                     "file": "media/E1/S1-v3.mp4", "sha256": "c" * 64, "job_id": "job-current",
                     "video_model": MINI,
                     "prompt_candidate": {"id": "next-candidate", "version": 4,
                         "request_hash": "d" * 64, "prompt": "下一版候选提示词", "video_model": LATEST,
                         "config": {"api_key": "SECRET-CANDIDATE-KEY"},
                         "reference_manifest": {"images": ["https://private.invalid/SECRET-IMAGE"],
                             "audio": [], "context": {}}},
                     "history": [{"version": 1, "file": "media/E1/S1-v1.mp4", "sha256": "a" * 64,
                                      "job_id": "job-old", "video_model": LATEST},
                                 {"version": 2, "file": "media/E1/S1-v2.mp4", "sha256": "b" * 64,
                                      "job_id": "job-price"}]}
        self.state = {"plan": {"status": "approved", "ratio": "16:9"}, "assets": [], "budget": {},
                      "settings": {"ark": {"video_model": FAST, "resolution": "720p", "api_key": "SECRET-KEY",
                                              "base_url": "https://private.invalid/SECRET-BASE"}},
                      "episodes": [{"id": "E1", "status": "generating", "shots": [self.shot]}],
                      "jobs": {
                          "job-current": {"kind": "video", "item_id": "S1", "output": "media/E1/S1-v3.mp4",
                              "video_request": {"video_model": PRO, "key": "SECRET-REQUEST"},
                              "billing": {"price": {"model": MINI}}, "result": {"url": "SECRET-RESULT"}},
                          "job-old": {"kind": "video", "item_id": "S1", "output": "media/E1/S1-v1.mp4",
                              "video_request": {"video_model": FAST}},
                          "job-price": {"kind": "video", "item_id": "S1", "output": "media/E1/S1-v2.mp4",
                              "billing": {"price": {"model": MINI, "api_key": "SECRET-PRICE"}}}}}

    def public_shot(self):
        return dashboard.public_state(self.state)["episodes"][0]["shots"][0]

    def test_current_and_history_use_actual_bound_requests_not_future_selection(self):
        before = copy.deepcopy(self.state)
        result = self.public_shot()
        self.assertEqual(result["generation_model"], PRO, "The request is stronger evidence than a later price or choice")
        self.assertEqual(result["prompt_candidate"]["video_model"], LATEST)
        self.assertEqual([item["generation_model"] for item in result["history"]], [FAST, MINI])
        text = json.dumps(dashboard.public_state(self.state))
        self.assertNotIn("SECRET-", text)
        self.assertNotIn("https://private.invalid", text)
        self.assertEqual(self.state, before)

    def test_missing_or_wrong_job_never_uses_current_or_future_defaults(self):
        self.state["jobs"] = {}
        self.shot["history"][0].pop("video_model")
        result = self.public_shot()
        self.assertNotIn("generation_model", result)
        self.assertTrue(all("generation_model" not in item for item in result["history"]))
        self.state["jobs"]["job-current"] = {"kind": "image", "item_id": "S1",
            "output": "media/E1/S1-v3.mp4", "video_request": {"video_model": FAST}}
        self.assertNotIn("generation_model", self.public_shot())
        self.state["jobs"]["job-current"].update(kind="video", item_id="OTHER")
        self.assertNotIn("generation_model", self.public_shot())
        self.state["jobs"]["job-current"].update(item_id="S1", output="media/E1/S1-v2.mp4")
        self.assertNotIn("generation_model", self.public_shot())

    def test_wrong_historical_job_version_does_not_label_the_old_video(self):
        self.state["jobs"]["job-old"].update(output="media/E1/S1-v4.mp4", video_request={"video_model": LATEST})
        history = self.public_shot()["history"]
        self.assertEqual(history[0]["version"], 1)
        self.assertNotIn("generation_model", history[0],
                         "A mismatched actual job must not fall back to a future stored choice")
        self.assertEqual(history[1]["generation_model"], MINI)

    def test_history_file_version_controls_its_job_binding_when_saved_version_is_stale(self):
        self.shot["history"][0]["version"] = 4
        before = copy.deepcopy(self.state)
        history = self.public_shot()["history"]
        self.assertEqual(history[0]["version"], 1)
        self.assertEqual(history[0]["generation_model"], FAST,
                         "A stale saved version cannot associate the old video with the next model")
        self.assertEqual(self.state, before)

    def test_archived_explicit_model_is_a_fallback_when_the_old_job_was_not_saved(self):
        self.state["jobs"].pop("job-old")
        self.shot["history"][0]["video_model"] = FAST
        self.assertEqual(self.public_shot()["history"][0]["generation_model"], FAST)

    def test_invalid_configured_model_does_not_expose_urls_or_keys_in_catalog(self):
        self.state["settings"]["ark"]["video_model"] = "https://private.invalid/SECRET-KEY?token=SECRET-TOKEN"
        result = dashboard.public_state(self.state)
        self.assertNotIn("SECRET-", json.dumps(result))
        self.assertNotIn("https://private.invalid", json.dumps(result))
        self.assertEqual(result["video_models"]["default_model"], "")


if __name__ == "__main__":
    unittest.main()
