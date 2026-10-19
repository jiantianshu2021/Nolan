"""Synthetic local HTTP intake tests; no model or provider calls are permitted."""
import http.client
import json
from pathlib import Path
import struct
import tempfile
import threading
import unittest
from unittest.mock import Mock, patch
import zlib

from norlan import core, dashboard, pipeline
from tests.test_asset_inputs import data_url, png_data


class DashboardIntakeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name).resolve()
        core.create_project(self.root, "合成 HTTP 测试原文，不能出现在网页接口。")
        core.install_plan(self.root, pipeline.demo_plan())
        self.server = dashboard.make_server(self.root, port=0)
        self.port = self.server.server_address[1]
        self.thread = threading.Thread(target=lambda: self.server.serve_forever(poll_interval=0.02), daemon=True)
        self.thread.start()
        self.run_patch = patch.object(dashboard, "_run", side_effect=AssertionError("intake must never run production"))
        self.run = self.run_patch.start()

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)
        self.run.assert_not_called()
        self.run_patch.stop()
        self.temp.cleanup()

    def request(self, method, path="/api/action", payload=None, headers=None, raw=None):
        connection = http.client.HTTPConnection("127.0.0.1", self.port, timeout=3)
        body = raw if raw is not None else json.dumps(payload).encode() if payload is not None else None
        options = {"Content-Type": "application/json", "X-Norlan-Token": self.server.context.token} if method == "POST" else {}
        options.update(headers or {})
        try:
            connection.request(method, path, body=body, headers=options)
            response = connection.getresponse()
            return response.status, dict(response.getheaders()), response.read()
        finally:
            connection.close()

    def state(self):
        status, _, body = self.request("GET", "/api/state")
        self.assertEqual(status, 200)
        return json.loads(body)

    def approve_plan(self):
        result = self.request("POST", payload={"action": "approve", "kind": "plan", "id": "plan", "version": 1})
        self.assertEqual(result[0], 200)

    def choose(self, revision=0, mode="gpt", description=""):
        return self.request("POST", payload={"action": "asset_input", "id": "hero", "version": 0,
                                             "input_revision": revision, "mode": mode, "description": description})

    def upload(self, revision=0, encoded=None):
        return self.request("POST", "/api/asset-reference", {"id": "hero", "version": 0,
                            "input_revision": revision, "data_url": encoded or data_url()})

    def test_approved_plan_then_choice_is_exposed_without_generation(self):
        self.assertEqual(self.choose()[0], 409)
        self.assertEqual(self.upload()[0], 409)
        self.approve_plan()
        initial = self.state()
        self.assertTrue(initial["assets"][0]["input_required"])
        self.assertTrue(initial["assets"][0]["input_editable"])
        status, _, body = self.choose()
        self.assertEqual(status, 200)
        asset = json.loads(body)["assets"][0]
        self.assertEqual(asset["creation_input"]["mode"], "gpt")
        self.assertEqual(asset["creation_input"]["revision"], 1)
        self.assertTrue(asset["creation_input"]["confirmed"])
        self.assertFalse(asset["input_required"])
        self.assertEqual(asset["status"], "planned")
        self.assertEqual(core.load_project(self.root)["jobs"], {})

    def test_upload_is_displayable_then_delete_revokes_media_access(self):
        self.approve_plan()
        status, _, body = self.upload()
        self.assertEqual(status, 200)
        asset = json.loads(body)["assets"][0]
        ref = asset["creation_input"]["references"][0]
        self.assertEqual(ref["mime"], "image/png")
        self.assertEqual(ref["bytes"], len(png_data()))
        self.assertNotIn("sha256", ref)
        self.assertTrue(asset["input_required"])
        image_status, headers, content = self.request("GET", "/media/" + ref["file"])
        self.assertEqual((image_status, content), (200, png_data()))
        self.assertEqual(headers["Content-Type"], "image/png")
        self.assertEqual(self.choose(1, "custom", "白衬衫")[0], 200)
        deletion = {"action": "remove_asset_reference", "id": "hero", "version": 0,
                    "input_revision": 2, "reference_id": ref["id"]}
        self.assertEqual(self.request("POST", payload=deletion)[0], 200)
        self.assertTrue((self.root / ref["file"]).is_file(), "original upload stays immutable")
        self.assertEqual(self.request("GET", "/media/" + ref["file"])[0], 404)
        self.assertTrue(self.state()["assets"][0]["input_required"])
        self.assertEqual(core.load_project(self.root)["jobs"], {})

    def test_upload_and_actions_enforce_csrf_origin_and_host(self):
        self.approve_plan()
        payloads = [("/api/action", {"action": "asset_input", "id": "hero", "version": 0,
                                    "input_revision": 0, "mode": "gpt", "description": ""}),
                    ("/api/asset-reference", {"id": "hero", "version": 0, "input_revision": 0,
                                              "data_url": data_url()})]
        for path, payload in payloads:
            for headers in ({"X-Norlan-Token": "bad"}, {"X-Norlan-Token": ""},
                            {"Origin": "https://foreign.invalid"}, {"Host": "foreign.invalid"}):
                with self.subTest(path=path, header=list(headers)):
                    self.assertEqual(self.request("POST", path, payload, headers)[0], 403)
        self.assertEqual(self.state()["assets"][0]["creation_input"]["revision"], 0)
        self.assertFalse(list(self.root.glob("inputs/assets/hero/*")))

    def test_stale_revision_bool_version_and_duplicate_upload_conflict(self):
        self.approve_plan()
        self.assertEqual(self.upload()[0], 200)
        self.assertEqual(self.choose(0, "custom", "蓝色短发")[0], 409)
        self.assertEqual(self.upload(0)[0], 409)
        self.assertEqual(self.upload(1)[0], 409)
        malformed = {"action": "asset_input", "id": "hero", "version": True,
                     "input_revision": 1, "mode": "custom", "description": "蓝色短发"}
        self.assertEqual(self.request("POST", payload=malformed)[0], 409)
        asset = self.state()["assets"][0]
        self.assertEqual(asset["creation_input"]["revision"], 1)
        self.assertEqual(len(asset["creation_input"]["references"]), 1)

    def test_bulk_gpt_requires_plan_version_and_keeps_custom_choice(self):
        self.approve_plan()
        self.assertEqual(self.choose(0, "custom", "白色衬衫")[0], 200)
        self.assertEqual(self.request("POST", payload={"action": "asset_inputs_gpt_remaining", "plan_version": 2})[0], 409)
        status, _, body = self.request("POST", payload={"action": "asset_inputs_gpt_remaining", "plan_version": 1})
        self.assertEqual(status, 200)
        hero, station = json.loads(body)["assets"]
        self.assertEqual(hero["creation_input"]["mode"], "custom")
        self.assertEqual(hero["creation_input"]["description"], "白色衬衫")
        self.assertEqual(station["creation_input"]["mode"], "gpt")
        self.assertTrue(station["creation_input"]["confirmed"])

    def test_normal_action_and_upload_have_separate_body_limits(self):
        self.approve_plan()
        action = json.dumps({"action": "pause", "paused": True}).encode()
        self.assertEqual(self.request("POST", raw=action + b" " * (dashboard.MAX_BODY - len(action)))[0], 200)
        self.assertEqual(self.request("POST", raw=b"", headers={"Content-Length": str(dashboard.MAX_BODY + 1)})[0], 413)
        # A valid PNG carrying harmless text exceeds 32 KiB but belongs to the
        # dedicated reference endpoint, whose independent request cap is 12 MiB.
        text = b"comment\0" + b"x" * 40000
        chunk = struct.pack(">I", len(text)) + b"tEXt" + text + struct.pack(">I", zlib.crc32(b"tEXt" + text))
        larger_png = png_data()[:-12] + chunk + png_data()[-12:]
        self.assertGreater(len(data_url(larger_png)), dashboard.MAX_BODY)
        self.assertEqual(self.upload(encoded=data_url(larger_png))[0], 200)
        self.assertEqual(self.request("POST", "/api/asset-reference", raw=b"",
                                      headers={"Content-Length": str(dashboard.MAX_REFERENCE_BODY + 1)})[0], 413)

    def test_upload_boundary_header_reads_at_limit_without_allocating_large_fixture(self):
        self.approve_plan()
        handler = dashboard._Handler.__new__(dashboard._Handler)
        handler.server = self.server
        handler.path = "/api/asset-reference"
        handler.headers = {"Host": f"127.0.0.1:{self.port}", "X-Norlan-Token": self.server.context.token,
                           "Content-Type": "application/json", "Content-Length": str(dashboard.MAX_REFERENCE_BODY)}
        payload = {"id": "hero", "version": 0, "input_revision": 0, "data_url": data_url()}
        handler.rfile = Mock()
        handler.rfile.read.return_value = json.dumps(payload).encode()
        handler._json, handler._error = Mock(), Mock()
        handler.do_POST()
        handler.rfile.read.assert_called_once_with(dashboard.MAX_REFERENCE_BODY)
        handler._error.assert_not_called()
        self.assertEqual(handler._json.call_args.args[0], 200)

    def test_state_exposes_workflow_progress_and_omits_original_source_and_provider_data(self):
        self.approve_plan()
        with core.transaction(self.root) as state:
            state["status"] = "awaiting_asset_input"
            state["settings"]["private_note"] = "SECRET-SETTINGS"
            state["episodes"][0]["provider_response"] = {"secret": "SECRET-PROVIDER"}
        public = self.state()
        progress = public["episodes"][0]["progress"]
        self.assertEqual(progress["percent"], 0)
        self.assertEqual(progress["stage"], "awaiting_asset_input")
        self.assertTrue(progress["waiting_for_human"])
        self.assertIn("不代表视频引擎", progress["basis"])
        text = json.dumps(public, ensure_ascii=False)
        self.assertNotIn("SECRET-", text)
        self.assertNotIn("合成 HTTP 测试原文", text)
        self.assertNotIn("jobs", public)


if __name__ == "__main__":
    unittest.main()
