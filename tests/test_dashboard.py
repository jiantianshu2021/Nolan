import http.client
import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

from norlan import dashboard


class FakeCore:
    def __init__(self, state):
        self.state = state
        self.approvals = []

    def load_project(self, _root):
        return self.state

    def approve(self, _root, kind, item_id, version):
        if version != 1:
            raise ValueError("sensitive-secret-should-not-leak")
        self.approvals.append((kind, item_id, version))
        return self.state

    def reject(self, *_args):
        return self.state

    def set_paused(self, _root, paused):
        self.state["paused"] = paused
        return self.state

    def request_caption_fix(self, _root, item_id, version, feedback):
        self.state["caption_request"] = {"id": item_id, "version": version, "feedback": feedback}
        return self.state


class DashboardTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        (self.root / "media").mkdir()
        (self.root / "media" / "clip.mp4").write_bytes(b"0123456789")
        (self.root / "novel.txt").write_text("private novel")
        (self.root / "media" / "unreferenced.png").write_bytes(b"private")
        self.state = {
            "title": "测试短剧", "mode": "demo", "status": "awaiting_review",
            "plan": {"status": "candidate", "version": 1, "style": "漫画", "ratio": "9:16", "secret": "SECRET"},
            "settings": {"api_key": "SECRET"}, "novel": "SECRET",
            "budget": {"limit": 10, "spent": 1, "reserved": 2, "unknown": 0, "currency": "CNY", "secret": "SECRET"},
            "assets": [], "episodes": [{"id": "E001", "version": 1, "file": "media/clip.mp4", "status": "awaiting_review", "provider_result": {"key": "SECRET"}, "shots": []}],
        }
        self.core = FakeCore(self.state)
        self.patch = patch.object(dashboard, "_core", return_value=self.core)
        self.patch.start()
        self.server = dashboard.make_server(self.root, port=0)
        self.port = self.server.server_address[1]
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)
        self.patch.stop()
        self.temp.cleanup()

    def request(self, method, path, payload=None, headers=None):
        connection = http.client.HTTPConnection("127.0.0.1", self.port, timeout=3)
        body = json.dumps(payload) if payload is not None else None
        all_headers = {}
        if body is not None:
            all_headers["Content-Type"] = "application/json"
            all_headers["X-Norlan-Token"] = self.server.context.token
        all_headers.update(headers or {})
        connection.request(method, path, body=body, headers=all_headers)
        response = connection.getresponse()
        result = response.status, dict(response.getheaders()), response.read()
        connection.close()
        return result

    def test_page_contains_session_and_restrictive_csp(self):
        status, headers, body = self.request("GET", "/")
        self.assertEqual(status, 200)
        self.assertIn(self.server.context.token.encode(), body)
        self.assertIn("frame-ancestors 'none'", headers["Content-Security-Policy"])
        self.assertNotIn(b"https://", body)

    def test_state_redacts_private_fields(self):
        status, _, body = self.request("GET", "/api/state")
        self.assertEqual(status, 200)
        self.assertNotIn(b"SECRET", body)
        self.assertIn("测试短剧", body.decode())

    def test_host_rebinding_blocked(self):
        status, _, _ = self.request("GET", "/api/state", headers={"Host": f"evil.example:{self.port}"})
        self.assertEqual(status, 403)

    def test_csrf_and_cross_origin_blocked(self):
        payload = {"action": "approve", "kind": "episode", "id": "E001", "version": 1}
        status, _, _ = self.request("POST", "/api/action", payload, {"X-Norlan-Token": "incorrect"})
        self.assertEqual(status, 403)
        status, _, _ = self.request("POST", "/api/action", payload, {"Origin": "https://evil.example"})
        self.assertEqual(status, 403)
        self.assertEqual(self.core.approvals, [])

    def test_approval_requires_current_version(self):
        payload = {"action": "approve", "kind": "episode", "id": "E001", "version": 1}
        status, _, _ = self.request("POST", "/api/action", payload)
        self.assertEqual(status, 200)
        payload["version"] = 2
        status, _, body = self.request("POST", "/api/action", payload)
        self.assertEqual(status, 409)
        self.assertNotIn(b"sensitive", body)
        self.assertEqual(self.core.approvals, [("episode", "E001", 1)])

    def test_bool_is_not_a_version(self):
        status, _, _ = self.request("POST", "/api/action", {"action": "approve", "kind": "episode", "id": "E001", "version": True})
        self.assertEqual(status, 409)

    def test_plan_and_shot_approvals_are_supported(self):
        for kind, item_id in (("plan", "plan"), ("shot", "E001-S001")):
            status, _, _ = self.request("POST", "/api/action", {"action": "approve", "kind": kind, "id": item_id, "version": 1})
            self.assertEqual(status, 200)
        self.assertEqual(self.core.approvals, [("plan", "plan", 1), ("shot", "E001-S001", 1)])

    def test_only_referenced_media_is_served(self):
        for path in ("/media/../novel.txt", "/media/%2e%2e/novel.txt", "/media/media/unreferenced.png", "/media/project.json"):
            self.assertEqual(self.request("GET", path)[0], 404)
        self.assertEqual(self.request("GET", "/media/media/clip.mp4")[2], b"0123456789")

    def test_symlinks_are_blocked_even_when_referenced(self):
        (self.root / "media" / "link.mp4").symlink_to(self.root / "media" / "clip.mp4")
        self.state["episodes"][0]["file"] = "media/link.mp4"
        self.assertEqual(self.request("GET", "/media/media/link.mp4")[0], 404)

    def test_video_range_and_head(self):
        status, headers, body = self.request("GET", "/media/media/clip.mp4", headers={"Range": "bytes=2-5"})
        self.assertEqual((status, body), (206, b"2345"))
        self.assertEqual(headers["Content-Range"], "bytes 2-5/10")
        self.assertEqual(self.request("GET", "/media/media/clip.mp4", headers={"Range": "bytes=-3"})[2], b"789")
        self.assertEqual(self.request("GET", "/media/media/clip.mp4", headers={"Range": "bytes=20-30"})[0], 416)
        self.assertEqual(self.request("GET", "/media/media/clip.mp4", headers={"Range": "bytes=0-1,5-7"})[0], 416)
        status, headers, body = self.request("HEAD", "/media/media/clip.mp4")
        self.assertEqual((status, body, headers["Content-Length"]), (200, b"", "10"))

    def test_request_content_type_and_size(self):
        payload = {"action": "pause", "paused": True}
        self.assertEqual(self.request("POST", "/api/action", payload, {"Content-Type": "text/plain"})[0], 415)
        self.assertEqual(self.request("POST", "/api/action", {"large": "x" * 40_000})[0], 413)
        self.assertEqual(self.request("GET", "/api/action?action=run")[0], 404)

    def test_subtitle_correction_and_explicit_subtitle_download(self):
        (self.root / "media" / "captions.srt").write_text("1\n00:00:00,000 --> 00:00:01,000\n你好\n", encoding="utf-8")
        episode = self.state["episodes"][0]
        episode.update(subtitle_file="media/captions.srt", subtitle_status="asr", music_status="absent", render_manifest={"subtitles": [{"text": "你好"}]}, review_note="音频尚未人工核对")
        status, _, body = self.request("GET", "/api/state")
        exposed = json.loads(body)["episodes"][0]
        self.assertTrue(exposed["has_subtitles"])
        self.assertEqual(exposed["review_note"], "音频尚未人工核对")
        status, headers, body = self.request("GET", "/media/media/captions.srt")
        self.assertEqual(status, 200)
        self.assertIn("attachment", headers["Content-Disposition"])
        status, _, _ = self.request("POST", "/api/action", {"action": "captions", "kind": "episode", "id": "E001", "version": 1, "feedback": "把你好改成您好"})
        self.assertEqual(status, 200)
        self.assertEqual(self.state["caption_request"]["version"], 1)

    def test_parallel_runs_are_refused(self):
        release, started = threading.Event(), threading.Event()
        def run(_root):
            started.set()
            release.wait(timeout=3)
            return self.state
        with patch.object(dashboard, "_run", side_effect=run):
            self.assertEqual(self.request("POST", "/api/action", {"action": "run"})[0], 202)
            self.assertTrue(started.wait(timeout=1))
            self.assertEqual(self.request("POST", "/api/action", {"action": "run"})[0], 409)
            self.assertEqual(self.request("POST", "/api/action", {"action": "pause", "paused": True})[0], 200)
            release.set()


if __name__ == "__main__":
    unittest.main()
