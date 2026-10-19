"""Synthetic/offline contracts for official Ark continuation artifacts."""
import hashlib
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from urllib.error import HTTPError

from norlan.providers import ArkProvider, ProviderError


class Response(io.BytesIO):
    def __init__(self, data):
        super().__init__(json.dumps(data).encode())
        self.headers = {"X-Request-Id": "synthetic-request"}


class OfficialProviderTests(unittest.TestCase):
    def setUp(self):
        env = patch.dict(os.environ, {"ARK_API_KEY": "synthetic-api-key"})
        env.start()
        self.addCleanup(env.stop)
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.api = ArkProvider({"video_model": "doubao-seedance-2-0-fast-260128"})

    def test_submit_requests_official_tail_for_every_generation(self):
        for previous in (None, "https://example.test/original.mp4?signature=synthetic"):
            with self.subTest(previous=previous), patch("norlan.providers._api_open", return_value=Response({"id": "synthetic-task"})) as http:
                self.api.submit_video("SYNTHETIC continuation", 12, [], previous_video=previous)
                request = http.call_args.args[0]
                payload = json.loads(request.data)
                self.assertTrue(payload["return_last_frame"])
                self.assertEqual(request.method, "POST")
                self.assertEqual(http.call_count, 1)
                videos = [item for item in payload["content"] if item["type"] == "video_url"]
                self.assertEqual(len(videos), int(previous is not None))
                if videos:
                    self.assertEqual(videos[0], {"type": "video_url", "video_url": {"url": previous}, "role": "reference_video"})

    def test_local_original_uses_exact_hash_mapping_without_transformation(self):
        video = self.root / "original.mp4"
        original = b"SYNTHETIC UNCHANGED ORIGINAL VIDEO"
        video.write_bytes(original)
        digest = hashlib.sha256(original).hexdigest()
        url = "https://example.test/original.mp4"
        self.api.config["reference_video_urls"] = {digest: url}
        with patch("norlan.providers._api_open", return_value=Response({"id": "synthetic-task"})) as http:
            self.api.submit_video("SYNTHETIC continuation", 12, [], previous_video=video)
        payload = json.loads(http.call_args.args[0].data)
        self.assertEqual(payload["content"][1]["video_url"]["url"], url)
        self.assertEqual(video.read_bytes(), original)

    def test_invalid_original_reference_stops_before_post(self):
        missing = self.root / "missing.mp4"
        unmapped = self.root / "unmapped.mp4"
        unmapped.write_bytes(b"SYNTHETIC original")
        invalid_values = [missing, unmapped, self.root / "tail.png",
                          "https://name:password@example.test/clip.mp4",
                          "https://example.test/clip.mp4\n?signature=private",
                          "https://[invalid/clip.mp4"]
        with patch("norlan.providers._api_open") as http:
            for value in invalid_values:
                with self.subTest(value=value), self.assertRaises(ProviderError) as caught:
                    self.api.submit_video("SYNTHETIC", 12, [], previous_video=value)
                self.assertNotIn("password", str(caught.exception))
                self.assertNotIn("signature=private", str(caught.exception))
            http.assert_not_called()

    def test_original_reference_over_200mb_stops_before_read_and_post(self):
        video = self.root / "large.mp4"
        with video.open("wb") as stream:
            stream.truncate(200_000_001)
        with patch("norlan.providers._api_open") as http, patch.object(Path, "read_bytes") as read:
            with self.assertRaisesRegex(ProviderError, "200 MB"):
                self.api.submit_video("SYNTHETIC", 12, [], previous_video=video)
            read.assert_not_called()
            http.assert_not_called()

    def test_existing_task_artifacts_use_one_get_and_expose_official_fields(self):
        video_url = "https://example.test/video.mp4?signature=video-secret"
        tail_url = "https://example.test/tail.jpeg?signature=tail-secret"
        response = {"status": "succeeded", "model": "doubao-seedance-2-0-fast-260128",
                    "content": {"video_url": video_url, "last_frame_url": tail_url},
                    "created_at": 1792431000, "updated_at": 1792431200,
                    "duration": 12, "framespersecond": 24, "ratio": "16:9", "output_format": "mp4",
                    "usage": {"completion_tokens": 100}, "private_extra": "never-return"}
        with patch("norlan.providers._api_open", return_value=Response(response)) as http:
            result = self.api.video_artifacts("synthetic-existing-task")
        self.assertEqual(http.call_count, 1)
        request = http.call_args.args[0]
        self.assertEqual(request.method, "GET")
        self.assertIsNone(request.data)
        self.assertTrue(request.full_url.endswith("/contents/generations/tasks/synthetic-existing-task"))
        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["url"], video_url)
        self.assertEqual(result["video_url"], video_url)
        self.assertEqual(result["last_frame_url"], tail_url)
        self.assertEqual(result["created_at"], response["created_at"])
        self.assertEqual(result["fps"], 24)
        self.assertEqual(result["duration"], 12)
        self.assertNotIn("private_extra", result)

    def test_absent_official_tail_stays_absent_without_creating_or_deriving(self):
        with patch("norlan.providers._api_open", return_value=Response({"status": "succeeded", "content": {"video_url": "https://example.test/original.mp4"}})) as http:
            result = self.api.video_artifacts("synthetic-existing-task")
        self.assertIsNone(result["last_frame_url"])
        self.assertEqual(http.call_count, 1)
        self.assertEqual(http.call_args.args[0].method, "GET")

    def test_invalid_tail_does_not_lose_valid_video_or_metering(self):
        for tail_url in ("http://example.test/tail.jpeg", "https://user:secret@example.test/tail.jpeg", "https://[invalid", {"url": "bad"}):
            data = {"status": "succeeded", "content": {"video_url": "https://example.test/original.mp4", "last_frame_url": tail_url},
                    "usage": {"completion_tokens": 37}, "created_at": True, "updated_at": "1792431200",
                    "ratio": [], "duration": float("nan"), "output_format": {}}
            with self.subTest(tail_url=tail_url), patch("norlan.providers._api_open", return_value=Response(data)):
                result = self.api.poll_video("synthetic-task")
            self.assertEqual(result["status"], "completed")
            self.assertIsNone(result["last_frame_url"])
            self.assertEqual(result["url"], "https://example.test/original.mp4")
            self.assertEqual(result["usage"], {"completion_tokens": 37})
            for field in ("created_at", "updated_at", "ratio", "duration", "output_format"):
                self.assertNotIn(field, result)

    def test_existing_task_failure_preserves_diagnostic_without_signed_links(self):
        data = {"status": "failed", "error": {"code": "InputVideoSensitiveContentDetected.PrivacyInformation",
                "message": "SYNTHETIC reference rejected: https://example.test/video.mp4?signature=private"}}
        with patch("norlan.providers._api_open", return_value=Response(data)) as http:
            result = self.api.video_artifacts("synthetic-failed-task")
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["failure"]["provider_code"], data["error"]["code"])
        self.assertEqual(result["failure"]["request_id"], "synthetic-request")
        self.assertNotIn("signature=private", json.dumps(result))
        self.assertNotIn("video_url", result)
        self.assertEqual(http.call_args.args[0].method, "GET")

    def test_read_error_does_not_create_video_or_hide_failure(self):
        body = io.BytesIO(json.dumps({"error": {"code": "NotFound", "message": "SYNTHETIC task expired"}}).encode())
        error = HTTPError("https://example.test/task", 404, "", {"X-Request-Id": "synthetic-404"}, body)
        with patch("norlan.providers._api_open", side_effect=error) as http:
            with self.assertRaises(ProviderError) as caught:
                self.api.video_artifacts("synthetic-expired-task")
        self.assertEqual(http.call_count, 1)
        self.assertEqual(http.call_args.args[0].method, "GET")
        self.assertEqual(caught.exception.failure["http_status"], 404)
        self.assertEqual(caught.exception.failure["provider_code"], "NotFound")
        self.assertFalse(caught.exception.uncertain)

    def test_invalid_task_id_cannot_make_a_request(self):
        with patch("norlan.providers._api_open") as http:
            for task_id in (None, "", "../other", "task?signature=private"):
                with self.subTest(task_id=task_id), self.assertRaises(ProviderError):
                    self.api.video_artifacts(task_id)
            http.assert_not_called()


if __name__ == "__main__":
    unittest.main()
