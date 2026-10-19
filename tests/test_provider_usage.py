"""Metering survives provider-output failures; all HTTP requests are mocked."""

import base64
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from norlan.providers import ArkProvider, ProviderError


IMAGE_MODEL = "doubao-seedream-5-0-flash-260915"
VIDEO_MODEL = "doubao-seedance-2-0-fast-260128"


class Response(io.BytesIO):
    def __init__(self, body):
        super().__init__(json.dumps(body).encode())
        self.headers = {"X-Request-Id": "synthetic-request-1"}


class ProviderUsageTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="norlan-provider-usage-test-")
        self.addCleanup(self.temporary.cleanup)
        self.output = Path(self.temporary.name) / "asset.png"
        environment = patch.dict(os.environ, {"ARK_API_KEY": "synthetic-unit-test-key"})
        environment.start()
        self.addCleanup(environment.stop)
        # Any unmocked download is a test failure, never a network request.
        network = patch("norlan.providers.urlopen", side_effect=AssertionError("network forbidden"))
        network.start()
        self.addCleanup(network.stop)
        self.api = ArkProvider({"image_model": IMAGE_MODEL, "video_model": VIDEO_MODEL})

    def test_download_failure_preserves_success_usage_before_download(self):
        result = {"model": IMAGE_MODEL, "usage": {"generated_images": 1, "output_tokens": 16384},
                  "data": [{"url": "https://example.test/image?signature=not-for-ledger"}]}
        records = []

        def failed_download(*_):
            self.assertEqual(len(records), 1)
            raise ProviderError("synthetic download failure")

        self.output.write_bytes(b"original")
        with patch("norlan.providers._api_open", return_value=Response(result)) as http, \
                patch.object(self.api, "download", side_effect=failed_download):
            with self.assertRaises(ProviderError) as caught:
                self.api.generate_image("synthetic prompt", self.output, on_usage=records.append)
        self.assertTrue(caught.exception.uncertain)
        self.assertEqual(caught.exception.request_id, "synthetic-request-1")
        self.assertEqual(http.call_count, 1)
        self.assertEqual(records, [{"model": IMAGE_MODEL, "usage": result["usage"],
                                    "request_id": "synthetic-request-1", "provider_status": "succeeded"}])
        self.assertEqual(self.output.read_bytes(), b"original")

    def test_decode_failure_still_records_metering_once(self):
        records = []
        result = {"model": IMAGE_MODEL, "usage": {"generated_images": 1},
                  "data": [{"b64_json": "not base64!"}]}
        with patch("norlan.providers._api_open", return_value=Response(result)) as http:
            with self.assertRaises(ProviderError):
                self.api.generate_image("synthetic prompt", self.output, on_usage=records.append)
        self.assertEqual(http.call_count, 1)
        self.assertEqual(records[0]["usage"]["generated_images"], 1)
        self.assertEqual(records[0]["provider_status"], "succeeded")
        self.assertFalse(self.output.exists())

    def test_partial_success_survives_first_item_error(self):
        records = []
        result = {"model": IMAGE_MODEL, "usage": {"generated_images": 1},
                  "data": [{"error": {"code": "Rejected", "message": "private provider detail"}},
                           {"url": "https://example.test/generated.png"}]}
        with patch("norlan.providers._api_open", return_value=Response(result)):
            with self.assertRaises(ProviderError):
                self.api.generate_image("synthetic prompt", self.output, on_usage=records.append)
        self.assertEqual(records[0]["provider_status"], "succeeded")
        self.assertEqual(records[0]["usage"], {"generated_images": 1})
        self.assertNotIn("private provider detail", json.dumps(records))

    def test_explicit_zero_failed_images_are_not_reported_as_success(self):
        for data in ([], [{"error": {"code": "Rejected"}}], [{"error": "invalid error shape"}]):
            with self.subTest(data=data):
                records = []
                result = {"model": IMAGE_MODEL, "usage": {"generated_images": 0}, "data": data}
                with patch("norlan.providers._api_open", return_value=Response(result)):
                    with self.assertRaises(ProviderError):
                        self.api.generate_image("synthetic prompt", self.output, on_usage=records.append)
                self.assertEqual(records[0]["provider_status"], "failed")
                self.assertEqual(records[0]["usage"], {"generated_images": 0})

    def test_missing_evidence_does_not_invent_model_count_or_success(self):
        records = []
        with patch("norlan.providers._api_open", return_value=Response({"data": []})):
            with self.assertRaises(ProviderError):
                self.api.generate_image("synthetic prompt", self.output, on_usage=records.append)
        self.assertIsNone(records[0]["model"])
        self.assertEqual(records[0]["usage"], {})
        self.assertEqual(records[0]["provider_status"], "unknown")

    def test_callback_has_only_metering_fields_and_no_provider_text(self):
        records = []
        result = {"model": "https://example.test/private-model?key=secret", "prompt": "private prompt",
                  "usage": {"generated_images": 1, "input_images": 0, "output_tokens": 25,
                            "total_tokens": True, "secret": "synthetic-secret", "completion_tokens": 9},
                  "data": [{"b64_json": base64.b64encode(b"synthetic image").decode()}]}
        with patch("norlan.providers._api_open", return_value=Response(result)):
            self.api.generate_image("synthetic prompt", self.output, on_usage=records.append)
        self.assertEqual(set(records[0]), {"model", "usage", "request_id", "provider_status"})
        self.assertIsNone(records[0]["model"])
        self.assertEqual(records[0]["usage"], {"generated_images": 1, "input_images": 0, "output_tokens": 25})
        self.assertNotIn("secret", json.dumps(records))

    def test_callback_failure_stops_local_save_without_repeating_post(self):
        result = {"model": IMAGE_MODEL, "usage": {"generated_images": 1},
                  "data": [{"b64_json": base64.b64encode(b"synthetic image").decode()}]}
        with patch("norlan.providers._api_open", return_value=Response(result)) as http, \
                patch("norlan.providers._atomic_bytes") as save:
            def failed_record(_):
                raise RuntimeError("synthetic ledger unavailable")
            with self.assertRaisesRegex(RuntimeError, "ledger unavailable"):
                self.api.generate_image("synthetic prompt", self.output, on_usage=failed_record)
        self.assertEqual(http.call_count, 1)
        save.assert_not_called()

    def test_legacy_image_return_shape_is_preserved(self):
        result = {"model": IMAGE_MODEL, "usage": {"generated_images": 1},
                  "data": [{"b64_json": base64.b64encode(b"synthetic image").decode()}]}
        with patch("norlan.providers._api_open", return_value=Response(result)):
            saved = self.api.generate_image("synthetic prompt", self.output)
        self.assertEqual(saved, {"output": str(self.output), "request_id": "synthetic-request-1",
                                 "usage": {"generated_images": 1}, "model": IMAGE_MODEL,
                                 "provider_status": "succeeded"})

    def test_completed_video_missing_or_invalid_url_retains_metering(self):
        for content in (None, {}, {"video_url": ""}, {"video_url": 17},
                        {"video_url": "http://example.test/video.mp4"}):
            with self.subTest(content=content):
                raw = {"model": VIDEO_MODEL, "status": "succeeded", "content": content,
                       "usage": {"completion_tokens": 108900}, "resolution": "720p", "service_tier": "default"}
                with patch("norlan.providers._api_open", return_value=Response(raw)) as http:
                    result = self.api.poll_video("synthetic-task-1")
                self.assertEqual(http.call_count, 1)
                self.assertEqual(http.call_args.args[0].method, "GET")
                self.assertEqual(result["status"], "completed")
                self.assertIsNone(result["url"])
                self.assertEqual(result["usage"], {"completion_tokens": 108900})
                self.assertEqual(result["model"], VIDEO_MODEL)

    def test_video_optional_fields_remain_absent_and_model_is_not_filled(self):
        for method in (self.api.video_usage, self.api.poll_video):
            with self.subTest(method=method.__name__):
                raw = {"status": "succeeded", "usage": {"completion_tokens": 108900}}
                with patch("norlan.providers._api_open", return_value=Response(raw)):
                    result = method("synthetic-task-1")
                self.assertNotIn("resolution", result)
                self.assertNotIn("service_tier", result)
                self.assertIsNone(result.get("model"))
                raw.update(resolution=None, service_tier=None)
                with patch("norlan.providers._api_open", return_value=Response(raw)):
                    explicit_null = method("synthetic-task-1")
                self.assertIn("resolution", explicit_null)
                self.assertIn("service_tier", explicit_null)
                self.assertIsNone(explicit_null["service_tier"])


if __name__ == "__main__":
    unittest.main()
