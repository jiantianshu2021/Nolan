"""Synthetic error payloads only: no provider requests or paid generations."""
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from urllib.error import HTTPError

from norlan.failures import exception_failure, sanitize_failure
from norlan.providers import ArkProvider, ProviderError


class SyntheticResponse(io.BytesIO):
    def __init__(self, body, headers=None):
        super().__init__(json.dumps(body).encode())
        self.headers = headers or {}


class FailureDetailsTests(unittest.TestCase):
    def setUp(self):
        self.env = patch.dict(os.environ, {"ARK_API_KEY": "synthetic-secret-key-12345"})
        self.env.start()
        self.addCleanup(self.env.stop)
        self.api = ArkProvider({"video_model": "synthetic-video", "image_model": "synthetic-image"})

    def test_http_rejection_records_only_safe_fields_and_never_retries(self):
        prompt = "这是不能出现在日志中的完整测试剧情"
        payload = {"error": {"code": "InvalidParameter.AudioDuration", "param": "content[2].audio_url",
                             "message": "The total audio duration 18s exceeds the limit 15s; " + prompt,
                             "debug": "private internals"},
                   "request_id": "body-request-1", "payload": {"prompt": prompt}}
        error = HTTPError("https://synthetic.test", 400, "untrusted error string", {"X-Request-Id": "header-request-1"},
                          io.BytesIO(json.dumps(payload).encode()))
        with patch("norlan.providers._api_open", side_effect=error) as http:
            with self.assertRaises(ProviderError) as caught:
                self.api.submit_video(prompt, 5, [])
        detail = caught.exception.failure
        self.assertEqual(http.call_count, 1)
        self.assertEqual(detail["stage"], "submission")
        self.assertEqual(detail["http_status"], 400)
        self.assertEqual(detail["request_id"], "header-request-1")
        self.assertEqual(detail["provider_code"], "InvalidParameter.AudioDuration")
        self.assertEqual(detail["parameter"], "content[2].audio_url")
        self.assertIn("18s exceeds the limit 15s", detail["provider_message"])
        self.assertFalse(detail["uncertain"])
        self.assertNotIn(prompt, json.dumps(detail, ensure_ascii=False))
        self.assertNotIn("private internals", json.dumps(detail))
        self.assertNotIn("untrusted error string", str(caught.exception))

    def test_http_body_is_bounded_and_non_json_body_never_persisted(self):
        class TrackingBody(io.BytesIO):
            def read(self, size=-1):
                reads.append(size)
                return super().read(size)
        reads = []
        body = TrackingBody(b"secret HTML signed response" * 1000)
        error = HTTPError("https://synthetic.test", 503, "private", {}, body)
        with patch("norlan.providers._api_open", side_effect=error):
            with self.assertRaises(ProviderError) as caught:
                self.api.submit_video("synthetic prompt", 5, [])
        self.assertEqual(reads, [16_384])
        self.assertTrue(caught.exception.failure["uncertain"])
        self.assertEqual(caught.exception.failure["http_status"], 503)
        self.assertNotIn("provider_message", caught.exception.failure)
        self.assertNotIn("secret HTML", json.dumps(caught.exception.failure))
        self.assertTrue(body.closed)

    def test_missing_header_uses_body_request_id_and_bad_fields_are_dropped(self):
        body = {"error": {"code": {"secret": "object"}, "message": ["unexpected"],
                          "param": "https://private.test/?token=private"}, "request_id": "synthetic-request-2"}
        error = HTTPError("https://synthetic.test", 429, "private", None, io.BytesIO(json.dumps(body).encode()))
        with patch("norlan.providers._api_open", side_effect=error):
            with self.assertRaises(ProviderError) as caught:
                self.api.submit_video("synthetic", 5, [])
        self.assertEqual(caught.exception.request_id, "synthetic-request-2")
        self.assertNotIn("provider_code", caught.exception.failure)
        self.assertNotIn("provider_message", caught.exception.failure)
        self.assertNotIn("parameter", caught.exception.failure)
        self.assertFalse(caught.exception.uncertain)

    def test_error_body_read_failure_preserves_original_http_outcome(self):
        class BrokenBody(io.BytesIO):
            def read(self, size=-1):
                raise RuntimeError("private transport details")
        for status, uncertain in ((400, False), (503, True)):
            error = HTTPError("https://synthetic.test", status, "private", {"X-Request-Id": "safe-request"}, BrokenBody())
            with self.subTest(status=status), patch("norlan.providers._api_open", side_effect=error):
                with self.assertRaises(ProviderError) as caught:
                    self.api.submit_video("synthetic", 5, [])
            self.assertEqual(caught.exception.failure["http_status"], status)
            self.assertEqual(caught.exception.uncertain, uncertain)
            self.assertEqual(caught.exception.request_id, "safe-request")
            self.assertNotIn("private", json.dumps(caught.exception.failure))

    def test_get_http_error_is_poll_and_never_uncertain_submission(self):
        error = HTTPError("https://synthetic.test", 500, "private", {"X-Tt-Logid": "synthetic-log-3"},
                          io.BytesIO(b'{"error":{"code":"InternalError","message":"Temporarily unavailable"}}'))
        with patch("norlan.providers._api_open", side_effect=error) as http:
            with self.assertRaises(ProviderError) as caught:
                self.api.poll_video("synthetic-task")
        self.assertEqual(caught.exception.failure["stage"], "poll")
        self.assertEqual(caught.exception.request_id, "synthetic-log-3")
        self.assertFalse(caught.exception.uncertain)
        self.assertEqual(http.call_count, 1)
        self.assertEqual(http.call_args.args[0].method, "GET")

    def test_generation_failure_keeps_specific_provider_cause(self):
        payload = {"status": "failed", "error": {"code": "SensitiveContent", "message": "Input image rejected by safety policy.",
                                                      "param": "content[1].image_url"}}
        with patch("norlan.providers._api_open", return_value=SyntheticResponse(payload, {"X-Request-Id": "generation-request"})):
            result = self.api.poll_video("synthetic-task")
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["error"], "SensitiveContent")
        self.assertEqual(result["failure"]["stage"], "generation")
        self.assertEqual(result["failure"]["request_id"], "generation-request")
        self.assertEqual(result["failure"]["provider_message"], "Input image rejected by safety policy.")

    def test_missing_task_id_and_parse_errors_remain_uncertain(self):
        for response in (SyntheticResponse({}), io.BytesIO(b"not JSON")):
            if not hasattr(response, "headers"):
                response.headers = {}
            with self.subTest(response=response), patch("norlan.providers._api_open", return_value=response) as http:
                with self.assertRaises(ProviderError) as caught:
                    self.api.submit_video("synthetic", 5, [])
                self.assertTrue(caught.exception.uncertain)
                self.assertTrue(caught.exception.failure["uncertain"])
                self.assertEqual(caught.exception.failure["stage"], "submission")
                self.assertEqual(http.call_count, 1)

    def test_secrets_signed_urls_base64_prompt_echo_and_extra_fields_are_removed(self):
        detail = sanitize_failure({"stage": "generation", "message": "生成失败", "unknown": "private",
                                   "provider_message": "duration exceeds 15s; Authorization=Bearer bearer-private "
                                   "api_key=unconfigured-private; token='another-private'; "
                                   "synthetic-secret-key-12345 https://private.test/video?signature=signed-private "
                                   "data:audio/wav;base64,QUJDREVGRw== " + "A" * 200 + " prompt='private-story'",
                                   "provider_code": "synthetic-secret-key-12345", "request_id": "safe-request"})
        serialized = json.dumps(detail, ensure_ascii=False)
        for secret in ("private-story", "bearer-private", "unconfigured-private", "another-private",
                       "synthetic-secret-key-12345", "signed-private", "QUJDREVGRw==", "A" * 80):
            self.assertNotIn(secret, serialized)
        self.assertIn("duration exceeds 15s", detail["provider_message"])
        self.assertNotIn("unknown", detail)
        self.assertNotIn("provider_code", detail)
        self.assertEqual(detail["request_id"], "safe-request")
        self.assertEqual(sanitize_failure(detail), detail)

    def test_unknown_exception_does_not_record_exception_message(self):
        detail = exception_failure(ValueError("private prompt and password"), "download")
        self.assertIn("ValueError", detail["message"])
        self.assertNotIn("private", json.dumps(detail))
        self.assertEqual(detail["stage"], "download")

    def test_local_validation_preserved_for_submission_and_contextualized_for_recovery(self):
        exc = ProviderError("参考文件缺失。")
        self.assertEqual(exception_failure(exc, "submission")["stage"], "local_validation")
        self.assertEqual(exception_failure(exc, "poll")["stage"], "poll")
        self.assertEqual(exception_failure(exc, "download")["stage"], "download")
        explicit = ProviderError("已明确的错误", stage="submission", uncertain=True, request_id="safe-request")
        self.assertEqual(exception_failure(explicit, "poll")["stage"], "submission")
        self.assertTrue(exception_failure(explicit, "poll")["uncertain"])

    def test_download_http_failure_records_status_without_signed_url(self):
        error = HTTPError("https://synthetic.test/file?signature=private", 403, "private", {"X-Request-Id": "download-request"},
                          io.BytesIO(b"private response"))
        with tempfile.TemporaryDirectory() as temp, patch("norlan.providers.urlopen", side_effect=error):
            output = Path(temp) / "video.mp4"
            output.write_bytes(b"old-video")
            with self.assertRaises(ProviderError) as caught:
                self.api.download("https://synthetic.test/file?signature=private", output)
            self.assertEqual(output.read_bytes(), b"old-video")
        self.assertEqual(caught.exception.failure["stage"], "download")
        self.assertEqual(caught.exception.failure["http_status"], 403)
        self.assertEqual(caught.exception.request_id, "download-request")
        self.assertNotIn("private", json.dumps(caught.exception.failure))


if __name__ == "__main__":
    unittest.main()
