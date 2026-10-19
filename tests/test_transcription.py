import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from urllib.error import HTTPError, URLError

from norlan.providers import ProviderError
from norlan.transcription import Transcriber, _NoRedirect


class Response(io.BytesIO):
    def __init__(self, data):
        super().__init__(json.dumps(data).encode())
        self.headers = {"X-Request-Id": "req_unit_test"}


class TranscriptionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.audio = Path(self.temp.name) / 'private-name-"-录音.wav'
        self.audio.write_bytes(b"RIFFactual-audio-bytes")
        self.env = patch.dict(os.environ, {"NORLAN_TEST_KEY": "test-unit-key"})
        self.env.start()
        self.probe = patch("norlan.transcription.probe", return_value={"duration": 6.0, "has_audio": True})
        self.probe.start()
        self.transcriber = Transcriber({"api_key_env": "NORLAN_TEST_KEY", "language": "zh"})

    def tearDown(self):
        self.probe.stop()
        self.env.stop()
        self.temp.cleanup()

    def response(self, segments=None, text="实际对白"):
        if segments is None:
            segments = [{"start": 0.2, "end": 2.1, "text": " 实际对白 "}]
        return Response({"duration": 6.0, "text": text, "segments": segments})

    def test_multipart_contains_actual_audio_and_timestamps_without_source_filename(self):
        with patch("norlan.transcription._open", return_value=self.response()) as opened:
            result = self.transcriber.transcribe(self.audio)
        self.assertEqual(result, [{"start": 0.2, "end": 2.1, "text": "实际对白"}])
        request, timeout = opened.call_args.args
        self.assertEqual(request.full_url, "https://api.openai.com/v1/audio/transcriptions")
        self.assertEqual(request.method, "POST")
        self.assertEqual(timeout, 120)
        self.assertEqual(request.get_header("Authorization"), "Bearer test-unit-key")
        self.assertIn("multipart/form-data; boundary=", request.get_header("Content-type"))
        for field in (b'filename="audio.wav"', b'name="timestamp_granularities[]"', b"segment", b"verbose_json", b"whisper-1", b"RIFFactual-audio-bytes", b'name="language"'):
            self.assertIn(field, request.data)
        self.assertNotIn(b"private-name", request.data)
        self.assertNotIn(b'name="prompt"', request.data)
        self.assertEqual(self.transcriber.last_request_id, "req_unit_test")

    def test_invalid_timestamps_and_unaligned_text_are_uncertain(self):
        invalid = [
            [{"start": -1, "end": 2, "text": "x"}],
            [{"start": 1, "end": 8, "text": "x"}],
            [{"start": 1, "end": 1, "text": "x"}],
            [{"start": 3, "end": 4, "text": "x"}, {"start": 2, "end": 3, "text": "y"}],
            [{"start": 0, "end": 3, "text": "x"}, {"start": 2.5, "end": 4, "text": "y"}],
            [{"start": False, "end": 2, "text": "x"}],
            [{"start": 0, "end": float("nan"), "text": "x"}],
            [],
        ]
        for segments in invalid:
            with self.subTest(segments=segments), patch("norlan.transcription._open", return_value=self.response(segments)) as opened:
                with self.assertRaises(ProviderError) as raised:
                    self.transcriber.transcribe(self.audio)
                self.assertTrue(raised.exception.uncertain)
                self.assertEqual(opened.call_count, 1)

    def test_short_boundary_rounding_is_clamped_to_real_duration(self):
        with patch("norlan.transcription._open", return_value=self.response([{"start": 5, "end": 6.02, "text": "结束"}])):
            self.assertEqual(self.transcriber.transcribe(self.audio)[0]["end"], 6.0)

    def test_silence_returns_no_subtitles(self):
        with patch("norlan.transcription._open", return_value=self.response([], text="")):
            self.assertEqual(self.transcriber.transcribe(self.audio), [])

    def test_no_audio_or_missing_file_fails_before_network(self):
        with patch("norlan.transcription._open") as opened, patch("norlan.transcription.probe", return_value={"duration": 6, "has_audio": False}):
            with self.assertRaises(ProviderError) as raised:
                self.transcriber.transcribe(self.audio)
            self.assertFalse(raised.exception.uncertain)
            opened.assert_not_called()
        with patch("norlan.transcription._open") as opened:
            with self.assertRaises(ProviderError):
                self.transcriber.transcribe(self.audio.parent / "missing.wav")
            opened.assert_not_called()

    def test_network_timeout_is_not_retried_and_does_not_leak_raw_error(self):
        with patch("norlan.transcription._open", side_effect=URLError("test-unit-key provider-secret")) as opened:
            with self.assertRaises(ProviderError) as raised:
                self.transcriber.transcribe(self.audio)
        self.assertTrue(raised.exception.uncertain)
        self.assertEqual(opened.call_count, 1)
        self.assertNotIn("test-unit-key", str(raised.exception))
        self.assertNotIn("provider-secret", str(raised.exception))

    def test_http_error_classification_and_no_raw_response(self):
        for code, uncertain in ((401, False), (429, False), (408, True), (500, True), (307, True)):
            with self.subTest(code=code):
                error = HTTPError("https://api.openai.com/v1/audio/transcriptions", code, "secret-error", {"X-Request-Id": "req_error"}, io.BytesIO(b"SECRET_RESPONSE"))
                with patch("norlan.transcription._open", side_effect=error) as opened:
                    with self.assertRaises(ProviderError) as raised:
                        self.transcriber.transcribe(self.audio)
                self.assertEqual(raised.exception.uncertain, uncertain)
                self.assertEqual(raised.exception.request_id, "req_error")
                self.assertNotIn("SECRET_RESPONSE", str(raised.exception))
                self.assertEqual(opened.call_count, 1)

    def test_non_json_response_is_uncertain(self):
        response = io.BytesIO(b"not-json-SECRET")
        response.headers = {}
        with patch("norlan.transcription._open", return_value=response):
            with self.assertRaises(ProviderError) as raised:
                self.transcriber.transcribe(self.audio)
        self.assertTrue(raised.exception.uncertain)
        self.assertNotIn("SECRET", str(raised.exception))

    def test_settings_validate_without_network(self):
        for config in ({"model": "gpt-4o-transcribe"}, {"base_url": "https://evil.example/v1"}, {"timeout": float("nan")}, {"timeout": True}, {"language": "Chinese"}, {"api_key_env": "BAD-NAME"}, {"mode": "local"}):
            with self.subTest(config=config), self.assertRaises(ProviderError):
                Transcriber(config)

    def test_redirect_handler_never_forwards_request(self):
        self.assertIsNone(_NoRedirect().redirect_request(None, None, 307, "", {}, "https://evil.example"))


if __name__ == "__main__":
    unittest.main()
