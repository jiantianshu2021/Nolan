from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, patch
from urllib.error import URLError

from norlan.music import MusicProvider, _audio_url, _signed_headers, _signing_key
from norlan.providers import ProviderError


class MusicTests(unittest.TestCase):
    def test_official_signing_key_vector(self):
        # Public Volcengine signature-method example, not an account credential.
        documented_example = "WkRZeE1EQmxPVGhsWWpWak5HVmtNbUUxTXpZeU9UVXlOMlE1TmpZeVlqTQ=="
        result = _signing_key(documented_example, "20250329", "cn-beijing", "billing")
        self.assertEqual(result.hex(), "b491ed164936de3bb06c1eb23326aa9587b5aaa6a4e02144b9d523bbebb7ca9f")

    def test_request_signs_body_and_utc_without_exposing_secret(self):
        now = datetime(2026, 10, 19, 12, 30, 1, tzinfo=timezone.utc)
        body = '{"Text":"中文音乐"}'.encode()
        headers = _signed_headers(body, "GenBGMForTime", "test-ak", "test-secret", now)
        self.assertEqual(headers["x-date"], "20261019T123001Z")
        self.assertEqual(headers["x-content-sha256"], hashlib.sha256(body).hexdigest())
        self.assertIn("20261019/cn-beijing/imagination/request", headers["authorization"])
        self.assertNotIn("test-secret", str(headers))
        changed = _signed_headers(body + b" ", "GenBGMForTime", "test-ak", "test-secret", now)
        self.assertNotEqual(headers["authorization"], changed["authorization"])

    def test_submit_pins_budget_duration_and_does_not_accept_override(self):
        provider = MusicProvider()
        with patch.object(provider, "_call", return_value=({"TaskID": "12345"}, "req")) as call:
            result = provider.submit("安静钢琴纯音乐，剧情后半段渐强", 45)
            self.assertEqual(result["id"], "12345")
            payload = call.call_args.args[1]
            self.assertEqual(payload["Segments"], [{"Name": "inst", "Duration": 45}])
            self.assertTrue(payload["AigcWatermark"])
            self.assertTrue(call.call_args.kwargs["generating"])
            call.assert_called_once()
        provider = MusicProvider({"segments": [{"Name": "intro", "Duration": 50}]})
        with self.assertRaises(ProviderError):
            provider.submit("中文提示词", 45)
        for invalid in (True, -1, 4, 121, 45.5, float("nan")):
            with self.assertRaises(ProviderError):
                MusicProvider().submit("中文提示词", invalid)

    def test_poll_maps_documented_status_and_preserves_original_id(self):
        provider = MusicProvider()
        for value in (0, 1):
            with patch.object(provider, "_call", return_value=({"TaskID": "123", "Status": value}, "req")):
                self.assertEqual(provider.poll("123")["status"], "pending")
        with patch.object(provider, "_call", return_value=({"TaskID": "123", "Status": 3, "FailureReason": {"Code": 300061}}, "req")):
            self.assertEqual(provider.poll("123")["status"], "failed")
        final = {"TaskID": "123", "Status": 2, "SongDetail": {"AudioUrl": "https://v26-default.douyinvod.com/song.wav", "Duration": 45.02}}
        with patch.object(provider, "_call", return_value=(final, "req")) as call:
            result = provider.poll("123")
            self.assertEqual(result["duration"], 45.02)
            self.assertEqual(result["status"], "completed")
            self.assertEqual(call.call_args.args, ("QuerySong", {"TaskID": "123"}))

    def test_lost_submission_is_uncertain_and_never_retried(self):
        opener = MagicMock()
        opener.open.side_effect = URLError("secret prompt must not leak")
        with patch.dict(os.environ, {"VOLC_ACCESS_KEY_ID": "test-ak", "VOLC_SECRET_ACCESS_KEY": "test-secret"}), patch("norlan.music.build_opener", return_value=opener):
            with self.assertRaises(ProviderError) as failure:
                MusicProvider().submit("中文配乐", 30)
            self.assertTrue(failure.exception.uncertain)
            self.assertNotIn("secret", str(failure.exception))
            opener.open.assert_called_once()
            opener.open.reset_mock()
            with self.assertRaises(ProviderError) as failure:
                MusicProvider().poll("123")
            self.assertFalse(failure.exception.uncertain)
            opener.open.assert_called_once()

    def test_invalid_download_destinations_rejected(self):
        for url in ("http://v1-default.douyinvod.com/a", "https://localhost/a", "file:///etc/passwd", "https://v1-default.douyinvod.com.evil.example/a", "https://user:pw@v1-default.douyinvod.com/a", "https://v51-default.douyinvod.com/a"):
            with self.assertRaises(ProviderError):
                _audio_url(url)

    def test_download_is_atomic_and_does_not_send_api_auth(self):
        response = MagicMock()
        response.geturl.return_value = "https://v2-default.douyinvod.com/a.wav"
        response.read.side_effect = [b"example-audio", b""]
        response.__enter__.return_value = response
        opener = MagicMock()
        opener.open.return_value = response
        with tempfile.TemporaryDirectory() as directory, patch("norlan.music.build_opener", return_value=opener):
            output = Path(directory) / "music.wav"
            MusicProvider().download("https://v2-default.douyinvod.com/a.wav", output)
            self.assertEqual(output.read_bytes(), b"example-audio")
            request = opener.open.call_args.args[0]
            self.assertFalse(any(key.lower() == "authorization" for key in request.headers))


if __name__ == "__main__":
    unittest.main()
