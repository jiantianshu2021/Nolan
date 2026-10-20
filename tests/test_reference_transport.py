"""Synthetic local reference fixtures and mocked HTTP; no real provider calls."""

import hashlib
import io
import json
import os
from pathlib import Path
import random
import shutil
import struct
import subprocess
import tempfile
import unittest
from unittest.mock import patch
import wave
import zlib

from norlan import media
from norlan.providers import ArkProvider, ProviderError


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def synthetic_png(path, width, height, *, alpha=False):
    """Real, deterministic incompressible pixels force the transport size path."""
    pixels = random.Random(103 if alpha else 97).randbytes(width * height * (4 if alpha else 3))
    row = width * (4 if alpha else 3)
    filtered = b"".join(b"\0" + pixels[offset:offset + row] for offset in range(0, len(pixels), row))

    def chunk(kind, data):
        return (struct.pack(">I", len(data)) + kind + data
                + struct.pack(">I", zlib.crc32(kind + data) & 0xffffffff))

    path.write_bytes(b"\x89PNG\r\n\x1a\n"
                     + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 6 if alpha else 2, 0, 0, 0))
                     + chunk(b"IDAT", zlib.compress(filtered)) + chunk(b"IEND", b""))
    return pixels


@unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "FFmpeg is required")
class ReferenceTransportMediaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fixtures = tempfile.TemporaryDirectory(prefix="nolan-synthetic-reference-fixtures-")
        cls.fixture_root = Path(cls.fixtures.name)
        cls.image = cls.fixture_root / "synthetic-opaque.png"
        synthetic_png(cls.image, 1056, 1000)
        cls.alpha_image = cls.fixture_root / "synthetic-alpha.png"
        rgba = synthetic_png(cls.alpha_image, 896, 896, alpha=True)
        cls.expected_alpha = rgba[3::4]
        cls.audio = cls.fixture_root / "synthetic-stereo.wav"
        with wave.open(str(cls.audio), "wb") as output:
            output.setnchannels(2)
            output.setsampwidth(4)
            output.setframerate(192000)
            output.writeframes(struct.pack("<ii", 2 ** 28, -(2 ** 28)) * (4 * 192000))
        cls.originals = {path: digest(path) for path in (cls.image, cls.alpha_image, cls.audio)}

    @classmethod
    def tearDownClass(cls):
        cls.fixtures.cleanup()

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="nolan-synthetic-reference-cache-")
        self.root = Path(self.temporary.name).resolve()
        self.cache = self.root / "transport"

    def tearDown(self):
        for path, expected in self.originals.items():
            self.assertEqual(digest(path), expected, "Approved source bytes must remain immutable")
        self.temporary.cleanup()

    def test_large_opaque_png_uses_jpeg_without_changing_dimensions_or_source(self):
        self.assertGreater(self.image.stat().st_size, 3_000_000)
        images, sounds, manifest = media.prepare_video_references([self.image], [], self.cache)
        self.assertEqual(sounds, [])
        self.assertEqual(len(images), 1)
        self.assertEqual(images[0].suffix, ".jpg")
        actual = media.probe(images[0])
        self.assertEqual((actual["width"], actual["height"]), (1056, 1000))
        self.assertEqual(actual["video_codec"], "mjpeg")
        self.assertLessEqual(images[0].stat().st_size, 3_000_000)
        item = manifest["images"][0]
        self.assertEqual(item["source_sha256"], self.originals[self.image])
        self.assertEqual(item["sha256"], digest(images[0]))
        self.assertTrue(item["derived"])
        self.assertEqual(item["strategy"]["quality"], 2)
        self.assertIsNone(item["strategy"]["long_edge_limit"])
        self.assertEqual(manifest["source_bytes"], self.image.stat().st_size)
        self.assertEqual(manifest["transport_bytes"], images[0].stat().st_size)

    def test_large_transparent_png_preserves_every_alpha_sample(self):
        self.assertGreater(self.alpha_image.stat().st_size, 3_000_000)
        images, _, manifest = media.prepare_video_references([self.alpha_image], [], self.cache)
        actual = media.probe(images[0])
        self.assertEqual((actual["width"], actual["height"]), (896, 896))
        self.assertIn(images[0].suffix, (".png", ".webp"))
        # Independently decode alpha, rather than trusting manifest flags or
        # the same ffprobe predicate used by the production helper.
        result = subprocess.run([
            "ffmpeg", "-v", "error", "-nostdin", "-protocol_whitelist", "file,pipe",
            "-i", str(images[0]), "-vf", "alphaextract", "-frames:v", "1",
            "-f", "rawvideo", "-pix_fmt", "gray", "pipe:1",
        ], capture_output=True, check=True, timeout=60)
        self.assertEqual(result.stdout, self.expected_alpha)
        self.assertTrue(manifest["images"][0]["alpha_preserved"])
        self.assertLessEqual(images[0].stat().st_size, 3_000_000)

    def test_oversized_audio_preserves_duration_channels_and_content_sign(self):
        self.assertGreater(self.audio.stat().st_size, 3_000_000)
        images, audio, manifest = media.prepare_video_references([], [self.audio], self.cache)
        self.assertEqual(images, [])
        self.assertEqual(len(audio), 1)
        with wave.open(str(audio[0]), "rb") as output:
            self.assertEqual(output.getnchannels(), 2)
            self.assertEqual(output.getsampwidth(), 2)
            self.assertEqual(output.getframerate(), 48000)
            self.assertEqual(output.getnframes() / output.getframerate(), 4)
            left, right = struct.unpack("<hh", output.readframes(1))
            self.assertGreater(left, 0)
            self.assertLess(right, 0)
        item = manifest["audio"][0]
        self.assertEqual(item["duration"], 4)
        self.assertEqual(item["source_channels"], item["channels"])
        self.assertFalse(item["strategy"]["trimmed"])
        self.assertFalse(item["strategy"]["loudness_changed"])
        self.assertLessEqual(audio[0].stat().st_size, 3_000_000)

    def test_small_references_are_passed_through_without_transcoding(self):
        small = self.root / "synthetic-small.png"
        synthetic_png(small, 16, 16)
        with patch("norlan.media._ffmpeg", side_effect=AssertionError("No encoding is needed")):
            images, _, manifest = media.prepare_video_references([small], [], self.cache)
        self.assertEqual(images, [small])
        self.assertFalse(manifest["images"][0]["derived"])
        self.assertEqual(manifest["images"][0]["source_sha256"], digest(small))
        self.assertFalse(self.cache.exists())

    def test_cache_is_reused_and_identical_bytes_name_the_current_source(self):
        first = media.prepare_video_references([self.image], [], self.cache)
        before = (first[0][0].stat().st_mtime_ns, first[0][0].read_bytes())
        other = self.root / "synthetic-same-approved-bytes.png"
        shutil.copyfile(self.image, other)
        with patch("norlan.media._ffmpeg", side_effect=AssertionError("Cache must not reencode")):
            again = media.prepare_video_references([self.image], [], self.cache)
            moved = media.prepare_video_references([other], [], self.cache)
        self.assertEqual(first, again)
        self.assertEqual(first[0], moved[0])
        self.assertEqual(moved[2]["images"][0]["source_path"], str(other.resolve()))
        self.assertEqual(moved[2]["images"][0]["source_sha256"], digest(other))
        self.assertEqual(before, (first[0][0].stat().st_mtime_ns, first[0][0].read_bytes()))

    def test_changed_cache_is_rejected_without_reencoding_or_overwriting(self):
        images, _, _ = media.prepare_video_references([self.image], [], self.cache)
        damaged = bytearray(images[0].read_bytes())
        damaged[-1] ^= 1
        images[0].write_bytes(damaged)
        with patch("norlan.media._ffmpeg", side_effect=AssertionError("Do not replace changed cache")):
            with self.assertRaisesRegex(media.MediaError, "缓存已改变或不完整"):
                media.prepare_video_references([self.image], [], self.cache)
        self.assertEqual(images[0].read_bytes(), damaged)

    def test_orphaned_identical_cache_recovers_marker_without_replacing_file(self):
        first = media.prepare_video_references([self.image], [], self.cache)
        target = first[0][0]
        before = (target.stat().st_mtime_ns, target.read_bytes())
        marker = next(self.cache.glob("*.json"))
        marker.unlink()  # Synthetic crash after file publication, before manifest publication.
        again = media.prepare_video_references([self.image], [], self.cache)
        self.assertEqual(first, again)
        self.assertTrue(marker.is_file())
        self.assertEqual(before, (target.stat().st_mtime_ns, target.read_bytes()))

    def test_animated_reference_is_rejected_instead_of_taking_the_first_frame(self):
        animation = self.root / "synthetic-two-frames.gif"
        subprocess.run([
            "ffmpeg", "-v", "error", "-nostdin", "-y", "-f", "lavfi",
            "-i", "testsrc2=size=32x32:rate=2", "-t", "1", str(animation),
        ], capture_output=True, check=True, timeout=30)
        original = digest(animation)
        with self.assertRaisesRegex(media.MediaError, "动画参考图"):
            media.prepare_video_references([animation], [], self.cache)
        self.assertEqual(digest(animation), original)
        self.assertFalse(self.cache.exists())


class SyntheticResponse(io.BytesIO):
    def __init__(self):
        super().__init__(b'{"id":"synthetic-no-provider-task"}')
        self.headers = {}


class ReferenceTransportProviderTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="nolan-synthetic-wire-")
        self.addCleanup(self.temporary.cleanup)
        self.reference = Path(self.temporary.name) / "synthetic-transport.png"
        self.reference.write_bytes(b"synthetic mocked image bytes")
        environment = patch.dict(os.environ, {"ARK_API_KEY": "synthetic-not-a-real-api-key"})
        environment.start()
        self.addCleanup(environment.stop)
        self.api = ArkProvider({"video_model": "doubao-seedance-2-0-fast-260128"})

    def test_preflight_counts_exact_utf8_base64_json_and_hash_without_http(self):
        with patch("norlan.providers._api_open") as http:
            payload = self.api._video_payload("合成 UTF-8 测试", 5, [self.reference])
            expected = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            result = self.api.preflight_video_request("合成 UTF-8 测试", 5, [self.reference])
            http.assert_not_called()
        self.assertEqual(result["request_bytes"], len(expected))
        self.assertEqual(result["request_limit_bytes"], 64_000_000)
        self.assertEqual(result["payload_sha256"], hashlib.sha256(expected).hexdigest())
        self.assertEqual(set(result), {"request_bytes", "request_limit_bytes", "payload_sha256"})

    def test_exact_limit_is_rejected_and_one_byte_more_limit_passes_without_http(self):
        expected_size = self.api.preflight_video_request("合成边界", 5, [self.reference])["request_bytes"]
        with patch("norlan.providers._api_open") as http:
            with patch("norlan.providers.REQUEST_LIMIT_BYTES", expected_size):
                with self.assertRaisesRegex(ProviderError, "64 MB") as caught:
                    self.api.preflight_video_request("合成边界", 5, [self.reference])
                self.assertFalse(caught.exception.uncertain)
                with self.assertRaises(ProviderError):
                    self.api.submit_video("合成边界", 5, [self.reference])
            with patch("norlan.providers.REQUEST_LIMIT_BYTES", expected_size + 1):
                self.assertEqual(self.api.preflight_video_request("合成边界", 5, [self.reference])["request_bytes"],
                                 expected_size)
            http.assert_not_called()

    def test_changed_reference_after_preflight_stops_before_any_http(self):
        prepared = self.api.preflight_video_request("合成哈希约束", 5, [self.reference])
        self.reference.write_bytes(b"synthetic changed transport bytes")
        with patch("norlan.providers._api_open") as http:
            with self.assertRaisesRegex(ProviderError, "预检的内容不一致") as caught:
                self.api.submit_video("合成哈希约束", 5, [self.reference],
                                      expected_payload_sha256=prepared["payload_sha256"])
            self.assertFalse(caught.exception.uncertain)
            http.assert_not_called()

    def test_expected_preflight_hash_matches_actual_mocked_wire_bytes(self):
        prepared = self.api.preflight_video_request("合成一致性", 5, [self.reference])
        with patch("norlan.providers._api_open", return_value=SyntheticResponse()) as http:
            result = self.api.submit_video("合成一致性", 5, [self.reference],
                                           expected_payload_sha256=prepared["payload_sha256"])
        self.assertEqual(result["id"], "synthetic-no-provider-task")
        self.assertEqual(http.call_count, 1)
        sent = http.call_args.args[0].data
        self.assertEqual(len(sent), prepared["request_bytes"])
        self.assertEqual(hashlib.sha256(sent).hexdigest(), prepared["payload_sha256"])


if __name__ == "__main__":
    unittest.main()
