"""Synthetic local PCM fixtures; no providers or network calls."""

import hashlib
import json
from pathlib import Path
import shutil
import struct
import tempfile
import unittest
from unittest.mock import patch
import wave

from norlan import media


@unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "FFmpeg is required")
class VoiceReferenceTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="norlan-synthetic-voice-test-")
        self.root = Path(self.temporary.name).resolve()
        self.output = self.root / "derived"

    def tearDown(self):
        self.temporary.cleanup()

    def voice(self, name, duration, sample_rate=24000):
        path = self.root / name
        sample_count = round(duration * sample_rate)
        # A repeating synthetic ramp makes an accidental time stretch, offset,
        # or concatenation visible in the exact PCM prefix assertion below.
        cycle = b"".join(struct.pack("<h", index * 101 - 10000) for index in range(199))
        samples = (cycle * (sample_count // 199 + 1))[:sample_count * 2]
        with wave.open(str(path), "wb") as output:
            output.setnchannels(1)
            output.setsampwidth(2)
            output.setframerate(sample_rate)
            output.writeframes(samples)
        return path

    def digest(self, path):
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def test_exactly_fifteen_seconds_preserves_original_files(self):
        paths = [self.voice("approved-a.wav", 7.5), self.voice("approved-b.wav", 7.5)]
        original_bytes = [path.read_bytes() for path in paths]
        with patch("norlan.media._ffmpeg", side_effect=AssertionError("no encoding needed")):
            result, metadata = media.prepare_voice_references(paths, self.output)
        self.assertEqual(result, paths)
        self.assertFalse(self.output.exists())
        self.assertEqual([path.read_bytes() for path in paths], original_bytes)
        for item, path in zip(metadata, paths):
            self.assertFalse(item["derived"])
            self.assertEqual(item["trim_start"], 0)
            self.assertEqual(item["trim_duration"], 7.5)
            self.assertEqual(item["source_sha256"], item["sha256"])
            self.assertEqual(item["sha256"], self.digest(path))

    def test_two_eight_second_samples_are_unmodified_prefix_excerpts(self):
        paths = [self.voice("approved-a.wav", 8), self.voice("approved-b.wav", 8)]
        original_bytes = [path.read_bytes() for path in paths]
        approval = self.root / "approval.json"
        approval.write_text('{"approved":true}', encoding="utf-8")
        result, metadata = media.prepare_voice_references(paths, self.output)
        self.assertEqual([path.read_bytes() for path in paths], original_bytes)
        self.assertEqual(approval.read_text(), '{"approved":true}')
        self.assertLessEqual(sum(item["trim_duration"] for item in metadata), 14.9)
        for source, target, item in zip(paths, result, metadata):
            self.assertNotEqual(source, target)
            self.assertTrue(item["derived"])
            self.assertEqual(item["trim_duration"], 7.45)
            self.assertEqual(item["source_sha256"], self.digest(source))
            self.assertEqual(item["sha256"], self.digest(target))
            self.assertIn(item["source_sha256"], target.name)
            with wave.open(str(source), "rb") as original, wave.open(str(target), "rb") as trimmed:
                self.assertEqual(original.getframerate(), trimmed.getframerate())
                count = trimmed.getnframes()
                self.assertEqual(trimmed.readframes(count), original.readframes(count))

    def test_three_provider_samples_with_aac_padding_fit_limit(self):
        paths = [self.voice(f"approved-{i}.wav", 5.088) for i in range(3)]
        result, metadata = media.prepare_voice_references(paths, self.output)
        self.assertEqual(len(result), 3)
        self.assertLessEqual(sum(media.probe(path)["duration"] for path in result), 15.0)
        self.assertTrue(all(item["trim_duration"] == 4.966 for item in metadata))

    def test_fractional_sample_duration_never_exceeds_allocation(self):
        paths = [self.voice(f"approved-{i}.wav", 5.088, 44100) for i in range(3)]
        _, metadata = media.prepare_voice_references(paths, self.output)
        self.assertTrue(all(2 <= item["trim_duration"] <= 4.966 for item in metadata))
        self.assertLessEqual(sum(item["trim_duration"] for item in metadata), 14.9)

    def test_single_long_sample_and_changed_source_get_distinct_derivatives(self):
        source = self.voice("approved.wav", 15.088)
        first, original_metadata = media.prepare_voice_references([source], self.output)
        self.assertEqual(original_metadata[0]["trim_duration"], 14.9)
        original_derivative = first[0].read_bytes()
        changed = bytearray(source.read_bytes())
        changed[-1] ^= 1
        source.write_bytes(changed)
        second, updated_metadata = media.prepare_voice_references([source], self.output)
        self.assertNotEqual(first, second)
        self.assertNotEqual(original_metadata[0]["source_sha256"], updated_metadata[0]["source_sha256"])
        self.assertEqual(first[0].read_bytes(), original_derivative)

    def test_shorter_approved_sample_is_preserved_when_other_requires_trim(self):
        short = self.voice("short.wav", 2)
        long = self.voice("long.wav", 14)
        result, metadata = media.prepare_voice_references([short, long], self.output)
        self.assertEqual(result[0], short)
        self.assertFalse(metadata[0]["derived"])
        self.assertEqual(metadata[1]["trim_duration"], 7.45)

    def test_cached_derivatives_are_verified_and_not_reencoded(self):
        paths = [self.voice("a.wav", 8), self.voice("b.wav", 9)]
        first_paths, first_metadata = media.prepare_voice_references(paths, self.output)
        before = [(path.stat().st_mtime_ns, path.read_bytes()) for path in first_paths]
        with patch("norlan.media._ffmpeg", side_effect=AssertionError("cache must be reused")):
            again_paths, again_metadata = media.prepare_voice_references(paths, self.output)
        self.assertEqual((again_paths, again_metadata), (first_paths, first_metadata))
        self.assertEqual([(path.stat().st_mtime_ns, path.read_bytes()) for path in first_paths], before)

    def test_changed_cached_audio_is_rejected_not_overwritten(self):
        paths = [self.voice("a.wav", 8), self.voice("b.wav", 9)]
        result, _ = media.prepare_voice_references(paths, self.output)
        original = paths[0].read_bytes()
        changed = bytearray(result[0].read_bytes())
        changed[-1] ^= 1
        result[0].write_bytes(changed)
        with self.assertRaisesRegex(ValueError, "缓存已改变"):
            media.prepare_voice_references(paths, self.output)
        self.assertEqual(result[0].read_bytes(), changed)
        self.assertEqual(paths[0].read_bytes(), original)

    def test_changed_manifest_duration_and_missing_manifest_are_rejected(self):
        paths = [self.voice("a.wav", 8), self.voice("b.wav", 9)]
        result, _ = media.prepare_voice_references(paths, self.output)
        manifest = result[0].with_suffix(".json")
        data = json.loads(manifest.read_text())
        data["trim_duration"] = 2
        manifest.write_text(json.dumps(data))
        with self.assertRaisesRegex(ValueError, "缓存已改变"):
            media.prepare_voice_references(paths, self.output)
        manifest.unlink()
        with self.assertRaisesRegex(ValueError, "缓存已改变"):
            media.prepare_voice_references(paths, self.output)

    def test_empty_and_invalid_sources_fail_before_generation(self):
        self.assertEqual(media.prepare_voice_references([], self.output), ([], []))
        good = self.voice("good.wav", 3)
        short = self.voice("too-short.wav", 1.999)
        invalid = self.root / "invalid.wav"
        invalid.write_text("synthetic invalid media")
        for paths in ([good] * 4, [short], [invalid], [self.root / "missing.wav"]):
            with self.subTest(paths=paths), patch("norlan.media._ffmpeg") as encode:
                with self.assertRaises(ValueError):
                    media.prepare_voice_references(paths, self.output)
                encode.assert_not_called()


if __name__ == "__main__":
    unittest.main()
