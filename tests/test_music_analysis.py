"""Explicitly synthetic local audio fixtures; no model or paid service calls."""
import array
import hashlib
import json
import math
from pathlib import Path
import shutil
import struct
import tempfile
import unittest
from unittest.mock import patch
import wave

from norlan import media, music_analysis
from norlan.music_analysis import (
    MusicAnalysisError, POLICY_VERSION, analyze_track, catalog_analysis, normalization_filter,
)


@unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "FFmpeg required")
class MusicAnalysisTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix="nolan-SYNTHETIC-music-analysis-")
        self.root = Path(self.directory.name)
        self.cache = self.root / "cache"

    def tearDown(self):
        self.directory.cleanup()

    def synthetic(self, name="SYNTHETIC music.wav", levels=((2, 0), (4, .025),
                                                            (4, .1), (4, .3)),
                  antiphase=False):
        path = self.root / name
        rate = 16000
        with wave.open(str(path), "wb") as target:
            target.setnchannels(2)
            target.setsampwidth(2)
            target.setframerate(rate)
            for duration, amplitude in levels:
                block = array.array("h")
                for i in range(round(duration * rate)):
                    value = round(32767 * amplitude * math.sin(2 * math.pi * 400 * i / rate))
                    block.extend((value, -value if antiphase else value))
                target.writeframes(block.tobytes())
        return path

    def test_measured_regions_skip_silent_intro_and_describe_energy_only(self):
        source = self.synthetic()
        before = source.read_bytes()
        result = analyze_track(source, self.cache)
        self.assertEqual(result["sha256"], hashlib.sha256(before).hexdigest())
        self.assertEqual(result["policy_version"], POLICY_VERSION)
        self.assertEqual(result["duration"], 14)
        self.assertEqual(result["sample_step"], .5)
        self.assertLess(result["envelope"][0]["rms_dbfs"], -60)
        self.assertGreater(result["envelope"][-1]["rms_dbfs"], -15)
        self.assertGreaterEqual(len(result["regions"]), 2)
        self.assertLessEqual(len(result["regions"]), 4)
        for region in result["regions"]:
            self.assertGreaterEqual(region["start"], 2)
            self.assertLess(region["start"], region["end"])
            self.assertLessEqual(region["end"], result["duration"])
            self.assertAlmostEqual(region["available_duration"], region["end"] - region["start"])
        self.assertIn("不代表试听", result["limitations"])
        self.assertEqual(source.read_bytes(), before)

    def test_stereo_antiphase_is_not_mistaken_for_silence(self):
        source = self.synthetic(levels=((4, .1),), antiphase=True)
        report = analyze_track(source)
        self.assertGreater(report["mean_dbfs"], -25)
        self.assertTrue(report["regions"])

    def test_cache_reuses_same_bytes_and_invalidates_changed_source(self):
        source = self.synthetic()
        first = analyze_track(source, self.cache)
        with patch.object(music_analysis, "_envelope", side_effect=AssertionError("cache miss")):
            self.assertEqual(analyze_track(source, self.cache), first)
        self.synthetic(levels=((14, .15),))
        second = analyze_track(source, self.cache)
        self.assertNotEqual(first["sha256"], second["sha256"])
        self.assertEqual(len(list(self.cache.glob("track-*.json"))), 2)

    def test_invalid_or_old_policy_cache_cannot_supply_evidence(self):
        source = self.synthetic()
        report = analyze_track(source, self.cache)
        cached = next(self.cache.glob("track-*.json"))
        data = json.loads(cached.read_text())
        data["identity"]["policy_version"] = "retired-policy"
        cached.write_text(json.dumps(data))
        with patch.object(music_analysis, "_envelope", wraps=music_analysis._envelope) as measure:
            self.assertEqual(analyze_track(source, self.cache), report)
            measure.assert_called_once()
        cached.write_text('{"identity": NaN}')
        self.assertEqual(analyze_track(source, self.cache), report)

    def test_compact_catalog_excludes_envelope_and_checks_approved_hash(self):
        source = self.synthetic()
        digest = hashlib.sha256(source.read_bytes()).hexdigest()
        report = catalog_analysis({"id": "synthetic", "sha256": digest}, source, self.cache)
        self.assertNotIn("envelope", report)
        self.assertLess(len(json.dumps(report, ensure_ascii=False)), 1800)
        self.assertLessEqual(len(report["regions"]), 4)
        with self.assertRaisesRegex(MusicAnalysisError, "已批准"):
            catalog_analysis({"sha256": "0" * 64}, source, self.cache)

    def test_silence_and_nonfinite_samples_are_not_amplified(self):
        source = self.synthetic(levels=((2, 0),))
        with self.assertRaisesRegex(MusicAnalysisError, "静音"):
            analyze_track(source, self.cache)
        with self.assertRaisesRegex(MusicAnalysisError, "静音"):
            normalization_filter(source, 0, 2, cache_dir=self.cache)
        # Explicitly synthetic corrupt-decoder fixture, not a production audio.
        def synthetic_nonfinite(args):
            Path(args[-1]).write_bytes(struct.pack("<ff", float("nan"), .1))
        with patch.object(media, "_ffmpeg", side_effect=synthetic_nonfinite):
            with self.assertRaisesRegex(MusicAnalysisError, "非有限"):
                analyze_track(source)

    def test_normalization_measures_actual_excerpt_and_reaches_target_when_safe(self):
        source = self.synthetic(levels=((2, .02), (3, .2)))
        before = source.read_bytes()
        quiet = normalization_filter(source, 0, 2, cache_dir=self.cache)
        loud = normalization_filter(source, 2, 3, cache_dir=self.cache)
        self.assertGreater(quiet["gain_db"] - loud["gain_db"], 19)
        self.assertAlmostEqual(quiet["output_lufs_estimate"], -18, places=2)
        self.assertLessEqual(quiet["output_true_peak_estimate"], -2)
        self.assertFalse(quiet["peak_limited"])
        self.assertRegex(quiet["filter"], r"^volume=-?[0-9]+\.[0-9]+dB$")
        rendered = self.root / "SYNTHETIC normalized.wav"
        media._ffmpeg(["-i", str(source), "-t", "2", "-af", quiet["filter"], str(rendered)])
        verification = normalization_filter(rendered, 0, 2)
        self.assertAlmostEqual(verification["input_lufs"], -18, delta=.15)
        self.assertEqual(source.read_bytes(), before)

    def test_peak_ceiling_takes_priority_and_ineligible_measurements_are_rejected(self):
        # Synthetic sparse impulses with large crest factor: reaching -18 LUFS
        # would exceed true peak. Preserve dynamics rather than adding limiting.
        source = self.root / "SYNTHETIC impulses.wav"
        with wave.open(str(source), "wb") as output:
            output.setnchannels(2)
            output.setsampwidth(2)
            output.setframerate(16000)
            samples = array.array("h")
            for i in range(16000 * 4):
                value = 28000 if i % 4000 == 0 else 0
                samples.extend((value, value))
            output.writeframes(samples.tobytes())
        result = normalization_filter(source, 0, 4, cache_dir=self.cache)
        self.assertTrue(result["peak_limited"])
        self.assertLessEqual(result["output_true_peak_estimate"], -2)
        self.assertLess(result["output_lufs_estimate"], -18)
        self.assertTrue(result["warnings"])
        for changes in ({"start": float("nan")}, {"duration": .1}, {"start": 3},
                        {"target_lufs": float("inf")}, {"peak_dbfs": 0}):
            args = {"start": 0, "duration": 4, **changes}
            with self.subTest(changes=changes), self.assertRaises(MusicAnalysisError):
                normalization_filter(source, cache_dir=self.cache, **args)

    def test_normalization_cache_is_bound_to_excerpt_and_targets(self):
        source = self.synthetic(levels=((4, .1),))
        result = normalization_filter(source, 0, 4, cache_dir=self.cache)
        with patch.object(music_analysis.subprocess, "run", side_effect=AssertionError("cache miss")):
            self.assertEqual(normalization_filter(source, 0, 4, cache_dir=self.cache), result)
        changed = normalization_filter(source, 0, 4, cache_dir=self.cache, target_lufs=-20)
        self.assertNotEqual(changed["gain_db"], result["gain_db"])
        self.assertEqual(len(list(self.cache.glob("normalization-*.json"))), 2)

    def test_very_quiet_material_has_a_bounded_gain(self):
        source = self.synthetic(levels=((4, .003),))
        report = normalization_filter(source, 0, 4)
        self.assertEqual(report["gain_db"], 20)
        self.assertLess(report["output_lufs_estimate"], -18)
        self.assertTrue(any("20 dB" in warning for warning in report["warnings"]))


if __name__ == "__main__":
    unittest.main()
