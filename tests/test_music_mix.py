"""Synthetic tone/video tests; these fixtures are not real speech or music."""

import array
import math
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from norlan import media, music_mix


@unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "FFmpeg required")
class SyntheticMusicMixTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix="nolan-synthetic-music-mix-")
        self.root = Path(self.directory.name)

    def tearDown(self):
        self.directory.cleanup()

    def tone(self, name, frequency=997, amplitude=0.05, duration=6):
        path = self.root / name
        media._ffmpeg(["-f", "lavfi", "-i",
                       f"aevalsrc={amplitude}*sin(2*PI*{frequency}*t):s=48000:d={duration}",
                       "-ac", "2", "-c:a", "pcm_f32le", str(path)])
        return path

    def clip(self, duration=6):
        path = self.root / "synthetic-native.mp4"
        media._ffmpeg(["-f", "lavfi", "-i", f"color=c=blue:s=160x120:r=30:d={duration}",
                       "-f", "lavfi", "-i", f"aevalsrc=0.15*sin(2*PI*440*t):s=48000:d={duration}",
                       "-t", str(duration), "-c:v", "libx264", "-pix_fmt", "yuv420p",
                       "-c:a", "aac", str(path)])
        return path

    @staticmethod
    def cue(start=0, end=6, **kwargs):
        return {"start": start, "end": end, "fade_in": 0.2, "fade_out": 0.2,
                "track_id": "synthetic-tone-not-real-music", "track_version": 1,
                "track_sha256": "a" * 64, "source_start": 0, **kwargs}

    def level(self, path, start, end):
        samples = music_mix._decode(path, end)
        region = samples[round(start * 32000):round(end * 32000)]
        return music_mix._db(sum(x * x for x in region) / len(region))

    def test_synthetic_manual_speech_duck_is_bounded_and_environment_cannot_trigger_it(self):
        native = self.tone("loud-environment.wav", frequency=440, amplitude=0.7)
        bed = self.tone("music.wav")
        target = self.root / "manual.wav"
        # Manual confirmation does not run a decoder/level check internally.
        with patch.object(music_mix, "_decode", side_effect=AssertionError("manual must not analyze")):
            plan = music_mix.prepare_mix(native, bed, target, duration=6,
                cues=[self.cue()], speech_windows=[{"start": 2, "end": 4}], adaptive=False)
        self.assertIsNone(plan["measurement"])
        outside = self.level(target, 0.5, 1.5)
        inside = self.level(target, 2.5, 3.5)
        after = self.level(target, 4.8, 5.5)
        self.assertAlmostEqual(inside - outside, -6, delta=0.15)
        self.assertAlmostEqual(after - outside, 0, delta=0.15)
        self.assertAlmostEqual(outside, self.level(bed, 0.5, 1.5), delta=0.15)

    def test_synthetic_automatic_mix_repairs_low_music_without_erasing_planned_gaps(self):
        native = self.tone("native.wav", frequency=440, amplitude=0.1, duration=8)
        bed = self.root / "bed-with-gap.wav"
        media._ffmpeg(["-f", "lavfi", "-i",
                       "aevalsrc=0.01*sin(2*PI*997*t)*between(t\\,1\\,7):s=48000:d=8",
                       "-ac", "2", "-c:a", "pcm_f32le", str(bed)])
        plan = music_mix.prepare_mix(native, bed, self.root / "prepared.wav", duration=8,
            cues=[self.cue(1, 7)], speech_windows=[{"start": 3, "end": 5}])
        self.assertEqual(plan["measurement"]["status"], "passed")
        self.assertAlmostEqual(plan["measurement"]["planned_silence_seconds"], 2, delta=0.1)
        self.assertLess(self.level(Path(plan["output"]), 0.1, 0.8), -100)
        self.assertLessEqual(plan["automatic_gain_max_db"], music_mix.MAX_ADAPTIVE_DB)
        self.assertEqual(plan["cues"][0]["source_sha256"], "a" * 64)

    def test_synthetic_missing_music_is_detected_in_actual_export(self):
        native = self.tone("native.wav", frequency=440, amplitude=0.1)
        bed = self.tone("music.wav", amplitude=0.03)
        plan = music_mix.prepare_mix(native, bed, self.root / "prepared.wav", duration=6,
            cues=[self.cue()], speech_windows=[])
        final = self.root / "missing-music.wav"
        shutil.copyfile(native, final)
        report = music_mix.audit_mix(native, Path(plan["output"]), final, plan=plan)
        self.assertEqual(report["status"], "needs_attention")
        self.assertTrue(any(issue["code"] == "export_music_unverified" for issue in report["issues"]))
        self.assertEqual(report["final_sha256"], media._sha256(final))

    def test_synthetic_export_projection_confirms_music_after_final_loudness(self):
        source = self.clip()
        bed = self.tone("music.wav", amplitude=0.02)
        result = media.compose([source], self.root / "synthetic-final.mp4", music=bed,
            music_gain=1, music_mix={"policy_version": 1, "cues": [self.cue()],
                "speech_windows": [{"start": 2, "end": 4}],
                "binding": {"source_key": "synthetic", "source_analysis_used": True}})
        report = result["music_mix"]
        self.assertEqual(report["status"], "passed", report)
        self.assertEqual(report["final_sha256"], result["sha256"])
        self.assertTrue(report["source_analysis_used"])
        self.assertTrue(any(row["status"] == "verified" for row in report["export_projection"]))
        self.assertLessEqual(report["repair_attempts"], 2)

    def test_synthetic_very_quiet_music_has_bounded_repairs_and_never_false_passes(self):
        source = self.clip()
        bed = self.tone("inaudible.wav", amplitude=0.002)
        progress = []
        result = media.compose([source], self.root / "bounded-final.mp4", music=bed,
            music_gain=1, music_mix={"cues": [self.cue()], "speech_windows": [],
                                     "on_progress": progress.append})
        report = result["music_mix"]
        self.assertEqual(report["repair_attempts"], 2)
        self.assertEqual(report["status"], "needs_attention")
        self.assertTrue(any(issue["code"] == "music_too_quiet" for issue in report["issues"]))
        self.assertTrue(Path(result["output"]).is_file())
        self.assertEqual(report["local_corrections_db"], {"0": 6.0})
        repairs = [row for row in progress if row["status"] == "repairing"]
        self.assertEqual([row["repair_attempts"] for row in repairs], [1, 2])
        self.assertEqual(repairs[-1]["bed_sha256"], media._sha256(bed))
        self.assertEqual(repairs[-1]["local_corrections_db"], {"0": 6.0})

    def test_synthetic_exhausted_repair_attempts_are_not_reset_on_resume(self):
        source = self.clip(duration=2)
        bed = self.tone("exhausted.wav", amplitude=0.002, duration=2)
        progress = []
        result = media.compose([source], self.root / "resumed-final.mp4", music=bed,
            music_gain=1, music_mix={"cues": [self.cue(0, 2)], "speech_windows": [],
                "repair_attempt": 2, "local_corrections_db": {"0": 6}, "on_progress": progress.append})
        self.assertEqual(result["music_mix"]["repair_attempts"], 2)
        self.assertEqual(result["music_mix"]["status"], "needs_attention")
        self.assertFalse(any(row["status"] == "repairing" for row in progress))

    def test_synthetic_manual_compose_does_not_check_or_change_slider_gain(self):
        source = self.clip(duration=2)
        bed = self.tone("manual-bed.wav", amplitude=0.02, duration=2)
        with patch.object(music_mix, "audit_mix", side_effect=AssertionError("manual must not audit")), \
             patch.object(music_mix, "_measure", side_effect=AssertionError("manual must not check")):
            result = media.compose([source], self.root / "manual-final.mp4", music=bed,
                music_gain=1, music_mix={"adaptive": False, "audit": False,
                    "cues": [self.cue(0, 2)], "speech_windows": []})
        self.assertEqual(result["music_mix"]["status"], "manual")
        self.assertEqual(result["music_mix"]["repair_attempts"], 0)

    def test_synthetic_planned_silence_and_boundary_fades_are_not_missing_music(self):
        native = self.tone("native.wav", frequency=440, amplitude=0.1)
        empty = self.tone("silence.wav", amplitude=0)
        plan = music_mix.prepare_mix(native, empty, self.root / "empty-prepared.wav", duration=6,
                                    cues=[], speech_windows=[])
        self.assertEqual(plan["measurement"]["status"], "passed")
        self.assertEqual(plan["measurement"]["planned_silence_seconds"], 6)
        plan = music_mix.prepare_mix(native, empty, self.root / "bad-source.wav", duration=6,
                                    cues=[self.cue()], speech_windows=[])
        self.assertEqual(plan["measurement"]["status"], "needs_attention")
        self.assertEqual(plan["measurement"]["issues"][0]["code"], "source_silence")

    def test_synthetic_invalid_unbounded_duck_and_repairs_are_rejected(self):
        native = self.tone("native.wav", frequency=440)
        bed = self.tone("bed.wav")
        for values in ({"duck_db": 7}, {"duck_db": math.nan}, {"repair_attempt": 3},
                       {"local_corrections_db": {"0": 8}}):
            with self.subTest(values=values), self.assertRaises(media.MediaError):
                music_mix.prepare_mix(native, bed, self.root / "bad.wav", duration=6,
                                      cues=[self.cue()], speech_windows=[], **values)

    def test_synthetic_manual_subtitle_edit_reuses_exact_prepared_music_and_duck(self):
        source = self.clip()
        bed = self.tone("baseline-bed.wav", amplitude=0.01)
        cue = self.cue()
        automatic = media.compose([source], self.root / "automatic.mp4", music=bed, music_gain=1,
            music_mix={"cues": [cue], "speech_windows": [{"start": 2, "end": 4}]})
        report = automatic["music_mix"]
        reuse = {"file": report["prepared_music_file"], "sha256": report["prepared_music_sha256"],
                 "automation_file": report["automation_file"], "automation_sha256": report["automation_sha256"]}
        self.assertTrue(Path(reuse["file"]).is_file())
        self.assertEqual(media._sha256(Path(reuse["automation_file"])), reuse["automation_sha256"])
        with patch.object(music_mix, "_decode", side_effect=AssertionError("manual must not analyze")), \
             patch.object(music_mix, "prepare_mix", side_effect=AssertionError("subtitle must not remix")), \
             patch.object(music_mix, "audit_mix", side_effect=AssertionError("manual must not check")):
            manual = media.compose([source], self.root / "manual-subtitles.mp4", music=bed, music_gain=1,
                subtitles=[{"start": 0.3, "end": 1, "text": "合成字幕修改"}], burn_subtitles=False,
                music_mix={"adaptive": False, "audit": False, "cues": [cue],
                    "speech_windows": [{"start": 0.3, "end": 1}], "reuse_prepared_music": reuse})
        self.assertEqual(manual["music_mix"]["status"], "manual")
        self.assertTrue(manual["music_mix"]["reused_prepared_music"])
        self.assertEqual(manual["music_mix"]["prepared_music_sha256"], report["prepared_music_sha256"])
        self.assertEqual(manual["music_mix"]["automation_sha256"], report["automation_sha256"])

    def test_synthetic_manual_music_preserves_baseline_and_applies_only_explicit_gain_change(self):
        native = self.tone("native.wav", frequency=440, amplitude=0.1)
        bed = self.tone("old-bed.wav", amplitude=0.01)
        cues = [self.cue(0, 3), self.cue(3, 6)]
        automatic = music_mix.prepare_mix(native, bed, self.root / "old-prepared.wav", duration=6,
            cues=cues, speech_windows=[{"start": 1, "end": 2}], repair_attempt=2,
            local_corrections_db={"0": 6, "1": 6})
        unchanged = self.root / "unchanged.wav"
        with patch.object(music_mix, "_decode", side_effect=AssertionError("manual must not analyze")):
            music_mix.prepare_mix(native, bed, unchanged, duration=6, cues=cues,
                speech_windows=[{"start": 1, "end": 2}], adaptive=False,
                baseline_automation=automatic["automation"])
        self.assertEqual(media._sha256(unchanged), automatic["prepared_music_sha256"])
        edited = self.root / "one-cue-edited.wav"
        media._ffmpeg(["-i", str(bed), "-af", "volume=0.501187:enable='gte(t,3)'",
                       "-c:a", "pcm_f32le", str(edited)])
        changed = self.root / "changed.wav"
        with patch.object(music_mix, "_decode", side_effect=AssertionError("manual must not analyze")):
            music_mix.prepare_mix(native, edited, changed, duration=6, cues=cues,
                speech_windows=[{"start": 1, "end": 2}], adaptive=False,
                baseline_automation=automatic["automation"])
        self.assertAlmostEqual(self.level(changed, 0.5, 0.9) - self.level(unchanged, 0.5, 0.9), 0, delta=0.01)
        self.assertAlmostEqual(self.level(changed, 3.7, 5.5) - self.level(unchanged, 3.7, 5.5), -6, delta=0.01)

    def test_synthetic_near_silent_final_cannot_pass_when_native_is_silent(self):
        native = self.tone("silent-native.wav", amplitude=0)
        bed = self.tone("audible-bed.wav", amplitude=0.03)
        plan = music_mix.prepare_mix(native, bed, self.root / "prepared.wav", duration=6,
                                    cues=[self.cue()], speech_windows=[])
        final = self.root / "almost-silent-final.wav"
        media._ffmpeg(["-i", plan["output"], "-af", "volume=0.00001", "-c:a", "pcm_f32le", str(final)])
        report = music_mix.audit_mix(native, Path(plan["output"]), final, plan=plan)
        self.assertEqual(report["status"], "needs_attention")
        self.assertTrue(any(issue["code"] == "export_music_unverified" for issue in report["issues"]))
        self.assertTrue(all(row["status"] == "unverified" for row in report["export_projection"]))

    def test_synthetic_quiet_non_speech_keeps_audible_music_without_contradictory_repairs(self):
        native = self.tone("quiet-environment.wav", frequency=440, amplitude=0.009)
        bed = self.tone("moderate-music.wav", amplitude=0.06)
        prepared = self.root / "quiet-scene-prepared.wav"
        plan = music_mix.prepare_mix(native, bed, prepared, duration=6,
                                    cues=[self.cue()], speech_windows=[])
        self.assertEqual(plan["measurement"]["status"], "passed", plan["measurement"])
        self.assertEqual(plan["measurement"]["repair_suggestions"], {})
        self.assertAlmostEqual(self.level(native, 1, 5), -46.94, delta=0.1)
        self.assertAlmostEqual(self.level(prepared, 1, 5), -32, delta=0.2)
        # A quiet non-dialogue background can sit above the ambient recording;
        # the same level must still be rejected over quiet actual dialogue.
        native_power = [10 ** (-47 / 10)] * 60
        music_power = [10 ** (-30 / 10)] * 60
        cue = music_mix._cue_rows([self.cue()], 6)
        report = music_mix._measure(native_power, music_power, music_power,
            cues=cue, speech=[], duration=6)
        self.assertEqual(report["status"], "passed")
        speech_report = music_mix._measure(native_power, music_power, music_power,
            cues=cue, speech=[{"start": 0, "end": 6}], duration=6)
        self.assertTrue(any(issue["code"] == "music_too_loud" for issue in speech_report["issues"]))
        loud = [10 ** (-14 / 10)] * 60
        loud_report = music_mix._measure(native_power, loud, loud,
            cues=cue, speech=[], duration=6)
        self.assertTrue(any(issue["code"] == "music_too_loud" for issue in loud_report["issues"]))


if __name__ == "__main__":
    unittest.main()
