"""Local, visibly synthetic fixtures only; no music provider calls."""
import array
import base64
import copy
import json
import math
from pathlib import Path
import shutil
import tempfile
import unittest
import wave

from norlan.media import MediaError, probe
from norlan.music_library import (
    MusicLibraryError, approve_track, approved_tracks, import_data_url, import_track,
    reject_track, render_music_bed, validate_cue_plan,
)


@unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "FFmpeg required")
class MusicLibraryTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix="nolan-synthetic-music-tests-")
        self.root = Path(self.directory.name).resolve()
        self.source = self.tone("SYNTHETIC TEST TONE.wav", 2.0)
        self.library = {"version": 1, "tracks": []}

    def tearDown(self):
        self.directory.cleanup()

    def tone(self, name, duration):
        output = self.root / name
        # 400 Hz and 48 kHz give exactly four periods per 480-sample block.
        block = array.array("h", [round(6000 * math.sin(2 * math.pi * 400 * i / 48000))
                                  for i in range(480)]).tobytes()
        with wave.open(str(output), "wb") as target:
            target.setnchannels(1)
            target.setsampwidth(2)
            target.setframerate(48000)
            target.writeframes((block * math.ceil(duration * 100))[:round(duration * 48000) * 2])
        return output

    def candidate(self, *, track_id="test-tone", version=1, source=None):
        track = import_track(self.root, source or self.source, "合成测试音，不是真实配乐",
                             track_id=track_id, version=version,
                             provenance={"source": "synthetic-test"})
        self.library["tracks"].append(track)
        return track

    def approved(self, **options):
        track = self.candidate(**options)
        approve_track(self.library, track["id"], track["version"], track["sha256"], project=self.root)
        return track

    @staticmethod
    def cue(track_id="test-tone", **changes):
        cue = {"track_id": track_id, "start": 0.25, "end": 0.75, "source_start": 0.0,
               "gain_db": -12.0, "fade_in": 0.05, "fade_out": 0.05, "reason": "合成测试"}
        return {**cue, **changes}

    def test_import_and_human_approval_bind_bytes_and_version(self):
        original = self.source.read_bytes()
        track = self.candidate()
        self.assertEqual(track["status"], "awaiting_approval")
        self.assertEqual(self.source.read_bytes(), original)
        self.assertTrue(Path(track["file"]).is_relative_to(self.root / "music" / "library"))
        self.assertEqual(approved_tracks(self.library, project=self.root), [])
        for version, digest in [(2, track["sha256"]), (1, "0" * 64)]:
            with self.assertRaises(MediaError):
                approve_track(self.library, track["id"], version, digest, project=self.root)
        approve_track(self.library, track["id"], 1, track["sha256"], project=self.root)
        self.assertEqual(track["approved_sha256"], track["sha256"])
        catalog = approved_tracks(self.library, project=self.root)
        self.assertEqual(catalog[0]["duration"], 2.0)
        self.assertNotIn("file", catalog[0])
        self.assertNotIn("provenance", catalog[0])
        Path(track["file"]).write_bytes(original + b"changed")
        with self.assertRaisesRegex(MediaError, "文件已改变"):
            approved_tracks(self.library, project=self.root)

    def test_relative_files_require_project_and_cannot_escape_library(self):
        track = self.candidate()
        track["file"] = str(Path(track["file"]).relative_to(self.root))
        with self.assertRaisesRegex(MediaError, "项目根目录"):
            approve_track(self.library, track["id"], 1, track["sha256"])
        approve_track(self.library, track["id"], 1, track["sha256"], project=self.root)
        self.assertEqual(len(approved_tracks(self.library, project=self.root)), 1)
        track["file"] = str(self.source)
        with self.assertRaisesRegex(MediaError, "受控音乐库"):
            approved_tracks(self.library, project=self.root)

    def test_reject_never_regenerates_and_removes_approval(self):
        track = self.approved()
        rejected = reject_track(self.library, track["id"], 1, track["sha256"], "过于紧张", project=self.root)
        self.assertEqual(rejected["status"], "rejected")
        self.assertEqual(rejected["feedback"], "过于紧张")
        self.assertNotIn("approved_sha256", rejected)
        self.assertTrue(Path(track["file"]).is_file())
        with self.assertRaisesRegex(MediaError, "试听确认"):
            validate_cue_plan({"cues": [self.cue()]}, self.library, 1, project=self.root)

    def test_upload_supports_real_audio_and_rejects_non_audio(self):
        encoded = base64.b64encode(self.source.read_bytes()).decode()
        result = import_data_url(self.root, "data:audio/wav;base64," + encoded, "上传的测试音")
        self.assertEqual(result["status"], "awaiting_approval")
        self.assertEqual(Path(result["file"]).read_bytes(), self.source.read_bytes())
        for value in ("data:text/plain;base64,SGVsbG8=", "data:audio/wav;base64,!!!",
                      "data:audio/wav;base64,SGVsbG8="):
            with self.assertRaises(MediaError):
                import_data_url(self.root, value, "非法音频")
        with self.assertRaises(MediaError):
            import_track(self.root, self.source, "测试", track_id="../../unsafe")

    def test_validated_plan_is_deterministic_and_source_bound(self):
        track = self.approved()
        original = {"summary": "测试留白", "cues": [self.cue()]}
        plan = validate_cue_plan(original, self.library, 1, project=self.root)
        self.assertEqual(plan["cues"][0]["track_sha256"], track["sha256"])
        self.assertEqual(plan["cues"][0]["track_version"], 1)
        self.assertEqual(plan, validate_cue_plan(plan, self.library, 1, project=self.root))
        changed = copy.deepcopy(original)
        changed["cues"][0]["gain_db"] = -15
        self.assertNotEqual(plan["fingerprint"], validate_cue_plan(changed, self.library, 1)["fingerprint"])
        self.assertNotIn("track_sha256", original["cues"][0])

    def test_model_catalog_does_not_expand_with_full_music_prompts(self):
        track = self.approved()
        track["prompt"] = "合成测试音乐说明" * 600
        catalog = approved_tracks(self.library, project=self.root)
        self.assertLessEqual(len(catalog[0]["prompt"]), 480)
        self.assertGreater(len(track["prompt"]), 4000)
        self.assertLess(len(json.dumps(catalog * 12, ensure_ascii=False)), 10000)
        self.assertNotIn("approved_at", catalog[0])

    def test_music_validation_failure_is_displayable_and_not_a_provider_retry(self):
        self.candidate()
        with self.assertRaises(MusicLibraryError) as failure:
            validate_cue_plan({"cues": [self.cue()]}, self.library, 1, project=self.root)
        self.assertTrue(failure.exception.safe_message)
        self.assertIn("试听确认", str(failure.exception))

    def test_rejects_unapproved_stale_out_of_bounds_or_nonfinite_cues(self):
        track = self.candidate()
        with self.assertRaisesRegex(MediaError, "试听确认"):
            validate_cue_plan({"cues": [self.cue()]}, self.library, 1)
        approve_track(self.library, track["id"], 1, track["sha256"])
        invalid = [
            {"start": True}, {"end": float("nan")}, {"gain_db": float("inf")},
            {"end": 1.1}, {"source_start": 1.75}, {"fade_in": 0.7},
            {"fade_in": 0.4, "fade_out": 0.4}, {"gain_db": 20},
            {"track_version": 2}, {"track_sha256": "0" * 64},
            {"loop": True}, {"filter": "amovie=/etc/passwd"},
        ]
        for changes in invalid:
            with self.subTest(changes=changes), self.assertRaises(MediaError):
                validate_cue_plan({"cues": [self.cue(**changes)]}, self.library, 1, project=self.root)

    def test_crossfade_must_cover_overlap_and_no_three_layer_music(self):
        self.approved()
        one = self.cue(start=0, end=1.0, fade_out=0.2)
        two = self.cue(start=0.8, end=1.8, fade_in=0.2)
        plan = validate_cue_plan({"cues": [two, one]}, self.library, 2)
        self.assertEqual(plan["cues"][0]["start"], 0)
        two["fade_in"] = 0.05
        with self.assertRaisesRegex(MediaError, "完整的淡出"):
            validate_cue_plan({"cues": [one, two]}, self.library, 2)
        two["fade_in"] = 0.2
        third = self.cue(start=0.9, end=1.9, fade_in=0.9)
        with self.assertRaisesRegex(MediaError, "最多允许两段"):
            validate_cue_plan({"cues": [one, two, third]}, self.library, 2)

    def test_render_keeps_silent_gaps_and_caches_only_exact_manifest(self):
        track = self.approved()
        source_bytes = Path(track["file"]).read_bytes()
        plan = {"cues": [self.cue()]}
        result = render_music_bed(plan, self.library, 1.0, self.root / "mix-v1.wav", project=self.root)
        self.assertEqual(result["duration"], 1.0)
        self.assertTrue(result["music_present"])
        self.assertEqual(Path(track["file"]).read_bytes(), source_bytes)
        with wave.open(result["output"], "rb") as output:
            values = array.array("h", output.readframes(output.getnframes()))
        def power(start, end):
            region = values[round(start * 48000) * 2:round(end * 48000) * 2]
            return sum(v * v for v in region) / len(region)
        self.assertEqual(power(0.05, 0.2), 0)
        self.assertGreater(power(0.4, 0.6), 1000)
        self.assertEqual(power(0.8, 0.95), 0)
        self.assertEqual(result, render_music_bed(plan, self.library, 1.0, Path(result["output"])))
        with self.assertRaisesRegex(MediaError, "新的版本"):
            render_music_bed({"cues": []}, self.library, 1.0, Path(result["output"]))
        Path(result["output"]).write_bytes(b"tampered")
        with self.assertRaisesRegex(MediaError, "新的版本"):
            render_music_bed(plan, self.library, 1.0, Path(result["output"]))

    def test_render_rechecks_approval_and_does_not_accept_stale_normalized_plan(self):
        track = self.approved()
        plan = validate_cue_plan({"cues": [self.cue()]}, self.library, 1.0, project=self.root)
        # A normalized plan contains the version/hash fields passed by the
        # orchestrator; those fields must be verified again at actual rendering.
        track["version"] = 2
        track["approved_version"] = 2
        with self.assertRaisesRegex(MediaError, "版本已改变"):
            render_music_bed(plan, self.library, 1.0, self.root / "stale.wav", project=self.root)
        self.assertFalse((self.root / "stale.wav").exists())
        track["version"] = track["approved_version"] = 1
        reject_track(self.library, track["id"], 1, track["sha256"], "不适合", project=self.root)
        with self.assertRaisesRegex(MediaError, "试听确认"):
            render_music_bed(plan, self.library, 1.0, self.root / "rejected.wav", project=self.root)
        self.assertFalse((self.root / "rejected.wav").exists())

    def test_explicit_zero_fade_does_not_apply_ffmpeg_default_fade(self):
        self.approved()
        result = render_music_bed({"cues": [self.cue(start=0, end=0.5, gain_db=0,
                                                       fade_in=0, fade_out=0)]},
                                  self.library, 0.5, self.root / "hard-cut.wav", project=self.root)
        with wave.open(result["output"], "rb") as output:
            values = array.array("h", output.readframes(output.getnframes()))
        early = values[round(0.03 * 48000) * 2:round(0.08 * 48000) * 2]
        late = values[round(0.38 * 48000) * 2:round(0.43 * 48000) * 2]
        early_power = sum(v*v for v in early) / len(early)
        late_power = sum(v*v for v in late) / len(late)
        self.assertGreater(early_power, 5_000_000)
        self.assertAlmostEqual(early_power / late_power, 1.0, delta=0.05)

    def test_130_second_episode_uses_explicit_cues_without_provider_limit_or_loop(self):
        source = self.tone("SYNTHETIC 45 SECOND TEST MUSIC.wav", 45)
        self.approved(source=source)
        cues = [self.cue(start=0, end=45, fade_in=0.5, fade_out=1),
                self.cue(start=44, end=89, fade_in=1, fade_out=1),
                self.cue(start=88, end=130, fade_in=1, fade_out=0.5)]
        result = render_music_bed({"summary": "显式复用已确认测试音乐", "cues": cues},
                                  self.library, 130, self.root / "long-bed.wav", project=self.root)
        self.assertEqual(len(result["cue_plan"]["cues"]), 3)
        self.assertAlmostEqual(probe(Path(result["output"]))["duration"], 130, places=4)
        # A single cue cannot silently loop the same 45-second asset to 130 s.
        with self.assertRaisesRegex(MediaError, "不能隐式循环"):
            validate_cue_plan({"cues": [self.cue(start=0, end=130)]}, self.library, 130)


if __name__ == "__main__":
    unittest.main()
