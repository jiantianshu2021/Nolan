"""Explicitly synthetic, local-only tests for atomic manual postproduction edits.

Source/video/export files are labelled dummy bytes; no visual/audio quality is
claimed. Music validation reads a real synthetic PCM WAV. No render or provider
is invoked by the edit API.
"""
from copy import deepcopy
from io import BytesIO
from pathlib import Path
import shutil
import tempfile
import unittest
import wave

from norlan import core, post_edits


class PostEditsTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="nolan-synthetic-manual-post-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        core.create_project(self.root, "仅本地手动编辑的合成测试小说", mode="demo")
        clip = self.root / "media/E001-S001-v1.mp4"
        clip.write_bytes(b"SYNTHETIC DUMMY VIDEO - NO ACTUAL VIDEO QUALITY CLAIM")
        final = self.root / "exports/E001-v2.mp4"
        final.write_bytes(b"SYNTHETIC DUMMY FINAL - IMMUTABLE SOURCE TEST")
        sha = core.digest(clip)
        self.shot = {"id": "E001-S001", "version": 1, "status": "accepted", "duration": 10,
            "file": "media/E001-S001-v1.mp4", "sha256": sha, "approved_sha256": sha,
            "approved_version": 1, "approved_at": core.now(),
            "human_decision": {"decision": "approved", "version": 1, "sha256": sha, "at": core.now()}}
        self.source_key = core._json_digest([{"id": self.shot["id"], "version": 1, "sha256": sha}])
        music = self.root / "music/library/synthetic-theme.wav"
        music.parent.mkdir(parents=True)
        buffer = BytesIO()
        with wave.open(buffer, "wb") as stream:
            stream.setnchannels(1); stream.setsampwidth(2); stream.setframerate(16000)
            stream.writeframes((b"\x40\x00\xc0\xff" * 8000) * 12)
        music.write_bytes(buffer.getvalue())
        msha = core.digest(music)
        self.track = {"id": "synthetic-theme", "title": "合成 PCM 测试音", "version": 1,
            "file": str(music.relative_to(self.root)), "sha256": msha, "duration": 12.,
            "status": "approved", "approved_sha256": msha, "approved_version": 1, "approved_at": core.now()}
        self.cues = [{"track_id": self.track["id"], "track_version": 1, "track_sha256": msha,
            "start": start, "end": end, "source_start": source, "gain_db": -20.,
            "fade_in": .8, "fade_out": .8, "reason": "原有合成测试安排"}
            for start, end, source in ((0., 4., 0.), (6., 10., 1.))]
        self.segments = [{"start": .5, "end": 2., "text": "原第一句"},
                         {"start": 3., "end": 5., "text": "原第二句"}]
        with core.transaction(self.root) as state:
            state["settings"]["postproduction"] = {"enabled": True}
            state["assets"] = [{"id": "immutable-character", "name": "已有角色"}]
            state["music_library"] = {"version": 1, "tracks": [deepcopy(self.track)]}
            state["budget"].update(limit=2000, approved=True)
            state["jobs"] = {"old-paid-job": {"kind": "image", "item_id": "old-character",
                "state": "completed", "accounting": "spent", "amount": 5., "currency": "CNY"}}
            state["episodes"] = [{"id": "E001", "title": "合成后期测试", "version": 2,
                "status": "awaiting_review", "shots": [deepcopy(self.shot)], "duration": 10.,
                "file": str(final.relative_to(self.root)), "sha256": core.digest(final),
                "approved_at": "legacy-marker", "approved_sha256": core.digest(final),
                "history": [], "postproduction": {"enabled": True, "ready": True,
                    "final_ready": True, "revision": 3, "source_key": self.source_key, "duration": 10.,
                    "status": "awaiting_final_approval", "subtitle_key": "preserve-subtitle-key",
                    "subtitle_proofread_key": "old-proofread-key", "music_key": "preserve-music-key",
                    "subtitles": {"status": "ready", "segments": deepcopy(self.segments),
                        "sync_check": {"status": "passed", "synthetic_sentinel": True},
                        "proofread_segments": deepcopy(self.segments)},
                    "music": {"status": "ready", "summary": "已有安排的说明", "cues": deepcopy(self.cues)}}}]
            state.update(status="awaiting_episode_review", paused=True)
        self.music_edits = [{"index": 0, "start": 1., "end": 3., "gain_db": -30.},
                            {"index": 1, "start": 7., "end": 10., "gain_db": -18.}]
        self.subtitle_edits = [{"index": 0, "start": .8, "end": 2.2, "text": "手动第一句"},
                               {"index": 1, "start": 3.1, "end": 5.4, "text": "手动第二句"}]

    def state(self):
        return core.load_project(self.root)

    def episode(self):
        return self.state()["episodes"][0]

    def update(self, callback):
        with core.transaction(self.root) as state:
            callback(state)

    def files(self):
        return {str(p.relative_to(self.root)): p.read_bytes()
                for folder in ("source", "media", "exports", "music")
                for p in (self.root / folder).rglob("*") if p.is_file()}

    def apply(self, section="subtitles", edits=None, **options):
        if edits is None:
            edits = self.subtitle_edits if section == "subtitles" else self.music_edits
        return post_edits.apply_edits(self.root, "E001", options.get("version", 2),
                                     options.get("revision", 3), section, edits)

    def assert_atomic_rejection(self, callback, error=ValueError):
        before, files = (self.root / "project.json").read_bytes(), self.files()
        with self.assertRaises(error):
            callback()
        self.assertEqual((self.root / "project.json").read_bytes(), before)
        self.assertEqual(self.files(), files)

    def test_manual_subtitles_keep_human_text_and_seconds_without_rendering_or_cost(self):
        before, files = self.state(), self.files()
        result = self.apply()
        ep, pp = result["episodes"][0], result["episodes"][0]["postproduction"]
        self.assertEqual({k: pp["manual_subtitles"][k] for k in ("segments", "source_key", "revision")}, {"segments": [
            {k: row[k] for k in ("start", "end", "text")} for row in self.subtitle_edits],
            "source_key": self.source_key, "revision": 4})
        self.assertEqual(pp["revision_target"], "manual_subtitles")
        self.assertNotIn("subtitle_key", pp)
        self.assertNotIn("subtitle_proofread_key", pp)
        self.assertNotIn("sync_check", pp["subtitles"])
        self.assertEqual(pp["subtitles"]["previous_sync_check"],
                         before["episodes"][0]["postproduction"]["subtitles"]["sync_check"])
        from norlan.subtitle_sync import caption_hash
        submission = pp["manual_edit_submission"]
        self.assertEqual(submission["section"], "subtitles")
        self.assertEqual(submission["revision"], 4)
        self.assertEqual(submission["source_key"], self.source_key)
        self.assertEqual(submission["caption_hash"], caption_hash(pp["manual_subtitles"]["segments"]))
        self.assertEqual(pp["manual_subtitles"]["submitted_caption_hash"], submission["caption_hash"])
        self.assertTrue(pp["manual_subtitles"]["submitted_at"])
        self.assertTrue(submission["submitted_at"])
        self.assertEqual(pp["music_key"], "preserve-music-key")
        self.assertEqual(ep["status"], "postproduction")
        self.assertFalse(pp["final_ready"])
        self.assertEqual(ep["history"][0]["file"], before["episodes"][0]["file"])
        self.assertNotIn("approved_at", ep)
        self.assertEqual(result["status"], "ready")
        self.assertFalse(result["paused"])
        for field in ("assets", "music_library", "jobs", "budget", "llm_calls"):
            self.assertEqual(result[field], before[field])
        self.assertEqual(ep["shots"], before["episodes"][0]["shots"])
        self.assertEqual(self.files(), files)

    @unittest.skipUnless(shutil.which("ffprobe"), "requires local music metadata verification")
    def test_manual_music_changes_only_gain_and_timing_and_preserves_subtitle_report(self):
        before, files = self.state(), self.files()
        result = self.apply("music")
        pp = result["episodes"][0]["postproduction"]
        plan = pp["manual_music_plan"]
        self.assertEqual(plan["summary"], "已有安排的说明")
        self.assertEqual(plan["source_key"], self.source_key)
        self.assertEqual(plan["revision"], 4)
        for original, cue, change in zip(self.cues, plan["cues"], self.music_edits):
            for key in ("track_id", "track_version", "track_sha256", "source_start", "reason"):
                self.assertEqual(cue[key], original[key])
            for key in ("start", "end", "gain_db"):
                self.assertEqual(cue[key], change[key])
        self.assertEqual(pp["subtitles"], before["episodes"][0]["postproduction"]["subtitles"])
        self.assertEqual(pp["subtitle_key"], "preserve-subtitle-key")
        self.assertNotIn("music_key", pp)
        self.assertEqual(pp["manual_edit_submission"]["section"], "music")
        self.assertEqual(pp["manual_edit_submission"]["source_key"], self.source_key)
        self.assertEqual(pp["manual_edit_submission"]["revision"], 4)
        self.assertNotIn("caption_hash", pp["manual_edit_submission"])
        for field in ("assets", "music_library", "jobs", "budget", "llm_calls"):
            self.assertEqual(result[field], before[field])
        self.assertEqual(self.files(), files)

    @unittest.skipUnless(shutil.which("ffprobe"), "requires local music metadata verification")
    def test_shortened_music_proportionally_clamps_both_fades(self):
        changes = deepcopy(self.music_edits)
        changes[0].update(start=1., end=1.4)
        pp = self.apply("music", changes)["episodes"][0]["postproduction"]
        cue = pp["manual_music_plan"]["cues"][0]
        self.assertAlmostEqual(cue["fade_in"], .2)
        self.assertAlmostEqual(cue["fade_out"], .2)

    def test_episode_version_and_post_revision_conflicts_are_atomic(self):
        for options in ({"version": 1}, {"version": True}, {"revision": 2}, {"revision": True}, {"revision": "3"}):
            with self.subTest(options=options):
                self.assert_atomic_rejection(lambda: self.apply(**options))

    def test_busy_run_lock_does_not_save_an_edit(self):
        with core.run_lock(self.root):
            self.assert_atomic_rejection(lambda: self.apply(), core.GateError)

    def test_finite_bounded_numbers_and_order_are_required_for_every_row(self):
        for value in (float("nan"), float("inf"), -1., True, "2", None, 11.):
            for field in ("start", "end"):
                with self.subTest(value=value, field=field):
                    changed = deepcopy(self.subtitle_edits); changed[0][field] = value
                    self.assert_atomic_rejection(lambda: self.apply(edits=changed), post_edits.EditError)
        changed = deepcopy(self.subtitle_edits); changed[0]["end"] = changed[0]["start"]
        self.assert_atomic_rejection(lambda: self.apply(edits=changed), post_edits.EditError)
        changed = deepcopy(self.subtitle_edits); changed[1]["start"] = 1.
        self.assert_atomic_rejection(lambda: self.apply(edits=changed), post_edits.EditError)

    def test_full_ordered_indices_only_no_client_track_path_or_asset_fields(self):
        for section, base in (("subtitles", self.subtitle_edits), ("music", self.music_edits)):
            for payload in (base[:-1], base + [deepcopy(base[-1])], list(reversed(base))):
                with self.subTest(section=section, payload=payload):
                    self.assert_atomic_rejection(lambda: self.apply(section, payload), post_edits.EditError)
            for key in ("track_id", "track_version", "track_sha256", "file", "path", "asset_id", "source_start", "fade_in"):
                changed = deepcopy(base); changed[0][key] = "UNTRUSTED SECRET SENTINEL"
                with self.subTest(section=section, key=key):
                    self.assert_atomic_rejection(lambda: self.apply(section, changed), post_edits.EditError)
            changed = deepcopy(base); changed[0]["index"] = False
            self.assert_atomic_rejection(lambda: self.apply(section, changed), post_edits.EditError)

    def test_music_gain_bounds_are_checked_before_any_plan_or_state_change(self):
        for value in (-61., .1, float("nan"), float("inf"), True, "-18"):
            changed = deepcopy(self.music_edits); changed[1]["gain_db"] = value
            with self.subTest(value=value):
                self.assert_atomic_rejection(lambda: self.apply("music", changed), post_edits.EditError)

    def test_empty_oversized_or_nul_text_rejected_without_echoing_input(self):
        for value in ("", "   ", "x" * 4001, "\x00SECRET_SENTINEL", None, 12):
            changed = deepcopy(self.subtitle_edits); changed[0]["text"] = value
            with self.subTest(value=value):
                self.assert_atomic_rejection(lambda: self.apply(edits=changed), post_edits.EditError)
        self.assertTrue(post_edits.EditError.safe_message)

    def test_nonapproved_shot_or_changed_source_record_does_not_retire_final(self):
        self.update(lambda s: s["episodes"][0]["shots"][0].update(status="reviewed"))
        self.assert_atomic_rejection(lambda: self.apply(), core.GateError)
        self.update(lambda s: s["episodes"][0]["shots"][0].update(status="accepted"))
        self.update(lambda s: s["episodes"][0]["postproduction"].update(source_key="0" * 64))
        self.assert_atomic_rejection(lambda: self.apply(), core.GateError)

    def test_changed_source_video_bytes_invalidate_edit_and_keep_original_export(self):
        (self.root / self.shot["file"]).write_bytes(b"INTENTIONALLY TAMPERED SYNTHETIC SOURCE")
        self.assert_atomic_rejection(lambda: self.apply(), core.GateError)

    @unittest.skipUnless(shutil.which("ffprobe"), "requires local music metadata verification")
    def test_invalid_music_approval_bytes_and_source_duration_fail_atomically(self):
        self.update(lambda s: s["music_library"]["tracks"][0].update(status="awaiting_approval"))
        self.assert_atomic_rejection(lambda: self.apply("music"), post_edits.EditError)
        self.update(lambda s: s["music_library"]["tracks"][0].update(status="approved", duration=11.))
        self.assert_atomic_rejection(lambda: self.apply("music"), post_edits.EditError)
        self.update(lambda s: s["music_library"]["tracks"][0].update(duration=12.))
        (self.root / self.track["file"]).write_bytes(b"INTENTIONALLY TAMPERED MUSIC")
        self.assert_atomic_rejection(lambda: self.apply("music"), post_edits.EditError)

    @unittest.skipUnless(shutil.which("ffprobe"), "requires local music metadata verification")
    def test_music_source_overrun_and_incomplete_crossfade_are_rejected(self):
        self.update(lambda s: s["episodes"][0]["postproduction"]["music"]["cues"][1].update(source_start=10.))
        self.assert_atomic_rejection(lambda: self.apply("music"), post_edits.EditError)
        self.update(lambda s: s["episodes"][0]["postproduction"]["music"]["cues"][1].update(source_start=1.))
        changed = deepcopy(self.music_edits); changed[0].update(start=2., end=8.)
        self.assert_atomic_rejection(lambda: self.apply("music", changed), post_edits.EditError)

    def test_existing_later_episode_blocks_edit_of_approved_prior_episode(self):
        def change(state):
            state["episodes"][0]["status"] = "approved"
            state["episodes"].append({"id": "E002", "shots": [{"file": "already-generated.mp4"}]})
        self.update(change)
        self.assert_atomic_rejection(lambda: self.apply(), core.GateError)

    @unittest.skipUnless(shutil.which("ffprobe"), "requires local music metadata verification")
    def test_consecutive_sections_keep_both_pending_human_edits(self):
        first = self.apply("music")
        original_plan = deepcopy(first["episodes"][0]["postproduction"]["manual_music_plan"])
        second = self.apply("subtitles", revision=4)
        pp = second["episodes"][0]["postproduction"]
        self.assertEqual(pp["revision"], 5)
        self.assertEqual(pp["manual_music_plan"]["cues"], original_plan["cues"])
        self.assertEqual(pp["manual_music_plan"]["revision"], 5)
        self.assertEqual(pp["manual_subtitles"]["revision"], 5)
        self.assertEqual(len(second["events"]), len(first["events"]) + 1)

    def test_repeated_subtitle_edit_uses_pending_table_and_new_revision(self):
        self.apply()
        changed = deepcopy(self.subtitle_edits); changed[1]["text"] = "再次手动修正"
        pp = self.apply(edits=changed, revision=4)["episodes"][0]["postproduction"]
        self.assertEqual(pp["manual_subtitles"]["segments"][1]["text"], "再次手动修正")
        self.assertEqual(pp["manual_subtitles"]["segments"][0]["start"], .8)
        self.assertEqual(pp["revision"], 5)

    def test_human_subtitle_submission_does_not_require_or_forge_machine_pass(self):
        self.update(lambda s: s["episodes"][0]["postproduction"]["subtitles"].update(
            sync_check={"status": "needs_attention", "issues": [{"code": "timing_mismatch"}]}))
        pp = self.apply()["episodes"][0]["postproduction"]
        self.assertNotIn("sync_check", pp["subtitles"])
        self.assertEqual(pp["subtitles"]["previous_sync_check"]["status"], "needs_attention")
        self.assertEqual(pp["manual_subtitles"]["segments"][0]["start"], .8)
        self.assertEqual(pp["manual_edit_submission"]["section"], "subtitles")


if __name__ == "__main__":
    unittest.main()
