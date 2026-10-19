"""Offline postproduction lifecycle with visibly synthetic clips/music.

The ASR and GPT boundaries are mocked. Real FFmpeg exports verify that the
approval gates operate on actual composed files, not fabricated media metadata.
"""
import copy
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from unittest.mock import Mock, patch

from norlan import core, episode_post, media, music_library


@unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "requires local FFmpeg; no providers")
class EpisodePostTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not media._has_filter("subtitles") or not media._font_path():
            raise unittest.SkipTest("requires libass and a local subtitle font")
        cls.fixture_dir = tempfile.TemporaryDirectory(prefix="nolan-synthetic-post-fixtures-")
        cls.fixture = Path(cls.fixture_dir.name)
        for index, color in enumerate(("blue", "red"), start=1):
            subprocess.run(["ffmpeg", "-v", "error", "-nostdin", "-y", "-f", "lavfi", "-i",
                f"color=c={color}:s=160x120:r=30:d=1", "-f", "lavfi", "-i",
                f"sine=frequency={400 + index * 100}:duration=1:sample_rate=48000",
                "-t", "1", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac",
                str(cls.fixture / f"synthetic-{index}.mp4")], check=True, capture_output=True)
        subprocess.run(["ffmpeg", "-v", "error", "-nostdin", "-y", "-f", "lavfi", "-i",
            "sine=frequency=220:duration=4:sample_rate=48000", "-c:a", "pcm_s16le",
            str(cls.fixture / "synthetic-music.wav")], check=True, capture_output=True)

    @classmethod
    def tearDownClass(cls):
        cls.fixture_dir.cleanup()

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="nolan-synthetic-post-test-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        core.create_project(self.root, "整部小说哨兵不得进入后期上下文" * 500, mode="demo")
        clips = self.root / "media" / "E001"
        clips.mkdir(parents=True)
        shots = []
        for index in (1, 2):
            sid = f"E001-S{index:03d}"
            destination = clips / f"{sid}-v1.mp4"
            shutil.copyfile(self.fixture / f"synthetic-{index}.mp4", destination)
            digest = core.digest(destination)
            shots.append({"id": sid, "version": 1, "status": "accepted", "duration": 1,
                "file": str(destination.relative_to(self.root)), "sha256": digest, "attempts": 1,
                "asset_ids": [], "prompt": "仅测试用合成色块和正弦波，不是真实对白",
                "dialogue": "计划台词不得冒充已识别声音", "approved_at": core.now(),
                "approved_version": 1, "approved_sha256": digest,
                "human_decision": {"decision": "approved", "version": 1, "sha256": digest, "at": core.now()}})
        with core.transaction(self.root) as state:
            state["settings"].update(postproduction={"enabled": True},
                subtitles={"mode": "local", "backend": "whisper_cpp", "language": "zh"},
                music={"mode": "none"}, brain={"provider": "codex", "billing": "subscription"})
            state["assets"] = [{"id": "character-test", "kind": "character", "name": "张山"}]
            state["episodes"] = [{"id": "E001", "title": "合成素材后期测试", "summary": "仅离线测试",
                "version": 0, "status": "postproduction", "shots": shots,
                "music_prompt": "测试低音主题；不调用音乐生成服务"}]
        self.raw = [{"start": 0.1, "end": 0.8, "text": "张三说话"},
                    {"start": 1.1, "end": 1.8, "text": "合成字幕样例"}]
        self.asr = Mock()
        self.asr.transcribe.return_value = copy.deepcopy(self.raw)
        self.asr.last_metadata = {"backend": "synthetic mock only", "audio_sha256": "0" * 64}
        self.asr_patch = patch("norlan.local_transcription.LocalTranscriber", return_value=self.asr)
        self.asr_patch.start()
        self.addCleanup(self.asr_patch.stop)
        self.connection_patch = patch("norlan.connection.production_brain_config",
                                      return_value={"provider": "codex", "billing": "subscription"})
        self.connection_patch.start()
        self.addCleanup(self.connection_patch.stop)
        self.brain = Mock(side_effect=self.brain_response)
        self.soundtrack = Mock(side_effect=AssertionError("test must not generate paid music"))

    def episode(self):
        return core.load_project(self.root)["episodes"][0]

    def shots(self):
        return copy.deepcopy(self.episode()["shots"])

    def brain_response(self, root, method, *args, **kwargs):
        if method == "proofread_subtitles":
            result = copy.deepcopy(args[0])
            for entry in result:
                entry["text"] = entry["text"].replace("张三", "张山")
            if args[1].get("feedback"):
                result[-1]["text"] = "用户要求校正后的字幕"
            return result
        if method == "arrange_music":
            context = args[0]
            track = context["catalog"][0]
            return {"summary": "仅测试的合成纯音配乐安排", "cues": [{"track_id": track["id"],
                "track_version": track["version"], "track_sha256": track["sha256"], "start": 0,
                "end": context["duration"], "source_start": 0, "gain_db": -20,
                "fade_in": 0.2, "fade_out": 0.2, "reason": "合成测试，无法评价实际音乐效果"}]}
        raise AssertionError("unexpected GPT method: " + method)

    def finish(self):
        return episode_post.finish(self.root, self.episode(), brain_call=self.brain,
                                   gate=core.load_project, soundtrack=self.soundtrack)

    def import_music(self, approve=False):
        track = music_library.import_track(self.root, self.fixture / "synthetic-music.wav",
                                           "测试用合成正弦波", track_id="synthetic-theme")
        track["file"] = str(Path(track["file"]).relative_to(self.root))
        with core.transaction(self.root) as state:
            state["music_library"] = {"version": 1, "tracks": [track]}
        if approve:
            episode_post.decide_music(self.root, track["id"], track["version"], track["sha256"])
        return track

    def calls(self, method):
        return [call for call in self.brain.call_args_list if call.args[1] == method]

    def test_actual_audio_asr_and_subscription_proofreading_then_first_music_waits(self):
        original = self.shots()
        self.assertFalse(self.finish())
        ep = self.episode()
        self.assertEqual(ep["postproduction"]["status"], "awaiting_music_selection")
        actual_audio = self.asr.transcribe.call_args.args[0]
        self.assertTrue(Path(actual_audio).is_file())
        self.assertTrue(Path(actual_audio).is_relative_to(self.root / "evidence"))
        self.assertTrue(media.probe(actual_audio)["has_audio"])
        call = self.calls("proofread_subtitles")[0]
        self.assertEqual(call.args[2], self.raw)
        self.assertEqual(call.args[3]["names"], [], "unused characters stay outside the proofreading context")
        self.assertNotIn("整部小说哨兵", json.dumps(call.args[3], ensure_ascii=False))
        corrected = ep["postproduction"]["subtitles"]["segments"]
        self.assertEqual(corrected[0]["text"], "张山说话")
        self.assertEqual([(s["start"], s["end"]) for s in corrected], [(s["start"], s["end"]) for s in self.raw])
        self.assertEqual(self.shots(), original)
        self.assertEqual(core.load_project(self.root)["jobs"], {})
        self.soundtrack.assert_not_called()

    def test_music_candidate_requires_first_audition_then_final_export_awaits_human(self):
        candidate = self.import_music()
        self.assertFalse(self.finish())
        self.assertEqual(self.episode()["postproduction"]["status"], "awaiting_music_approval")
        self.assertFalse(self.calls("arrange_music"))
        self.assertNotIn("file", self.episode())
        episode_post.decide_music(self.root, candidate["id"], 1, candidate["sha256"])
        self.assertTrue(self.finish())
        ep = self.episode()
        self.assertEqual(ep["status"], "awaiting_review")
        self.assertEqual(ep["postproduction"]["status"], "awaiting_final_approval")
        self.assertTrue(ep["subtitles_burned"])
        self.assertEqual(ep["music_status"], "approved_library_mixed")
        self.assertTrue(core.safe_path(self.root, ep["subtitle_file"]).is_file())
        self.assertEqual(len(self.calls("proofread_subtitles")), 1, "restart must reuse proofread subtitles")
        self.soundtrack.assert_not_called()
        approved = core.approve(self.root, "episode", ep["id"], ep["version"])
        self.assertEqual(approved["episodes"][0]["status"], "approved")

    def test_approved_music_library_is_reused_without_candidate_or_paid_call(self):
        self.import_music(approve=True)
        original = self.shots()
        self.assertTrue(self.finish())
        ep = self.episode()
        self.assertEqual(self.shots(), original)
        self.assertEqual(len(self.calls("arrange_music")), 1)
        self.assertEqual(len(core.load_project(self.root)["music_library"]["tracks"]), 1)
        self.assertNotIn("approved_at", ep)
        self.soundtrack.assert_not_called()
        episode_post.check_final(self.root, core.load_project(self.root), ep)

    def test_subtitle_only_revision_keeps_videos_and_music_and_requires_new_approval(self):
        self.import_music(approve=True)
        self.assertTrue(self.finish())
        original = self.episode()
        original_shots = self.shots()
        core.approve(self.root, "episode", original["id"], original["version"])
        episode_post.request(self.root, original["id"], original["version"], "subtitles", "只更正最后一句字幕")
        self.assertTrue(self.finish())
        updated = self.episode()
        self.assertEqual(self.shots(), original_shots)
        self.assertEqual(updated["version"], original["version"] + 1)
        self.assertNotIn("approved_at", updated)
        self.assertEqual(updated["postproduction"]["music_key"], original["postproduction"]["music_key"])
        self.assertEqual(len(self.calls("arrange_music")), 1)
        self.assertEqual(updated["render_manifest"]["subtitles"][-1]["text"], "用户要求校正后的字幕")
        self.assertTrue(core.safe_path(self.root, original["file"]).is_file())
        self.soundtrack.assert_not_called()

    def test_music_only_revision_keeps_corrected_subtitles_without_reproofreading(self):
        self.import_music(approve=True)
        self.assertTrue(self.finish())
        original = self.episode()
        original_shots = self.shots()
        episode_post.request(self.root, original["id"], original["version"], "music", "背景再轻一点")
        self.assertTrue(self.finish())
        updated = self.episode()
        self.assertEqual(self.shots(), original_shots)
        self.assertEqual(updated["render_manifest"]["subtitles"], original["render_manifest"]["subtitles"])
        self.assertEqual(len(self.calls("proofread_subtitles")), 1)
        self.assertEqual(len(self.calls("arrange_music")), 2)
        self.assertEqual(self.calls("arrange_music")[-1].args[2]["feedback"], "背景再轻一点")
        self.assertEqual(updated["status"], "awaiting_review")

    def test_restart_during_music_wait_reuses_proofreading_and_source(self):
        self.assertFalse(self.finish())
        first = self.episode()["postproduction"]
        with patch.object(media, "compose", wraps=media.compose) as compose:
            self.assertFalse(self.finish())
        second = self.episode()["postproduction"]
        self.assertEqual(len(self.calls("proofread_subtitles")), 1)
        self.assertEqual(first["subtitle_key"], second["subtitle_key"])
        self.assertEqual(first["dry_sha256"], second["dry_sha256"])
        compose.assert_not_called()
        self.soundtrack.assert_not_called()

    def test_caption_batches_are_bounded_and_do_not_drop_segments(self):
        self.raw = [{"start": i / 100, "end": (i + 0.5) / 100, "text": f"合成{i}"} for i in range(121)]
        self.asr.transcribe.return_value = self.raw
        self.assertFalse(self.finish())
        batches = self.calls("proofread_subtitles")
        self.assertEqual([len(call.args[2]) for call in batches], [60, 60, 1])
        self.assertEqual(self.episode()["postproduction"]["subtitles"]["segments"], self.raw)

    def test_final_approval_rejects_modified_srt_music_plan_and_final_bytes(self):
        self.import_music(approve=True)
        self.assertTrue(self.finish())
        baseline = core.load_project(self.root)
        ep = baseline["episodes"][0]
        srt = core.safe_path(self.root, ep["subtitle_file"])
        original_srt = srt.read_bytes()
        srt.write_bytes(original_srt + b"\nmodified\n")
        with self.assertRaises(core.GateError):
            core.approve(self.root, "episode", ep["id"], ep["version"])
        srt.write_bytes(original_srt)
        with core.transaction(self.root) as state:
            state["episodes"][0]["postproduction"]["music_key"] = "tampered-plan"
        with self.assertRaises(core.GateError):
            core.approve(self.root, "episode", ep["id"], ep["version"])
        with core.transaction(self.root) as state:
            state["episodes"][0]["postproduction"]["music_key"] = ep["postproduction"]["music_key"]
        final = core.safe_path(self.root, ep["file"])
        final.write_bytes(final.read_bytes() + b"tampered-final")
        with self.assertRaises(core.GateError):
            core.approve(self.root, "episode", ep["id"], ep["version"])

    def test_missing_local_asr_is_configuration_wait_without_cloud_fallback(self):
        self.asr.check_ready.side_effect = ValueError("请安装本地识别依赖")
        self.assertFalse(self.finish())
        self.assertEqual(self.episode()["postproduction"]["status"], "configuration_required")
        self.assertIn("请安装本地识别依赖", self.episode()["postproduction"]["missing_configuration"])
        self.asr.transcribe.assert_not_called()
        self.brain.assert_not_called()
        self.soundtrack.assert_not_called()

    def test_valid_but_changed_music_cue_plan_cannot_approve_an_old_mix(self):
        self.import_music(approve=True)
        self.assertTrue(self.finish())
        ep = self.episode()
        with core.transaction(self.root) as state:
            # Still within the legal gain range, but these are no longer the
            # instructions used to render the audio awaiting human approval.
            state["episodes"][0]["postproduction"]["music"]["cues"][0]["gain_db"] = -12
        with self.assertRaises(core.GateError):
            core.approve(self.root, "episode", ep["id"], ep["version"])


if __name__ == "__main__":
    unittest.main()
