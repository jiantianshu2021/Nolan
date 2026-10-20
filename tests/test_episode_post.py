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
        self.verifier = Mock()
        self.verifier.identity = {"backend": "synthetic-independent-ctc-only", "model_sha256": "b" * 64}
        self.verifier.last_metadata = {"synthetic_only": True, "changes": []}
        self.verifier.verify.side_effect = self.sync_observations
        self.verifier.repair.side_effect = self.sync_repair
        self.verifier_patch = patch("norlan.local_alignment.LocalSubtitleVerifier", return_value=self.verifier)
        self.verifier_patch.start()
        self.addCleanup(self.verifier_patch.stop)
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

    def sync_observations(self, audio, captions):
        return {"independent": True, "method": "independent_ctc", "full_audio_checked": True,
                "audio_sha256": core.digest(Path(audio)), "checker_identity": self.verifier.identity,
                "cues": [{"cue_index": i, "start": c["start"], "end": c["end"], "text": c["text"],
                          "coverage": 1.0, "consistent": True} for i, c in enumerate(captions)],
                "contradictions": []}

    def sync_repair(self, captions, report, attempt):
        for row in report["aligned_cues"]:
            captions[row["cue_index"]].update(start=row["start"], end=row["end"])
        return captions

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
        post = ep["postproduction"]
        self.assertEqual(post["music"]["render_policy"]["version"], 2)
        self.assertEqual(post["music_mix_check"]["status"], "passed")
        self.assertEqual(post["music_mix_check"]["final_sha256"], ep["sha256"])
        catalog = self.calls("arrange_music")[0].args[2]["catalog"]
        self.assertTrue(catalog[0]["audio_analysis"]["regions"])
        self.assertNotIn("envelope", catalog[0]["audio_analysis"])

    def test_new_mix_evidence_is_bound_to_export_and_cannot_be_relabelled_manual(self):
        self.import_music(approve=True)
        self.assertTrue(self.finish())
        original = core.load_project(self.root)
        for mutate in (
                lambda p: p["music_mix_check"].update(final_sha256="0" * 64),
                lambda p: p["music_mix_check"].update(status="manual"),
                lambda p: p["music_mix_check"].update(status="needs_attention"),
                lambda p: p.pop("music_mix_check")):
            state = copy.deepcopy(original)
            ep = state["episodes"][0]
            mutate(ep["postproduction"])
            with self.subTest(report=ep["postproduction"].get("music_mix_check", {}).get("status")):
                with self.assertRaises(core.GateError):
                    episode_post.check_final(self.root, state, ep)

    def test_interrupted_music_revision_reuses_fixed_feedback_snapshot_and_plan(self):
        self.import_music(approve=True)
        self.assertTrue(self.finish())
        original = self.episode()
        episode_post.request(self.root, original["id"], original["version"], "music", "换原曲中的另一个片段")

        def revised_brain(root, method, *args, **kwargs):
            result = self.brain_response(root, method, *args, **kwargs)
            if method == "arrange_music":
                result["cues"][0]["source_start"] = 1.
            return result

        self.brain.side_effect = revised_brain
        with patch.object(media, "compose", side_effect=media.MediaError("synthetic interruption before final export")):
            with self.assertRaises(media.MediaError):
                self.finish()
        interrupted = self.episode()["postproduction"]
        self.assertEqual(interrupted["music"]["cues"][0]["source_start"], 1.)
        self.assertEqual(interrupted["music_feedback_source"][0]["source_start"], 0.)
        self.assertTrue(self.finish())
        self.assertEqual(len(self.calls("arrange_music")), 2,
                         "resuming an arranged revision must not ask GPT again")
        self.assertEqual(self.episode()["postproduction"]["music_key"], interrupted["music_key"])

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
        asr_calls = self.asr.transcribe.call_count
        checker_calls = self.verifier.verify.call_count
        episode_post.request(self.root, original["id"], original["version"], "music", "背景再轻一点")
        self.assertTrue(self.finish())
        updated = self.episode()
        self.assertEqual(self.shots(), original_shots)
        self.assertEqual(updated["render_manifest"]["subtitles"], original["render_manifest"]["subtitles"])
        self.assertEqual(len(self.calls("proofread_subtitles")), 1)
        self.assertEqual(len(self.calls("arrange_music")), 2)
        self.assertEqual(self.calls("arrange_music")[-1].args[2]["feedback"], "背景再轻一点")
        self.assertEqual(updated["status"], "awaiting_review")
        self.assertEqual(self.asr.transcribe.call_count, asr_calls)
        self.assertEqual(self.verifier.verify.call_count, checker_calls)

    def test_timing_revision_uses_new_audio_windows_not_old_cues_and_keeps_music(self):
        self.import_music(approve=True)
        self.assertTrue(self.finish())
        original = self.episode()
        original_shots = self.shots()
        # Synthetic ASR changes both the number and positions of cues. This
        # models real re-alignment, not a single fixed offset or text-only edit.
        aligned = [{"start": 0.31, "end": 0.62, "text": "张三说话"},
                   {"start": 1.21, "end": 1.43, "text": "合成字幕"},
                   {"start": 1.52, "end": 1.94, "text": "样例"}]
        self.asr.transcribe.return_value = copy.deepcopy(aligned)
        # A folder scan may add unrelated candidates; this must not erase or
        # replan the already approved music during a subtitle-only request.
        candidate = music_library.import_track(self.root, self.fixture / "synthetic-music.wav",
                                               "无关试听候选", track_id="unrelated-candidate")
        candidate["file"] = str(Path(candidate["file"]).relative_to(self.root))
        with core.transaction(self.root) as state:
            state["music_library"]["tracks"].append(candidate)
        episode_post.request(self.root, original["id"], original["version"], "subtitles",
                             "前五条出现太早，请按实际人声重新对齐")
        self.assertTrue(self.finish())
        updated = self.episode()
        call = self.calls("proofread_subtitles")[-1]
        self.assertEqual(call.args[2], aligned, "old feedback timestamps must not override current audio")
        context = call.args[3]
        self.assertIn("张山说话", context["previous_corrected_text_reference_only"])
        self.assertIsInstance(context["previous_corrected_text_reference_only"], str)
        self.assertNotIn("subtitle_feedback_source", context)
        self.assertNotIn("整部小说哨兵", json.dumps(context, ensure_ascii=False))
        result = updated["postproduction"]["subtitles"]["segments"]
        self.assertEqual([(s["start"], s["end"]) for s in result],
                         [(s["start"], s["end"]) for s in aligned])
        self.assertEqual(updated["render_manifest"]["subtitles"], result)
        self.assertEqual(updated["postproduction"]["music_key"], original["postproduction"]["music_key"])
        self.assertEqual(updated["postproduction"]["music"]["sha256"], original["postproduction"]["music"]["sha256"])
        self.assertEqual(len(self.calls("arrange_music")), 1)
        self.assertEqual(self.shots(), original_shots)
        self.assertNotEqual(updated["file"], original["file"])
        self.assertEqual(updated["status"], "awaiting_review")
        self.assertTrue(core.safe_path(self.root, original["file"]).is_file())
        episode_post.check_final(self.root, core.load_project(self.root), updated)
        self.soundtrack.assert_not_called()

    def test_independent_sync_check_repairs_times_then_rechecks_before_final(self):
        self.import_music(approve=True)
        original_shots = self.shots()
        intended = [(0.41, 0.87), (1.22, 1.88)]
        def observed(audio, captions):
            result = self.sync_observations(audio, captions)
            for row, (start, end) in zip(result["cues"], intended):
                row.update(start=start, end=end)
            return result
        self.verifier.verify.side_effect = observed
        self.assertTrue(self.finish())
        updated = self.episode()
        sub = updated["postproduction"]["subtitles"]
        self.assertEqual([(c["start"], c["end"]) for c in sub["segments"]], intended)
        self.assertEqual(sub["sync_check"]["status"], "passed")
        self.assertEqual(sub["sync_check"]["attempts"], 1)
        self.assertEqual(self.verifier.verify.call_count, 2)
        self.assertEqual(self.verifier.repair.call_count, 1)
        self.assertEqual(len(self.calls("proofread_subtitles")), 1)
        self.assertEqual(self.shots(), original_shots)
        self.assertEqual(updated["status"], "awaiting_review")
        episode_post.check_final(self.root, core.load_project(self.root), updated)
        self.soundtrack.assert_not_called()

    def test_unresolved_sync_is_bounded_and_continue_never_resets_attempts(self):
        self.import_music(approve=True)
        original = self.shots()
        def uncertain(audio, captions):
            result = self.sync_observations(audio, captions)
            result["cues"][0]["consistent"] = False
            return result
        self.verifier.verify.side_effect = uncertain
        self.assertTrue(self.finish())
        post = self.episode()["postproduction"]
        self.assertEqual(post["status"], "subtitle_sync_needs_attention")
        self.assertEqual(post["subtitles"]["sync_check"]["attempts"], 2)
        self.assertTrue(post["final_ready"])
        self.assertEqual(post["preview_kind"], "subtitle_diagnostic")
        self.assertTrue(self.episode()["subtitles_burned"])
        self.assertTrue(core.safe_path(self.root, self.episode()["file"]).is_file())
        with self.assertRaisesRegex(core.GateError, "字幕同步自动检查尚未通过"):
            episode_post.check_final(self.root, core.load_project(self.root), self.episode())
        verify_count = self.verifier.verify.call_count
        export = self.episode()["file"]
        self.assertFalse(self.finish())
        self.assertEqual(self.verifier.verify.call_count, verify_count)
        self.assertEqual(self.episode()["file"], export)
        self.assertEqual(self.verifier.repair.call_count, 2)
        self.assertEqual(len(self.calls("proofread_subtitles")), 1)
        self.assertEqual(len(self.calls("arrange_music")), 1)
        self.assertEqual(self.shots(), original)
        # Production marks the episode generating before reentering finish.
        # Reusing a diagnostic must restore the viewable review state each time.
        from norlan import pipeline, dashboard
        with core.transaction(self.root) as state:
            state["episodes"][0]["reviews"] = []
            # Public projection uses the current list-shaped attempt history.
            for shot in state["episodes"][0]["shots"]:
                shot["attempts"] = []
        synthetic_cut = {"key": "synthetic-cached-diagnostic-cut", "sha256": "e" * 64,
                         "review_skipped": {"reason": "disabled", "sha256": "e" * 64}}
        with patch.object(pipeline, "_gate", side_effect=core.load_project), \
                patch.object(pipeline, "_shot", return_value=True), \
                patch.object(pipeline, "_rough_cut", return_value=synthetic_cut):
            for _ in range(2):
                pipeline._episode(self.root, "E001")
                self.assertEqual(self.episode()["status"], "awaiting_review")
                public = dashboard.public_state(core.load_project(self.root))["episodes"][0]
                self.assertTrue(public["postproduction"]["final_available"])
                self.assertFalse(public["postproduction"]["final_ready"])
        self.assertEqual(self.verifier.verify.call_count, verify_count)
        # An explicit new subtitle opinion starts a new bounded repair cycle.
        ep = self.episode()
        episode_post.request(self.root, ep["id"], ep["version"], "subtitles", "重新检查这句实际声音")
        self.verifier.verify.side_effect = self.sync_observations
        self.assertTrue(self.finish())
        self.assertEqual(self.episode()["postproduction"]["subtitles"]["sync_check"]["attempts"], 0)
        self.soundtrack.assert_not_called()

    def test_spelling_conflict_can_restore_audio_text_without_resetting_repair_budget(self):
        self.import_music(approve=True)
        self.raw[0]["text"] = "让我看了"
        self.asr.transcribe.return_value = copy.deepcopy(self.raw)
        def script_spelling(root, method, *args, **kwargs):
            result = self.brain_response(root, method, *args, **kwargs)
            if method == "proofread_subtitles":
                result[0]["text"] = "账我看了。"
            return result
        self.brain.side_effect = script_spelling
        def words_from_audio(audio, captions):
            result = self.sync_observations(audio, captions)
            for row in result["cues"]:
                row["boundary_supported"] = True
            if captions[0]["text"] != self.raw[0]["text"]:
                result["cues"][0].update(coverage=.75, consistent=False, boundary_supported=False)
            return result
        self.verifier.verify.side_effect = words_from_audio
        original_shots = self.shots()
        self.assertTrue(self.finish())
        subtitles = self.episode()["postproduction"]["subtitles"]
        self.assertEqual(subtitles["segments"][0]["text"], self.raw[0]["text"])
        self.assertEqual(subtitles["sync_check"]["status"], "passed")
        self.assertEqual(subtitles["sync_check"]["attempts"], 2)
        self.assertEqual(subtitles["previous_sync_check"]["status"], "needs_attention")
        self.assertEqual(subtitles["text_recovery"]["status"], "recovered")
        self.assertEqual(self.shots(), original_shots)
        self.assertEqual(self.episode()["status"], "awaiting_review")
        episode_post.check_final(self.root, core.load_project(self.root), self.episode())
        self.soundtrack.assert_not_called()

    def test_missing_or_changed_sync_report_blocks_final_human_approval(self):
        self.import_music(approve=True)
        self.assertTrue(self.finish())
        baseline = core.load_project(self.root)
        ep = baseline["episodes"][0]
        with core.transaction(self.root) as state:
            state["episodes"][0]["postproduction"]["subtitles"].pop("sync_check")
        with self.assertRaisesRegex(core.GateError, "同步自动检查尚未通过"):
            core.approve(self.root, "episode", ep["id"], ep["version"])
        with core.transaction(self.root) as state:
            state["episodes"][0]["postproduction"]["subtitles"] = copy.deepcopy(ep["postproduction"]["subtitles"])
            state["episodes"][0]["postproduction"]["subtitles"]["sync_check"]["evidence_hash"] = "e" * 64
        with self.assertRaisesRegex(core.GateError, "记录与当前音轨或字幕不一致"):
            core.approve(self.root, "episode", ep["id"], ep["version"])

    def test_continue_reenters_postproduction_for_legacy_export_without_sync_report(self):
        from norlan import pipeline
        self.import_music(approve=True)
        self.assertTrue(self.finish())
        ep = self.episode()
        with patch.object(pipeline, "_gate", side_effect=core.load_project), \
                patch.object(pipeline, "_shot", side_effect=AssertionError("reentered production")) as shot:
            pipeline._episode(self.root, ep["id"])
            shot.assert_not_called()
            with core.transaction(self.root) as state:
                state["episodes"][0]["postproduction"]["subtitles"].pop("sync_check")
            with self.assertRaisesRegex(AssertionError, "reentered production"):
                pipeline._episode(self.root, ep["id"])
            shot.assert_called_once()

    def test_legacy_subtitle_check_keeps_previous_export_and_its_provenance(self):
        self.import_music(approve=True)
        self.assertTrue(self.finish())
        original = self.episode()
        with core.transaction(self.root) as state:
            state["episodes"][0]["postproduction"]["subtitles"].pop("sync_check")
        self.assertTrue(self.finish())
        updated = self.episode()
        self.assertEqual(updated["history"][-1]["file"], original["file"])
        self.assertEqual(updated["history"][-1]["sha256"], original["sha256"])
        self.assertEqual(updated["history"][-1]["render_manifest"], original["render_manifest"])
        self.assertEqual(core.digest(core.safe_path(self.root, original["file"])), original["sha256"])
        self.assertEqual(updated["version"], original["version"] + 1)

    def test_manual_music_slider_and_time_edits_render_without_gpt_or_generation(self):
        from norlan.post_edits import apply_edits
        self.import_music(approve=True)
        self.assertTrue(self.finish())
        original = self.episode()
        original_shots = self.shots()
        self.brain.reset_mock()
        cue = original["postproduction"]["music"]["cues"][0]
        apply_edits(self.root, original["id"], original["version"], original["postproduction"].get("revision", 0),
                    "music", [{"index": 0, "start": .1, "end": 1.8, "gain_db": -30}])
        self.assertTrue(self.finish())
        updated = self.episode()
        revised = updated["postproduction"]["music"]["cues"][0]
        self.assertEqual((revised["start"], revised["end"], revised["gain_db"]), (.1, 1.8, -30))
        self.assertEqual(revised["track_sha256"], cue["track_sha256"])
        self.assertEqual(updated["render_manifest"]["subtitles"], original["render_manifest"]["subtitles"])
        self.assertEqual(self.shots(), original_shots)
        self.assertNotEqual(updated["file"], original["file"])
        self.brain.assert_not_called()
        self.soundtrack.assert_not_called()
        episode_post.check_final(self.root, core.load_project(self.root), updated)

    def test_manual_pipeline_goes_directly_to_local_postproduction(self):
        from norlan import pipeline
        from norlan.post_edits import apply_edits
        self.import_music(approve=True)
        self.assertTrue(self.finish())
        original = self.episode()
        apply_edits(self.root, original["id"], original["version"], original["postproduction"].get("revision", 0),
                    "music", [{"index": 0, "start": .1, "end": 1.8, "gain_db": -30}])
        with patch.object(pipeline, "_gate", side_effect=core.load_project), \
                patch.object(pipeline, "_shot", side_effect=AssertionError("manual must not review shots")), \
                patch.object(episode_post, "finish", return_value=True) as local:
            pipeline._episode(self.root, original["id"])
        local.assert_called_once()

    def test_manual_run_never_advances_unrelated_assets_or_model_calls(self):
        from norlan import pipeline
        from norlan.post_edits import apply_edits
        self.import_music(approve=True)
        self.assertTrue(self.finish())
        original = self.episode()
        apply_edits(self.root, original["id"], original["version"], original["postproduction"].get("revision", 0),
                    "music", [{"index": 0, "start": .1, "end": 1.8, "gain_db": -30}])
        with patch.object(pipeline, "_gate", side_effect=core.load_project), \
                patch.object(pipeline, "_recover", side_effect=AssertionError("no provider recovery")), \
                patch.object(pipeline, "_assets", side_effect=AssertionError("no asset generation")), \
                patch.object(pipeline, "_brain", side_effect=AssertionError("no GPT")):
            result = pipeline.run(self.root)
        self.assertEqual(result["episodes"][0]["status"], "awaiting_review")
        self.assertEqual(result["episodes"][0]["version"], original["version"] + 1)

    def test_manual_subtitle_times_are_preserved_without_rechecking(self):
        from norlan.post_edits import apply_edits
        self.import_music(approve=True)
        self.assertTrue(self.finish())
        original = self.episode()
        original_shots = self.shots()
        self.brain.reset_mock(); self.asr.transcribe.reset_mock(); self.verifier.repair.reset_mock(); self.verifier.verify.reset_mock()
        edits = [{"index": i, "start": c["start"] + .04, "end": c["end"] + .04, "text": c["text"]}
                 for i, c in enumerate(original["postproduction"]["subtitles"]["segments"])]
        apply_edits(self.root, original["id"], original["version"], original["postproduction"].get("revision", 0), "subtitles", edits)
        self.assertTrue(self.finish())
        updated = self.episode()
        self.assertTrue(updated["postproduction"]["music_mix_check"]["reused_prepared_music"])
        self.assertEqual(updated["postproduction"]["music_mix_check"]["prepared_music_sha256"],
                         original["postproduction"]["music_mix_check"]["prepared_music_sha256"],
                         "editing captions must retain the previously balanced music exactly")
        self.assertEqual(updated["render_manifest"]["subtitles"],
            [{"start": round(c["start"], 6), "end": round(c["end"], 6), "text": c["text"]} for c in edits])
        self.assertEqual(updated["postproduction"]["music"]["sha256"], original["postproduction"]["music"]["sha256"])
        self.assertEqual(self.shots(), original_shots)
        self.brain.assert_not_called(); self.asr.transcribe.assert_not_called(); self.verifier.repair.assert_not_called(); self.verifier.verify.assert_not_called()
        episode_post.check_final(self.root, core.load_project(self.root), updated)

    def test_manual_times_skip_checker_and_wait_for_final_human_decision(self):
        from norlan.post_edits import apply_edits
        self.import_music(approve=True)
        self.assertTrue(self.finish())
        original = self.episode()
        self.verifier.repair.reset_mock(); self.verifier.verify.reset_mock(); self.verifier.check_ready.reset_mock()
        edits = [{"index": i, **c} for i, c in enumerate(original["postproduction"]["subtitles"]["segments"])]
        edits[0].update(start=.02, end=.2)
        apply_edits(self.root, original["id"], original["version"], original["postproduction"].get("revision", 0), "subtitles", edits)
        self.verifier.verify.side_effect = AssertionError("manual edits must not trigger sync checks")
        self.assertTrue(self.finish())
        updated = self.episode()
        self.assertNotEqual(updated["file"], original["file"])
        self.assertEqual(updated["render_manifest"]["subtitles"][0]["start"], .02)
        self.assertFalse(updated["postproduction"]["manual_preview"])
        self.assertEqual(updated["postproduction"]["status"], "awaiting_final_approval")
        self.assertEqual(updated["postproduction"]["subtitles"]["sync_check"]["status"], "manual")
        self.assertEqual(updated["status"], "awaiting_review")
        self.assertNotIn("approved_at", updated)
        self.verifier.repair.assert_not_called(); self.verifier.verify.assert_not_called(); self.verifier.check_ready.assert_not_called()
        episode_post.check_final(self.root, core.load_project(self.root), updated)
        with core.transaction(self.root) as state:
            state["episodes"][0]["postproduction"]["manual_subtitles"]["segments"][0]["start"] = .03
        with self.assertRaises(core.GateError):
            episode_post.check_final(self.root, core.load_project(self.root), self.episode())

    def test_subtitle_only_revision_does_not_rearrange_invalid_music(self):
        self.import_music(approve=True)
        self.assertTrue(self.finish())
        original = self.episode()
        bed = core.safe_path(self.root, original["postproduction"]["music"]["file"])
        bed.write_bytes(bed.read_bytes() + b"changed synthetic bed")
        episode_post.request(self.root, original["id"], original["version"], "subtitles", "重新对齐字幕")
        with self.assertRaisesRegex(core.GateError, "不会自动重新编配"):
            self.finish()
        self.assertEqual(len(self.calls("arrange_music")), 1)
        self.assertFalse(self.episode()["postproduction"]["final_ready"])
        self.assertEqual(self.episode()["postproduction"]["subtitles"]["status"], "ready")
        self.soundtrack.assert_not_called()

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
