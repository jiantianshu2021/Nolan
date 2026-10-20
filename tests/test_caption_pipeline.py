"""Caption lifecycle tests with real state transitions and zero paid calls."""
import copy
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from norlan import core, pipeline
from norlan.providers import ProviderError
from tests.test_pipeline import choose_gpt_assets, fake_media, write, run_with_shot_approvals


class CaptionPipelineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        core.create_project(self.root, "剧本写了：这是计划台词，不能冒充转录。", mode="demo")
        self.media_context = fake_media()
        self.media_context.__enter__()
        self.addCleanup(self.media_context.__exit__, None, None, None)
        state = pipeline.run(self.root)
        core.approve(self.root, "plan", "plan", state["plan"]["version"])
        choose_gpt_assets(self.root)
        state = pipeline.run(self.root)
        for asset in state["assets"]:
            core.approve(self.root, "asset", asset["id"], asset["version"])
        state = run_with_shot_approvals(self.root)
        self.assertEqual(state["status"], "awaiting_episode_review", state.get("last_error"))
        # Recompose already accepted real-state clips with ASR enabled. No video,
        # visual review, or voice service is needed or called by these tests.
        core.configure(self.root, {"subtitles": {"mode": "openai",
                                                   "model": "whisper-1", "language": "zh"}})
        with core.transaction(self.root) as state:
            state["mode"] = "live"
            episode = state["episodes"][0]
            episode["status"] = "revision_requested"
            for shot in episode["shots"]:
                shot["dialogue"] = "完全不同的计划台词"
        self.segments = [{"start": 0.25, "end": 1.5, "text": "实际音轨里的台词"},
                         {"start": 4.2, "end": 6.0, "text": "这是张三"}]

    def episode(self):
        return core.load_project(self.root)["episodes"][0]

    def transcriber(self, segments=None):
        service = Mock()
        service.last_request_id = "test-asr-request"
        service.transcribe.return_value = copy.deepcopy(self.segments if segments is None else segments)
        return service

    def asr_jobs(self):
        return [job for job in core.load_project(self.root)["jobs"].values()
                if job["kind"] == "transcription"]

    def test_transcribes_actual_normalized_cut_not_planned_dialogue(self):
        service = self.transcriber()
        pipeline.media.compose.reset_mock()
        with patch("norlan.transcription.Transcriber", return_value=service), patch.object(pipeline, "_brain") as brain, patch.object(pipeline, "_ark") as ark:
            state = pipeline.run(self.root)
        self.assertEqual(state["status"], "awaiting_episode_review", state.get("last_error"))
        actual_audio = service.transcribe.call_args.args[0]
        self.assertIsInstance(actual_audio, Path)
        self.assertEqual(actual_audio.read_bytes(), b"voice")
        self.assertTrue(actual_audio.is_relative_to(self.root / "evidence"))
        self.assertTrue(actual_audio.parent.name.startswith("asr-"))
        calls = pipeline.media.compose.call_args_list
        dry = next(call for call in calls if "-dialogue-" in str(call.args[1]))
        final = calls[-1]
        self.assertEqual(dry.args[0], [core.safe_path(self.root, shot["file"]) for shot in self.episode()["shots"]])
        self.assertNotIn("subtitles", dry.kwargs)
        self.assertEqual(final.kwargs["subtitles"], self.segments)
        self.assertNotIn("计划台词", str(final.kwargs["subtitles"]))
        self.assertEqual(self.episode()["subtitle_status"], "automatic_asr_needs_review")
        self.assertEqual(self.asr_jobs()[0]["accounting"], "spent")
        self.assertTrue(self.asr_jobs()[0]["cost_track"])
        self.assertEqual(core.load_project(self.root)["budget"]["cost_pending_jobs"], 1)
        self.assertIsNone(core.load_project(self.root)["budget"]["actual_cost"])
        service.transcribe.assert_called_once()
        brain.assert_not_called()
        ark.assert_not_called()

    def test_completed_cache_avoids_a_second_billable_call(self):
        service = self.transcriber()
        with patch("norlan.transcription.Transcriber", return_value=service):
            first, status = pipeline._captions(self.root, self.episode())
            second, _ = pipeline._captions(self.root, self.episode())
        self.assertEqual(first, second)
        self.assertEqual(status, "automatic_asr_needs_review")
        service.transcribe.assert_called_once()
        self.assertEqual(len(self.asr_jobs()), 1)
        self.assertTrue(self.asr_jobs()[0]["cost_track"])
        self.assertEqual(core.load_project(self.root)["budget"]["cost_pending_jobs"], 1)

    def test_uncertain_asr_stops_all_automatic_resubmission(self):
        service = self.transcriber()
        service.transcribe.side_effect = ProviderError("connection lost", uncertain=True)
        with patch("norlan.transcription.Transcriber", return_value=service):
            first = pipeline.run(self.root)
            second = pipeline.run(self.root)
            with self.assertRaises(core.GateError):
                pipeline._captions(self.root, self.episode())
        self.assertEqual(first["status"], "blocked")
        self.assertEqual(second["status"], "blocked")
        self.assertEqual(self.asr_jobs()[0]["accounting"], "unknown")
        self.assertEqual(self.asr_jobs()[0]["amount"], 0)
        self.assertTrue(self.asr_jobs()[0]["cost_track"])
        self.assertEqual(second["budget"]["cost_pending_jobs"], 1)
        self.assertIsNone(second["budget"]["actual_cost"])
        service.transcribe.assert_called_once()
        self.assertEqual(len(self.asr_jobs()), 1)

    def test_missing_provider_key_creates_no_submission(self):
        before = len(core.load_project(self.root)["jobs"])
        with patch.dict(os.environ, {}, clear=True), patch("norlan.transcription._open") as network:
            with self.assertRaises(ProviderError):
                pipeline._captions(self.root, self.episode())
        self.assertEqual(len(core.load_project(self.root)["jobs"]), before)
        self.assertEqual(core.load_project(self.root)["budget"]["spent"], 0)
        network.assert_not_called()

    def test_caption_text_fix_preserves_clips_and_requires_new_episode_approval(self):
        service = self.transcriber()
        with patch("norlan.transcription.Transcriber", return_value=service):
            state = pipeline.run(self.root)
        original = copy.deepcopy(state["episodes"][0])
        core.approve(self.root, "episode", original["id"], original["version"])
        core.request_caption_fix(self.root, original["id"], original["version"], "把张三改成张山，仅修改字幕")
        after_request = self.episode()
        self.assertEqual(after_request["status"], "revision_requested")
        self.assertNotIn("approved_sha256", after_request)
        corrected = copy.deepcopy(self.segments)
        corrected[1]["text"] = "这是张山"
        with patch.object(pipeline, "_brain", return_value=corrected) as brain, patch.object(pipeline, "_video") as video, patch("norlan.transcription.Transcriber") as asr:
            state = pipeline.run(self.root)
        current = state["episodes"][0]
        self.assertEqual(state["status"], "awaiting_episode_review", state.get("last_error"))
        self.assertEqual(current["version"], original["version"] + 1)
        self.assertEqual(current["render_manifest"]["subtitles"], corrected)
        self.assertEqual(current["subtitle_status"], "human_requested_text_edit_needs_review")
        self.assertEqual([(s["file"], s["sha256"], s["version"], s["attempts"]) for s in current["shots"]],
                         [(s["file"], s["sha256"], s["version"], s["attempts"]) for s in original["shots"]])
        self.assertTrue((self.root / original["file"]).is_file())
        self.assertEqual(brain.call_args.args[1], "correct_subtitles")
        self.assertEqual(brain.call_args.args[2], self.segments)
        self.assertEqual(len(self.asr_jobs()), 1)
        asr.assert_not_called()
        video.assert_not_called()
        with self.assertRaises(core.GateError):
            core.approve(self.root, "episode", current["id"], original["version"])
        core.approve(self.root, "episode", current["id"], current["version"])

    def test_changed_shot_invalidates_old_corrected_subtitles_and_asr_cache(self):
        service = self.transcriber()
        with patch("norlan.transcription.Transcriber", return_value=service):
            pipeline.run(self.root)
        original = self.episode()
        core.request_caption_fix(self.root, "E001", original["version"], "字幕张三改成张山")
        corrected = copy.deepcopy(self.segments)
        corrected[1]["text"] = "这是张山"
        with patch.object(pipeline, "_brain", return_value=corrected):
            pipeline.run(self.root)
        self.assertEqual(self.episode()["verified_subtitles"], corrected)
        original = self.episode()
        first_fingerprint = self.asr_jobs()[0]["fingerprint"]
        core.reject(self.root, "episode", "E001", original["version"], "实际对白重新生成", ["E001-S002"])
        self.assertNotIn("verified_subtitles", self.episode())
        self.assertEqual(self.episode()["subtitle_status"], "needs_retiming")
        # Stand in for successful regeneration, retaining the actual replacement
        # bytes and hash so cache identity changes as it would in production.
        replacement = write(self.root / "media" / "replacement.mp4", b"different picture and speech")
        with core.transaction(self.root) as state:
            shot = state["episodes"][0]["shots"][1]
            shot.update(status="accepted", file=str(replacement.relative_to(self.root)), sha256=core.digest(replacement))
        fresh_segments = [{"start": 0.1, "end": 2.0, "text": "新镜头里不同的实际对白"}]
        second_service = self.transcriber(fresh_segments)
        with patch("norlan.transcription.Transcriber", return_value=second_service):
            result, _ = pipeline._captions(self.root, self.episode())
        self.assertEqual(result, fresh_segments)
        self.assertEqual(len(self.asr_jobs()), 2)
        self.assertNotEqual(first_fingerprint, self.asr_jobs()[1]["fingerprint"])
        second_service.transcribe.assert_called_once()

    def test_pending_text_fix_cannot_reuse_the_old_timeline_after_clip_rework(self):
        service = self.transcriber()
        with patch("norlan.transcription.Transcriber", return_value=service):
            pipeline.run(self.root)
        original = self.episode()
        core.request_caption_fix(self.root, "E001", original["version"], "字幕改字")
        core.reject(self.root, "episode", "E001", original["version"], "先修改真实对白", ["E001-S002"])
        replacement = write(self.root / "media" / "replacement.mp4", b"replacement speech")
        with core.transaction(self.root) as state:
            shot = state["episodes"][0]["shots"][1]
            shot.update(status="accepted", file=str(replacement.relative_to(self.root)), sha256=core.digest(replacement))
        with patch.object(pipeline, "_brain") as brain, patch("norlan.transcription.Transcriber", return_value=self.transcriber()):
            pipeline._captions(self.root, self.episode())
        brain.assert_not_called()
        self.assertFalse(self.episode().get("caption_feedback"))
        self.assertEqual(len(self.asr_jobs()), 2)


if __name__ == "__main__":
    unittest.main()
