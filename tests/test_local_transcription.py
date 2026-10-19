"""Synthetic local-ASR fixtures; these tests never install models or use APIs."""
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from norlan.local_transcription import LocalTranscriber, _cpp_segments
from norlan.providers import ProviderError


class LocalTranscriptionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.audio = self.root / "synthetic 录音.wav"
        self.audio.write_bytes(b"synthetic audio bytes")
        self.model = self.root / "synthetic model.bin"
        self.model.write_bytes(b"synthetic model")
        self.executable = self.root / "synthetic whisper-cli"
        self.executable.write_bytes(b"synthetic executable")
        self.config = {"mode": "local", "model_path": str(self.model), "language": "zh"}
        self.result = {"transcription": [{"offsets": {"from": 200, "to": 2100}, "text": " 实际对白 "}]}
        self.which = patch("norlan.local_transcription.shutil.which", return_value=str(self.executable))
        self.which.start()
        self.addCleanup(self.which.stop)
        self.probe = patch("norlan.local_transcription.probe", return_value={"duration": 6.0, "has_audio": True})
        self.probe.start()
        self.addCleanup(self.probe.stop)
        self.transcriber = LocalTranscriber(self.config, cache_dir=self.root / "cache")

    def run_fixture(self, args, **kwargs):
        if "-of" in args:
            output = Path(args[args.index("-of") + 1]).with_suffix(".json")
            output.write_text(json.dumps(self.result), encoding="utf-8")
        else:
            Path(args[-1]).write_bytes(b"synthetic normalized PCM")
        return subprocess.CompletedProcess(args, 0)

    def test_millisecond_timestamps_actual_audio_and_safe_argv(self):
        with patch("norlan.local_transcription.subprocess.run", side_effect=self.run_fixture) as run:
            result = self.transcriber.transcribe(self.audio)
        self.assertEqual(result, [{"start": 0.2, "end": 2.1, "text": "实际对白"}])
        self.assertEqual(run.call_count, 2)
        ffmpeg, asr = [call.args[0] for call in run.call_args_list]
        self.assertIn(str(self.audio.resolve()), ffmpeg)
        self.assertIn("16000", ffmpeg)
        self.assertIn("pcm_s16le", ffmpeg)
        self.assertEqual(asr[0], str(self.executable.resolve()))
        self.assertIn(str(self.model.resolve()), asr)
        self.assertNotIn("--prompt", asr)
        self.assertNotIn("--translate", asr)
        self.assertFalse(any(call.kwargs.get("shell") for call in run.call_args_list))
        self.assertEqual(self.transcriber.last_metadata["cache_hit"], False)

    def test_content_cache_skips_unchanged_recognition_and_changes_invalidate(self):
        with patch("norlan.local_transcription.subprocess.run", side_effect=self.run_fixture) as run:
            first = self.transcriber.transcribe(self.audio)
            first_key = self.transcriber.last_metadata["cache_key"]
            self.assertEqual(self.transcriber.transcribe(self.audio), first)
            self.assertEqual(run.call_count, 2)
            self.assertTrue(self.transcriber.last_metadata["cache_hit"])
            self.audio.write_bytes(b"new synthetic audio")
            self.transcriber.transcribe(self.audio)
            self.assertEqual(run.call_count, 4)
            self.assertNotEqual(first_key, self.transcriber.last_metadata["cache_key"])
            self.model.write_bytes(b"new synthetic model")
            self.transcriber.transcribe(self.audio)
            self.assertEqual(run.call_count, 6)
            self.executable.write_bytes(b"new synthetic executable")
            self.transcriber.transcribe(self.audio)
            self.assertEqual(run.call_count, 8)

    def test_cache_can_be_reused_after_source_renamed_without_rerunning(self):
        with patch("norlan.local_transcription.subprocess.run", side_effect=self.run_fixture) as run:
            original = self.transcriber.transcribe(self.audio)
            other = self.root / "renamed.wav"
            other.write_bytes(self.audio.read_bytes())
            self.assertEqual(self.transcriber.transcribe(other), original)
            self.assertEqual(run.call_count, 2)

    def test_corrupt_cache_is_recomputed_not_used(self):
        with patch("norlan.local_transcription.subprocess.run", side_effect=self.run_fixture) as run:
            self.transcriber.transcribe(self.audio)
            cached = next((self.root / "cache").glob("*.json"))
            cached.write_text("{broken", encoding="utf-8")
            self.transcriber.transcribe(self.audio)
            self.assertEqual(run.call_count, 4)
            self.assertFalse(self.transcriber.last_metadata["cache_hit"])

    def test_silence_can_return_no_segments_without_script_substitution(self):
        self.result = {"transcription": []}
        with patch("norlan.local_transcription.subprocess.run", side_effect=self.run_fixture):
            self.assertEqual(self.transcriber.transcribe(self.audio), [])

    def test_missing_dependencies_have_actionable_errors_and_never_run(self):
        with patch("norlan.local_transcription.shutil.which", return_value=None), patch("norlan.local_transcription.subprocess.run") as run:
            with self.assertRaisesRegex(ProviderError, "brew install whisper-cpp"):
                self.transcriber.transcribe(self.audio)
            run.assert_not_called()
        self.model.unlink()
        with patch("norlan.local_transcription.subprocess.run") as run:
            with self.assertRaisesRegex(ProviderError, "model_path"):
                self.transcriber.transcribe(self.audio)
            run.assert_not_called()

    def test_invalid_audio_stops_before_asr(self):
        with patch("norlan.local_transcription.probe", return_value={"duration": 6, "has_audio": False}), patch("norlan.local_transcription.subprocess.run") as run:
            with self.assertRaises(ProviderError) as error:
                self.transcriber.transcribe(self.audio)
            self.assertFalse(error.exception.uncertain)
            run.assert_not_called()

    def test_timeout_is_local_failure_no_retry_or_sensitive_error_output(self):
        with patch("norlan.local_transcription.subprocess.run", side_effect=subprocess.TimeoutExpired("synthetic", 600, stderr=b"private bytes")) as run:
            with self.assertRaisesRegex(ProviderError, "超时") as error:
                self.transcriber.transcribe(self.audio)
        self.assertEqual(run.call_count, 1)
        self.assertFalse(error.exception.uncertain)
        self.assertNotIn("private", str(error.exception))
        self.assertFalse(list((self.root / "cache").glob("*.json")))

    def test_audio_changed_during_conversion_is_not_published(self):
        def changing(args, **kwargs):
            result = self.run_fixture(args, **kwargs)
            self.audio.write_bytes(b"modified during recognition")
            return result
        with patch("norlan.local_transcription.subprocess.run", side_effect=changing) as run:
            with self.assertRaisesRegex(ProviderError, "发生变化"):
                self.transcriber.transcribe(self.audio)
            self.assertEqual(run.call_count, 1)
            self.assertFalse(list((self.root / "cache").glob("*.json")))

    def test_invalid_json_and_timestamps_are_not_cached(self):
        for result in ({}, {"transcription": "text"}, {"transcription": [], "text": "untimed"},
                       {"transcription": [{"offsets": {"from": 1, "to": 7000}, "text": "beyond"}]},
                       {"transcription": [{"offsets": {"from": False, "to": 2000}, "text": "bool"}]},
                       {"transcription": [{"offsets": {"from": 1, "to": 2000}, "text": "x" * 4001}]}):
            with self.subTest(result=str(result)[:80]):
                self.result = result
                with patch("norlan.local_transcription.subprocess.run", side_effect=self.run_fixture):
                    with self.assertRaises(ProviderError) as error:
                        self.transcriber.transcribe(self.audio)
                    self.assertFalse(error.exception.uncertain)
                self.assertFalse(list((self.root / "cache").glob("*.json")))

    def test_overlap_and_small_rounding(self):
        with self.assertRaises(ValueError):
            _cpp_segments({"transcription": [
                {"offsets": {"from": 0, "to": 3000}, "text": "a"},
                {"offsets": {"from": 2500, "to": 4000}, "text": "b"}]}, 6)
        self.assertEqual(_cpp_segments({"transcription": [{"offsets": {"from": 5000, "to": 6020}, "text": "结束"}]}, 6)[0]["end"], 6)

    def test_configuration_validation(self):
        for config in ({"mode": "openai"}, {"backend": "api"}, {"language": "Chinese"},
                       {"timeout": True}, {"timeout": float("nan")}, {"threads": True},
                       {"threads": 99}, {"executable": ""}, {"model_path": "\x00"}):
            with self.subTest(config=config):
                with self.assertRaises(ProviderError):
                    LocalTranscriber({**self.config, **config})


if __name__ == "__main__":
    unittest.main()
