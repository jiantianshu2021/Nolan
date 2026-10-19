"""VAD original-clock alignment; no project approval, paid call or model download."""
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
import wave
from unittest.mock import patch

from norlan.local_transcription import LocalTranscriber, _crop_pcm, _cpp_segments, _vad_windows, _merge_speech_windows, _recognition_window
from norlan.providers import ProviderError


def write_pcm(path, frames):
    with wave.open(str(path), "wb") as sound:
        sound.setnchannels(1); sound.setsampwidth(2); sound.setframerate(16000)
        sound.writeframes(frames)


class LocalSpeechTimingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.audio = self.root / "original.wav"
        write_pcm(self.audio, b"\x00\x00" * 128000)
        self.model = self.root / "whisper.bin"; self.model.write_bytes(b"whisper model")
        self.vad_model = self.root / "silero.bin"; self.vad_model.write_bytes(b"vad model")
        self.exe = self.root / "whisper-cli"; self.exe.write_bytes(b"whisper exe")
        self.vad = self.root / "whisper-vad-speech-segments"; self.vad.write_bytes(b"vad exe")
        self.config = {"model_path": str(self.model), "vad_model_path": str(self.vad_model)}
        self.windows = "Detected 2 speech segments:\nSpeech segment 0: start = 240.00, end = 320.00\nSpeech segment 1: start = 540.00, end = 620.00\n"
        self.calls = []
        self.stop = False
        self.which = patch("norlan.local_transcription.shutil.which", side_effect=lambda name: str(self.vad if name == "whisper-vad-speech-segments" else self.exe))
        self.which.start(); self.addCleanup(self.which.stop)
        self.probe = patch("norlan.local_transcription.probe", return_value={"duration": 8.0, "has_audio": True})
        self.probe.start(); self.addCleanup(self.probe.stop)
        self.transcriber = LocalTranscriber(self.config, cache_dir=self.root / "cache")

    def fixture(self, args, **kwargs):
        self.calls.append(args)
        if "-vm" in args:
            return subprocess.CompletedProcess(args, 0, stdout=self.windows)
        if "-of" in args:
            output = Path(args[args.index("-of") + 1]).with_suffix(".json")
            output.write_text(json.dumps({"transcription": [{"offsets": {"from": 250, "to": 1050}, "text": "实际说话"}]}))
        else:
            shutil.copyfile(self.audio, args[-1])
        return subprocess.CompletedProcess(args, 0)

    def test_real_speech_windows_anchor_nonzero_start_and_keep_long_pause(self):
        before = hashlib.sha256(self.audio.read_bytes()).hexdigest()
        with patch("norlan.local_transcription.subprocess.run", side_effect=self.fixture):
            result = self.transcriber.transcribe(self.audio)
            metadata = self.transcriber.last_metadata
            again = self.transcriber.transcribe(self.audio)
        self.assertEqual(result, [{"start": 2.4, "end": 3.2, "text": "实际说话"},
                                  {"start": 5.4, "end": 6.2, "text": "实际说话"}])
        self.assertEqual(result, again)
        self.assertEqual(hashlib.sha256(self.audio.read_bytes()).hexdigest(), before)
        self.assertEqual(len(self.calls), 4)
        self.assertTrue(self.transcriber.last_metadata["cache_hit"])
        self.assertEqual(self.transcriber.last_metadata["speech_window_count"], 2)
        self.assertIn("original-clock", metadata["timing_method"])
        self.assertEqual(metadata["vad_parameters"]["min_silence_ms"], 100)
        self.assertIn("vad_model_sha256", metadata)
        self.assertNotEqual(metadata["vad_executable_sha256"], metadata["executable_sha256"])

    def test_overlapping_detector_chunks_restore_original_clock_and_deduplicate(self):
        write_pcm(self.audio, b"\x00\x00" * 560000)
        self.probe.stop()
        with patch("norlan.local_transcription.probe", return_value={"duration": 35.0, "has_audio": True}):
            def chunks(args, **kwargs):
                if "-vm" in args:
                    self.calls.append(args)
                    name = Path(args[args.index("-f") + 1]).name
                    if name == "vad-window-0.wav":
                        text = "Detected 1 speech segments:\nSpeech segment 0: start = 1800.00, end = 2000.00\n"
                    elif name == "vad-window-10000.wav":
                        text = "Detected 1 speech segments:\nSpeech segment 0: start = 800.00, end = 1200.00\n"
                    else:
                        text = "Detected 2 speech segments:\nSpeech segment 0: start = 0.00, end = 200.00\nSpeech segment 1: start = 800.00, end = 900.00\n"
                    return subprocess.CompletedProcess(args, 0, stdout=text)
                return self.fixture(args, **kwargs)
            with patch("norlan.local_transcription.subprocess.run", side_effect=chunks):
                result = self.transcriber.transcribe(self.audio)
        self.assertEqual([r["start"] for r in result], [18.0, 28.0])
        record = json.loads(next((self.root / "cache").glob("*.json")).read_text())
        self.assertEqual(record["speech_windows"], [[18, 22], [28, 29]])
        self.assertEqual(sum("-vm" in call for call in self.calls), 3)
        self.assertEqual(sum("-of" in call for call in self.calls), 2)

    def test_recognition_context_cannot_cross_neighboring_speech_windows(self):
        windows = [(2.4, 3.2), (3.3, 4), (8.0, 9.0)]
        left, right = _recognition_window(windows, 0, 10)
        self.assertAlmostEqual(left, 2.15)
        self.assertAlmostEqual(right, 3.25)
        left, right = _recognition_window(windows, 1, 10)
        self.assertAlmostEqual(left, 3.25)
        self.assertAlmostEqual(right, 4.25)

    def test_window_union_keeps_silent_gaps_instead_of_spreading_subtitles(self):
        self.assertEqual(_merge_speech_windows([(18, 20), (18.1, 22), (20, 22), (28, 29)]),
                         [(18, 22), (28, 29)])

    def test_vad_silence_skips_whisper_entirely(self):
        self.windows = "Detected 0 speech segments:\n"
        with patch("norlan.local_transcription.subprocess.run", side_effect=self.fixture):
            self.assertEqual(self.transcriber.transcribe(self.audio), [])
        self.assertEqual(len(self.calls), 2)
        self.assertFalse(any("-of" in call for call in self.calls))

    def test_vad_model_and_binary_changes_invalidate_cache(self):
        with patch("norlan.local_transcription.subprocess.run", side_effect=self.fixture):
            self.transcriber.transcribe(self.audio)
            key = self.transcriber.last_metadata["cache_key"]
            self.vad_model.write_bytes(b"changed vad model")
            self.transcriber.transcribe(self.audio)
            self.assertNotEqual(self.transcriber.last_metadata["cache_key"], key)
            key = self.transcriber.last_metadata["cache_key"]
            self.vad.write_bytes(b"changed vad executable")
            self.transcriber.transcribe(self.audio)
        self.assertNotEqual(self.transcriber.last_metadata["cache_key"], key)
        self.assertEqual(len(self.calls), 12)

    def test_old_whole_episode_timing_cache_cannot_supply_early_subtitles(self):
        with patch("norlan.local_transcription.subprocess.run", side_effect=self.fixture):
            expected = self.transcriber.transcribe(self.audio)
            cached = next((self.root / "cache").glob("*.json"))
            record = json.loads(cached.read_text())
            record["identity"]["schema"] = "nolan-whisper-cpp-v1"
            record["segments"] = [{"start": 0, "end": 1, "text": "旧版提前字幕"}]
            cached.write_text(json.dumps(record))
            actual = self.transcriber.transcribe(self.audio)
        self.assertEqual(actual, expected)
        self.assertFalse(self.transcriber.last_metadata["cache_hit"])
        self.assertEqual(len(self.calls), 8)

    def test_cancellation_between_speech_windows_leaves_no_half_cache(self):
        transcriber = LocalTranscriber(self.config, cache_dir=self.root / "cache", cancel_check=lambda: self.stop)
        def stopped(args, **kwargs):
            result = self.fixture(args, **kwargs)
            if "-of" in args:
                self.stop = True
            return result
        with patch("norlan.local_transcription.subprocess.run", side_effect=stopped):
            with self.assertRaisesRegex(ProviderError, "暂停"):
                transcriber.transcribe(self.audio)
        self.assertEqual(len(self.calls), 3)
        self.assertFalse(list((self.root / "cache").glob("*.json")))

    def test_invalid_vad_response_never_falls_back_to_untimed_asr(self):
        self.windows = "Detected 2 speech segments:\nSpeech segment 0: start = 0.0, end = 100.0\n"
        with patch("norlan.local_transcription.subprocess.run", side_effect=self.fixture):
            with self.assertRaises(ProviderError):
                self.transcriber.transcribe(self.audio)
        self.assertEqual(len(self.calls), 2)
        self.assertFalse(list((self.root / "cache").glob("*.json")))

    def test_missing_vad_model_is_actionable_and_no_fallback(self):
        self.vad_model.unlink()
        with patch("norlan.local_transcription.subprocess.run") as run:
            with self.assertRaisesRegex(ProviderError, "install-local-asr"):
                self.transcriber.transcribe(self.audio)
        run.assert_not_called()

    def test_centisecond_units_complete_counts_and_monotonic_validation(self):
        self.assertEqual(_vad_windows(self.windows, 8), [(2.4, 3.2), (5.4, 6.2)])
        for text in ("", self.windows.replace("Detected 2", "Detected 3"), self.windows.replace("segment 1", "segment 2"),
                     self.windows.replace("540.00", "250.00"), self.windows.replace("620.00", "999.00")):
            with self.subTest(text=text):
                with self.assertRaises(ValueError): _vad_windows(text, 8)

    def test_pcm_crop_preserves_actual_samples_and_never_changes_input(self):
        samples = b"\x10\x00" * 16000 + b"\x20\x00" * 16000
        write_pcm(self.audio, samples)
        output = self.root / "window.wav"
        self.assertEqual(_crop_pcm(self.audio, output, 0.75, 1.25), .5)
        with wave.open(str(output), "rb") as cut:
            self.assertEqual(cut.readframes(cut.getnframes()), b"\x10\x00" * 4000 + b"\x20\x00" * 4000)
        with wave.open(str(self.audio), "rb") as original:
            self.assertEqual(original.readframes(original.getnframes()), samples)

    def test_subsecond_decoder_end_clamped_only_with_real_speech_boundary(self):
        raw = {"transcription": [{"offsets": {"from": 0, "to": 2000}, "text": "接住他"}]}
        self.assertEqual(_cpp_segments(raw, .68, speech_window=True)[0]["end"], .68)
        with self.assertRaises(ValueError): _cpp_segments(raw, .68)
        multi = {"transcription": [{"offsets": {"from": 0, "to": 1270}, "text": "First"},
                                    {"offsets": {"from": 1270, "to": 2000}, "text": "sentence"}]}
        self.assertEqual(_cpp_segments(multi, 1.82, speech_window=True)[-1]["end"], 1.82)
        raw["transcription"][0]["offsets"]["to"] = 10000
        with self.assertRaises(ValueError): _cpp_segments(raw, .68, speech_window=True)


@unittest.skipUnless(os.environ.get("NOLAN_TEST_REAL_LOCAL_ASR") == "1", "opt-in actual offline speech alignment test")
class ActualLocalSpeechTimingTests(unittest.TestCase):
    def test_synthetic_speech_with_intro_and_long_pause_uses_real_audio_clock(self):
        # Uses the installed local macOS voice and installed local models only.
        # No network, model download, original project, or GPT access is involved.
        if not Path("/usr/bin/say").is_file(): self.skipTest("macOS local TTS unavailable")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            spoken = root / "spoken.aiff"; pcm = root / "spoken.wav"
            subprocess.run(["/usr/bin/say", "-v", "Samantha", "-r", "150", "-o", str(spoken), "This is the first sentence."], check=True, capture_output=True)
            subprocess.run(["ffmpeg", "-v", "error", "-i", str(spoken), "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le", str(pcm)], check=True, capture_output=True)
            with wave.open(str(pcm), "rb") as audio: speech = audio.readframes(audio.getnframes())
            speech_duration = len(speech) / 32000
            tone = b"".join(int(1200 * math.sin(2 * math.pi * 440 * i / 16000)).to_bytes(2, "little", signed=True) for i in range(16000))
            first_start = 2.5
            second_start = first_start + speech_duration + 23.7
            audio = root / "synthetic.wav"
            write_pcm(audio, tone + b"\0\0" * 24000 + speech + b"\0\0" * 379200 + speech + b"\0\0" * 16000)
            before = hashlib.sha256(audio.read_bytes()).hexdigest()
            t = LocalTranscriber({"mode": "local", "language": "en"}, cache_dir=root / "cache")
            result = t.transcribe(audio)
            self.assertGreaterEqual(len(result), 2)
            self.assertTrue(any(first_start - .25 <= s["start"] <= first_start + .6 for s in result), result)
            self.assertTrue(any(second_start - .25 <= s["start"] <= second_start + .6 for s in result), result)
            self.assertFalse(any(s["start"] < first_start - .25 for s in result), result)
            self.assertFalse(any(first_start + speech_duration + .3 < s["start"] < second_start - .25 for s in result), result)
            self.assertTrue(any("sentence" in s["text"].lower() for s in result), result)
            self.assertEqual(hashlib.sha256(audio.read_bytes()).hexdigest(), before)
            self.assertEqual(t.transcribe(audio), result)
            self.assertTrue(t.last_metadata["cache_hit"])
