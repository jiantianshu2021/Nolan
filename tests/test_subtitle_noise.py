"""Synthetic PCM and mocked local decoders only; never contact a provider."""
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
import wave
from unittest.mock import patch

from norlan import subtitle_noise as noise


class SubtitleNoiseTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="nolan-synthetic-noise-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.audio = self.root / "synthetic actual audio.wav"
        self.write_audio(self.audio, 16000)
        self.model = self.root / "synthetic model.bin"
        self.model.write_bytes(b"synthetic small model")
        self.executable = self.root / "synthetic whisper-cli"
        self.executable.write_bytes(b"synthetic executable")
        self.python = self.root / "synthetic checker python"
        self.python.write_bytes(b"synthetic isolated runtime")
        self.config = {"model_path": str(self.model), "alignment_python": str(self.python)}
        self.cue = {"start": 10, "end": 12, "text": "MING PAO CANADA"}
        self.texts = ["系统绑定成功", "赠送加工台图纸"]
        self.phonetics = [["other"], ["neighbor"], ["different"]]
        self.asr_count, self.commands, self.crops, self.cancelled = 0, [], [], False
        self.which = patch("norlan.subtitle_noise.shutil.which", return_value=str(self.executable))
        self.which.start()
        self.addCleanup(self.which.stop)
        self.probe = patch("norlan.subtitle_noise.probe", return_value={"duration": 20.0, "has_audio": True})
        self.probe.start()
        self.addCleanup(self.probe.stop)

    def write_audio(self, path, rate):
        with wave.open(str(path), "wb") as output:
            output.setparams((1, 2, rate, 0, "NONE", "not compressed"))
            output.writeframes(b"\0\0" * (rate * 20))

    def fixture(self, args, **kwargs):
        self.commands.append(args)
        if "-of" in args:
            index = self.asr_count % 2
            self.asr_count += 1
            with wave.open(args[args.index("-f") + 1], "rb") as crop:
                self.crops.append((crop.getframerate(), crop.getnframes()))
            Path(args[args.index("-of") + 1]).with_suffix(".json").write_text(
                json.dumps({"transcription": [{"text": self.texts[index]}]}, ensure_ascii=False))
        elif "-c" in args:
            return subprocess.CompletedProcess(args, 0, stdout=json.dumps({"version": "synthetic", "units": self.phonetics}))
        else:
            self.write_audio(Path(args[-1]), 16000)
        return subprocess.CompletedProcess(args, 0)

    def call(self, cue=None, config=None, **kwargs):
        return noise.confirm_unsupported_caption(self.audio, self.cue if cue is None else cue,
                                                  self.config if config is None else config,
                                                  cache_dir=self.root / "cache", **kwargs)

    def test_two_different_original_clock_contexts_without_prompt_or_vad(self):
        before = self.audio.read_bytes()
        with patch("norlan.subtitle_noise.subprocess.run", side_effect=self.fixture) as run:
            result = self.call(config={**self.config, "prompt": "never forward planned dialogue", "api_key": "SECRET"})
        self.assertTrue(result["unsupported"])
        self.assertEqual(result["transcripts"], self.texts)
        self.assertEqual(result["method"], "two_origin_whisper_context")
        self.assertEqual(len(result["evidence_hash"]), 64)
        self.assertEqual(self.crops, [(16000, 112000), (16000, 152000)])
        self.assertEqual(self.audio.read_bytes(), before)
        self.assertEqual(run.call_count, 2)
        for call in run.call_args_list:
            args = call.args[0]
            self.assertEqual(args[0], str(self.executable.resolve()))
            self.assertEqual(args[args.index("-ml") + 1], "0")
            self.assertIn("-oj", args)
            self.assertNotIn("--prompt", args)
            self.assertNotIn("--vad", args)
            self.assertNotIn("SECRET", str(args))
            self.assertFalse(call.kwargs.get("shell"))

    def test_any_supporting_context_prevents_rejection(self):
        for texts in (["MING PAO CANADA", "noise"], ["noise", "Ming—Pao Canada"],
                      ["MIN PAO CANADA", "noise"], ["CANADA", "different"]):
            self.texts = list(texts)
            cache = self.root / "cache"
            for path in cache.glob("*.json"):
                path.unlink()
            with patch("norlan.subtitle_noise.subprocess.run", side_effect=self.fixture):
                self.assertFalse(self.call()["unsupported"], texts)

    def test_partial_chinese_or_unicode_support_is_conservatively_retained(self):
        for target, observed in (("厂章给我，我再去谈", "厂长给我"), ("ＡＢＣ公司", "ABC公司"),
                                 ("快点", "快"), ("楼上的先下来", "楼上下来")):
            self.assertTrue(noise._supports_caption(target, observed))

    def test_phonetic_support_prevents_rejecting_a_homophone(self):
        cue = {**self.cue, "text": "降维民生"}
        self.texts = ["将为明升", "近邻句子"]
        self.phonetics = [["jiang", "wei", "min", "sheng"], ["jiang", "wei", "ming", "sheng"], ["jin", "lin"]]
        with patch("norlan.subtitle_noise.subprocess.run", side_effect=self.fixture):
            result = self.call(cue)
        self.assertFalse(result["unsupported"])
        self.assertEqual(result["diagnostic"], "caption_phonetically_supported")
        self.assertEqual(len(self.commands), 3)

    def test_chinese_both_contexts_and_phonetic_non_support_can_supply_extra_evidence(self):
        cue = {**self.cue, "text": "明报出品"}
        self.texts = ["降维科技系统", "赠送加工台图纸"]
        with patch("norlan.subtitle_noise.subprocess.run", side_effect=self.fixture):
            self.assertTrue(self.call(cue)["unsupported"])

    def test_unavailable_phonetics_preserves_unsupported_chinese_caption(self):
        cue = {**self.cue, "text": "明报出品"}
        self.python.unlink()
        with patch("norlan.subtitle_noise.subprocess.run", side_effect=self.fixture):
            result = self.call(cue)
        self.assertFalse(result["unsupported"])
        self.assertEqual(result["diagnostic"], "phonetic_check_unavailable")

    def test_cache_binds_actual_audio_model_executable_crops_and_parameters(self):
        with patch("norlan.subtitle_noise.subprocess.run", side_effect=self.fixture):
            first = self.call()
            cached = self.call()
            self.assertEqual(self.asr_count, 2)
            self.assertTrue(cached["cache_hit"])
            self.assertEqual(first["evidence_hash"], cached["evidence_hash"])
            self.model.write_bytes(b"changed synthetic model")
            self.call()
            self.assertEqual(self.asr_count, 4)
            self.executable.write_bytes(b"changed synthetic executable")
            self.call()
            self.assertEqual(self.asr_count, 6)
            self.call({**self.cue, "start": 10.5})
            self.assertEqual(self.asr_count, 8)
            self.call(config={**self.config, "language": "en"})
            self.assertEqual(self.asr_count, 10)
            with self.audio.open("ab") as stream:
                stream.write(b"trailing synthetic bytes")
            self.call()
            self.assertEqual(self.asr_count, 12)

    def test_corrupt_cache_recomputes_instead_of_trusting_a_rejection_flag(self):
        with patch("norlan.subtitle_noise.subprocess.run", side_effect=self.fixture):
            self.call()
            cache = next((self.root / "cache").glob("*.json"))
            cache.write_text('{"unsupported": true}')
            result = self.call()
        self.assertTrue(result["unsupported"])
        self.assertFalse(result["cache_hit"])
        self.assertEqual(self.asr_count, 4)
        with patch("norlan.subtitle_noise.subprocess.run", side_effect=self.fixture):
            cache.write_text("[]")
            self.assertTrue(self.call()["unsupported"])
        self.assertEqual(self.asr_count, 6)

    def test_real_pcm_is_normalized_to_16k_before_sample_crops(self):
        self.write_audio(self.audio, 24000)
        with patch("norlan.subtitle_noise.subprocess.run", side_effect=self.fixture):
            self.assertTrue(self.call()["unsupported"])
        self.assertEqual(self.commands[0][0], "ffmpeg")
        self.assertEqual(self.crops, [(16000, 112000), (16000, 152000)])

    def test_same_clipped_origin_does_not_claim_two_independent_contexts(self):
        with patch("norlan.subtitle_noise.subprocess.run", side_effect=self.fixture) as run:
            result = self.call({**self.cue, "start": 0, "end": 2})
        self.assertFalse(result["unsupported"])
        self.assertEqual(result["diagnostic"], "distinct_origins_unavailable")
        run.assert_not_called()

    def test_timeout_or_cli_failure_never_rejects_or_leaks_private_error(self):
        failures = [subprocess.TimeoutExpired("synthetic", 600, stderr="SECRET"), OSError("SECRET")]
        for failure in failures:
            with patch("norlan.subtitle_noise.subprocess.run", side_effect=failure):
                result = self.call()
            self.assertFalse(result["unsupported"])
            self.assertNotIn("SECRET", str(result))
            self.assertFalse(list((self.root / "cache").glob("*.json")))
        with patch("norlan.subtitle_noise.subprocess.run", return_value=subprocess.CompletedProcess([], 1)):
            self.assertFalse(self.call()["unsupported"])

    def test_invalid_json_or_incomplete_response_never_rejects(self):
        def invalid(args, **kwargs):
            output = Path(args[args.index("-of") + 1]).with_suffix(".json")
            output.write_text('{"text":"untimed","transcription":[]}')
            return subprocess.CompletedProcess(args, 0)
        with patch("norlan.subtitle_noise.subprocess.run", side_effect=invalid):
            result = self.call()
        self.assertFalse(result["unsupported"])
        self.assertFalse(list((self.root / "cache").glob("*.json")))

    def test_cancel_before_or_during_recognition_never_rejects(self):
        with patch("norlan.subtitle_noise.subprocess.run", side_effect=self.fixture) as run:
            self.assertFalse(self.call(cancel_check=lambda: True)["unsupported"])
            run.assert_not_called()
        def cancel_after_first(args, **kwargs):
            completed = self.fixture(args, **kwargs)
            self.cancelled = True
            return completed
        with patch("norlan.subtitle_noise.subprocess.run", side_effect=cancel_after_first):
            result = self.call(cancel_check=lambda: self.cancelled)
        self.assertFalse(result["unsupported"])
        self.assertEqual(result["diagnostic"], "cancelled")
        self.assertFalse(list((self.root / "cache").glob("*.json")))

    def test_changed_source_or_model_during_run_is_not_cached_as_rejection(self):
        def mutate(args, **kwargs):
            completed = self.fixture(args, **kwargs)
            self.model.write_bytes(b"changed during recognition")
            return completed
        with patch("norlan.subtitle_noise.subprocess.run", side_effect=mutate):
            result = self.call()
        self.assertFalse(result["unsupported"])
        self.assertFalse(list((self.root / "cache").glob("*.json")))

    def test_invalid_inputs_are_safe_non_rejections(self):
        with patch("norlan.subtitle_noise.subprocess.run", side_effect=AssertionError("must not recognize")):
            for cue in ({}, {**self.cue, "start": False}, {**self.cue, "end": 30}, {**self.cue, "text": ""}):
                self.assertFalse(self.call(cue)["unsupported"])
            for config in ({"mode": "openai"}, {**self.config, "threads": True}, {**self.config, "timeout": float("nan")}):
                self.assertFalse(self.call(config=config)["unsupported"])


if __name__ == "__main__":
    unittest.main()
