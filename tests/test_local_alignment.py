"""Synthetic contract tests. No cloud calls or model downloads."""
from copy import deepcopy
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from norlan import alignment_worker as worker, sensevoice_worker as sense
from norlan.local_alignment import LocalSubtitleVerifier, DEFAULT_MODEL, ZIPFORMER_MODEL
from norlan.providers import ProviderError
from norlan.subtitle_sync import caption_hash


class ByteAnchorTests(unittest.TestCase):
    def test_split_utf8_uses_first_byte_time_not_final_fragment(self):
        # Official byte-BPE spelling of the Chinese character 违.
        result = worker.decode_anchors(["▁Ə", "ť", "ł"], [.92, 1., 1.04], 2.)
        self.assertEqual(result, [{"char": "违", "time": .92}])

    def test_incomplete_terminal_bytes_never_become_characters(self):
        self.assertEqual(worker.decode_anchors(["▁Ə"], [.1], 1.), [])

    def test_bad_clock_or_unknown_symbol_rejected(self):
        for tokens, clocks in [(["▁ƍŁġ", "▁ƌıį"], [.8, .1]), (["🙂"], [.2]), (["▁ƍŁġ"], [float("nan")])]:
            with self.subTest(tokens=tokens), self.assertRaises(ValueError):
                worker.decode_anchors(tokens, clocks, 1.)

    def test_integer_reading_expands_units_and_zero(self):
        self.assertEqual(worker._integer_words("2990"), "二千九百九十")
        self.assertEqual(worker._integer_words("10001"), "一万零一")
        self.assertEqual(worker._integer_words("10"), "十")

    def test_missing_interior_word_can_be_supported_but_missing_edge_cannot(self):
        target = [{"key": c} for c in "abcde"]
        hypothesis = [{"key": c, "time": t} for c, t in zip("acde", [2., 2.3, 2.5, 2.8])]
        mapping = worker.match_units(target, hypothesis)
        result = worker._extent(target, hypothesis, mapping, 5.)
        self.assertEqual(result["coverage"], .8)
        self.assertTrue(result["consistent"])
        self.assertGreater(result["end"], 2.8)  # last emission is not a cue end
        result = worker._extent(target, hypothesis[:-1], worker.match_units(target, hypothesis[:-1]), 5.)
        self.assertFalse(result["consistent"])
        self.assertFalse(result["boundary_supported"])

    def test_short_or_collapsed_anchors_cannot_prove_timing(self):
        target = [{"key": c} for c in "ab"]
        hypothesis = [{"key": c, "time": 2.} for c in "ab"]
        self.assertFalse(worker._extent(target, hypothesis, {0: 0, 1: 1}, 4.)["consistent"])
        self.assertFalse(worker._extent(target, hypothesis, {0: 0}, 4.)["consistent"])

    def test_flush_padding_does_not_supply_audio_evidence(self):
        checker = object.__new__(worker.Checker)
        checker.duration, checker.rate, checker.samples, checker.cache = 1., 16000, [0.] * 16000, {}
        stream = SimpleNamespace(accept_waveform=lambda *_: None, input_finished=lambda: None)
        result = SimpleNamespace(tokens=["▁ƍŁġ", "▁ƌıį"], timestamps=[.2, 1.2], text="最后")
        checker.recognizer = SimpleNamespace(create_stream=lambda: stream, is_ready=lambda _: False,
                                             get_result_all=lambda _: result)
        with patch.dict("sys.modules", {"numpy": SimpleNamespace(zeros=lambda *_, **__: [])}), patch.object(
                worker, "units", lambda text, anchors: anchors):
            observed = checker.recognize()
        self.assertEqual(observed["text"], "最")
        self.assertEqual(observed["anchors"], [{"char": "最", "time": .2}])

    def test_zero_origin_uses_changed_right_context_not_same_cached_check_twice(self):
        checker = object.__new__(worker.Checker)
        checker.duration = 5.
        anchors = [{"char": c, "time": t} for c, t in zip("最后一车", [.1, .3, .5, .7])]

        def unit_stub(text, anchors=None):
            return [{"key": c, "char": c, **({"time": anchors[i]["time"]} if anchors else {})}
                    for i, c in enumerate(text)]

        def recognize(start=0., end=None):
            return {"origin": start, "end": 5. if end is None else end, "text": "最后一车",
                    "anchors": anchors, "units": unit_stub("最后一车", anchors)}

        checker.recognize = recognize
        with patch.object(worker, "units", unit_stub):
            result = checker.verify([{"start": .02, "end": 1., "text": "最后一车"}])
        self.assertTrue(result["cues"][0]["consistent"])
        first, second = result["cues"][0]["local_checks"]
        self.assertEqual(first["origin"], second["origin"])
        self.assertGreater(abs(first["end"] - second["end"]), .4)


class VerifierRepairTests(unittest.TestCase):
    @staticmethod
    def supported_addition():
        return {"start": 13.3, "end": 13.8, "text": "退了",
                "method": "two_models_two_contexts_original_clock", "models": ["sense_voice", "zipformer_ctc"],
                "checks": [{"backend": model, "origin": origin, "end": 16.,
                            "alignment": {"start": 13.3, "end": 13.8, "coverage": 1., "consistent": True,
                                          "boundary_supported": True, "matched_anchors": 2, "target_units": 2}}
                           for model in ("sense_voice", "zipformer_ctc") for origin in (10., 9.)]}

    def test_default_sensevoice_and_explicit_zipformer_compatibility(self):
        strong = LocalSubtitleVerifier({})
        self.assertEqual(strong.backend, "sense_voice")
        self.assertEqual(strong.model, DEFAULT_MODEL.resolve())
        self.assertEqual(strong.worker.name, "sensevoice_worker.py")
        light = LocalSubtitleVerifier({"alignment_backend": "zipformer_ctc"})
        self.assertEqual(light.model, ZIPFORMER_MODEL.resolve())
        self.assertEqual(str(light.worker.name), "alignment_worker.py")
        for config in ({"alignment_backend": "unknown"}, {"alignment_backend": []},
                       {"alignment_model_path": None}, {"alignment_coverage_model_path": False}):
            with self.assertRaises(ProviderError):
                LocalSubtitleVerifier(config)

    def test_missing_dependencies_fail_closed_with_install_instruction(self):
        with tempfile.TemporaryDirectory() as temp:
            checker = LocalSubtitleVerifier({"alignment_python": str(Path(temp) / "missing")})
            with self.assertRaisesRegex(ProviderError, "install-subtitle-alignment"):
                checker.check_ready()

    def test_repair_moves_audio_time_and_retains_text(self):
        checker = LocalSubtitleVerifier({})
        captions = [{"start": 0., "end": 2., "text": "最后一车"}]
        checker._last_caption_hash = caption_hash(captions)
        checker.last_observations = {"unsupported_cues": []}
        result = checker.repair(captions, {"aligned_cues": [{"cue_index": 0, "start": 6.56, "end": 7.4}]}, 1)
        self.assertEqual(result, [{"start": 6.56, "end": 7.4, "text": "最后一车"}])
        self.assertEqual(captions[0]["start"], 0.)
        self.assertEqual(checker.changes[0]["kind"], "realigned")

    def test_changed_captions_cannot_reuse_old_evidence(self):
        checker = LocalSubtitleVerifier({})
        captions = [{"start": 0., "end": 2., "text": "最后一车"}]
        checker._last_caption_hash = caption_hash(captions)
        changed = deepcopy(captions); changed[0]["text"] = "另一句话"
        with self.assertRaises(ProviderError):
            checker.repair(changed, {}, 1)

    def test_ctc_zero_support_alone_does_not_delete_possible_speech(self):
        with tempfile.TemporaryDirectory() as temp:
            checker = LocalSubtitleVerifier({}, cache_dir=Path(temp))
            checker._last_audio = Path(temp) / "audio.wav"
            captions = [{"start": 40., "end": 42., "text": "unconfirmed speech"}]
            checker._last_caption_hash = caption_hash(captions)
            checker.last_observations = {"unsupported_cues": [
                {"cue_index": 0, "independent_text_in_interval": "", "checks": [
                    {"matched_anchors": 0}, {"matched_anchors": 0}]}]}
            with patch("norlan.subtitle_noise.confirm_unsupported_caption", return_value={"unsupported": False}):
                self.assertEqual(checker.repair(captions, {}, 1), captions)

    def test_coverage_hallucination_requires_separate_ctc_and_two_whisper_contexts(self):
        with tempfile.TemporaryDirectory() as temp:
            checker = LocalSubtitleVerifier({}, cache_dir=Path(temp))
            checker._secondary_identity = {"available": True}
            issue = {"start": 13, "end": 14, "code": "uncovered_speech", "observed_text": "退了",
                     "secondary_ctc": {"zero_support": True, "language_supported": True}}
            with patch("norlan.subtitle_noise.confirm_unsupported_caption", return_value={
                    "unsupported": True, "evidence_hash": "a" * 64}) as double_check:
                result = checker._crosscheck_uncovered(Path(temp) / "synthetic.wav", {"contradictions": [issue]})
                self.assertEqual(result["contradictions"], [])
                self.assertEqual(len(result["coverage_exclusions"]), 1)
                self.assertEqual(len(checker.changes), 1)
                double_check.assert_called_once()
            for secondary in ({"zero_support": False, "language_supported": True},
                              {"zero_support": True, "language_supported": False}, {}):
                candidate = {**issue, "secondary_ctc": secondary}
                with patch("norlan.subtitle_noise.confirm_unsupported_caption") as double_check:
                    result = checker._crosscheck_uncovered(Path(temp) / "synthetic.wav", {"contradictions": [candidate]})
                    self.assertEqual(result["contradictions"], [candidate])
                    double_check.assert_not_called()
            with patch("norlan.subtitle_noise.confirm_unsupported_caption", return_value={"unsupported": False}):
                result = checker._crosscheck_uncovered(Path(temp) / "synthetic.wav", {"contradictions": [issue]})
                self.assertEqual(result["contradictions"], [issue])

    def test_only_dual_model_dual_context_actual_word_and_clock_can_be_added(self):
        addition = self.supported_addition()
        self.assertTrue(LocalSubtitleVerifier._addition_supported(addition))
        for mutation in (lambda a: a["checks"].pop(),
                         lambda a: a["checks"][0].update(backend="zipformer_ctc"),
                         lambda a: a["checks"][1].update(origin=10.),
                         lambda a: a["checks"][0]["alignment"].update(start=14.),
                         lambda a: a["checks"][0]["alignment"].update(start=float("nan")),
                         lambda a: a["checks"][0]["alignment"].update(coverage=.5),
                         lambda a: a["checks"][0]["alignment"].update(boundary_supported=False)):
            invalid = deepcopy(addition)
            mutation(invalid)
            self.assertFalse(LocalSubtitleVerifier._addition_supported(invalid))

    def test_supported_actual_words_insert_in_gap_and_record_evidence_without_overwriting_script(self):
        with tempfile.TemporaryDirectory() as temp:
            checker = LocalSubtitleVerifier({}, cache_dir=Path(temp))
            checker._last_audio = Path(temp) / "synthetic.wav"
            captions = [{"start": 11., "end": 13., "text": "厂章给我，我再去谈。"},
                        {"start": 16., "end": 18., "text": "你一夜没睡"}]
            checker._last_caption_hash = caption_hash(captions)
            checker.last_observations = {"supported_additions": [self.supported_addition()]}
            with patch("norlan.local_alignment.probe", return_value={"duration": 20.}):
                result = checker.repair(captions, {}, 1)
            self.assertEqual(result[1], {"start": 13.3, "end": 13.8, "text": "退了"})
            self.assertEqual(result[0], captions[0])
            self.assertEqual(checker.changes[-1]["kind"], "added_from_audio")
            self.assertEqual(checker.changes[-1]["evidence"]["models"], ["sense_voice", "zipformer_ctc"])
            checker.changes.clear()
            overlapping = [{"start": 11., "end": 14., "text": captions[0]["text"]}]
            checker._last_caption_hash = caption_hash(overlapping)
            with patch("norlan.local_alignment.probe", return_value={"duration": 20.}):
                self.assertEqual(checker.repair(overlapping, {}, 1), overlapping)
            self.assertFalse(checker.changes)


class SenseVoiceAnchorTests(unittest.TestCase):
    @staticmethod
    def units(text, anchors=None):
        return [{"key": c, "char": c, **({"time": anchors[i]["time"]} if anchors is not None else {})}
                for i, c in enumerate(text)]

    def test_unicode_tokens_are_seconds_and_metadata_is_not_subtracted_again(self):
        self.assertEqual(sense.decode_anchors(["楼", "上", "的"], [.54, .66, .84], 2.),
                         [{"char": "楼", "time": .54}, {"char": "上", "time": .66}, {"char": "的", "time": .84}])
        self.assertEqual(sense.decode_anchors(["<|zh|>", "▁go"], [0., .2], 1.),
                         [{"char": "g", "time": .2}, {"char": "o", "time": .2}])

    def test_broken_or_collapsed_outside_clock_is_not_evidence(self):
        for tokens, times in ((["字"], []), (["甲", "乙"], [.8, .1]),
                              (["字"], [float("nan")]), (["字"], [True]), (["字"], [2.])):
            with self.assertRaises(ValueError):
                sense.decode_anchors(tokens, times, 1.)

    def test_local_crop_restores_original_clock_without_artificial_flush_audio(self):
        checker = object.__new__(sense.SenseVoiceChecker)
        checker.duration, checker.rate, checker.samples, checker.cache = 20., 10, list(range(200)), {}
        accepted = []
        stream = SimpleNamespace(accept_waveform=lambda rate, samples: accepted.append((rate, samples)),
                                 result=SimpleNamespace(tokens=["楼", "上", "的"], timestamps=[.2, .4, .6]))
        checker.recognizer = SimpleNamespace(create_stream=lambda: stream, decode_stream=lambda _: None)
        with patch.object(sense, "units", self.units):
            result = checker.recognize(10., 12.)
            self.assertIs(checker.recognize(10., 12.), result)
        self.assertEqual(accepted, [(10, list(range(100, 120)))])
        self.assertEqual([a["time"] for a in result["anchors"]], [10.2, 10.4, 10.6])

    def test_long_scan_uses_fixed_overlaps_and_has_no_gap_or_duplicate_anchor(self):
        checker = object.__new__(sense.SenseVoiceChecker)
        checker.duration, checker.rate, checker.samples, checker.cache = 45., 10, list(range(450)), {}
        all_times = [.5, 17.4, 17.6, 32.4, 32.6, 44.5]
        created = []
        def create():
            stream = SimpleNamespace()
            def accept(rate, samples):
                stream.first, stream.last = samples[0] / rate, (samples[-1] + 1) / rate
            stream.accept_waveform = accept
            created.append(stream)
            return stream
        def decode(stream):
            selected = [(str(i), t - stream.first) for i, t in enumerate(all_times)
                        if stream.first <= t < stream.last]
            stream.result = SimpleNamespace(tokens=[r[0] for r in selected], timestamps=[r[1] for r in selected])
        checker.recognizer = SimpleNamespace(create_stream=create, decode_stream=decode)
        with patch.object(sense, "units", self.units):
            result = checker.recognize()
            self.assertIs(checker.recognize(), result)
        self.assertEqual([a["time"] for a in result["anchors"]], all_times)
        self.assertEqual(result["text"], "012345")
        self.assertEqual(len(created), 3)
        self.assertEqual([(w["owned_start"], w["owned_end"]) for w in result["scan_windows"]],
                         [(0., 17.5), (17.5, 32.5), (32.5, 45.)])

    def test_two_supported_local_contexts_can_recover_a_global_edge_miss(self):
        checker = object.__new__(sense.SenseVoiceChecker)
        checker.duration, checker.coverage_model_dir = 5., None
        def recognize(start=0., end=None):
            chars = "abce" if end is None else "abcd"
            anchors = [{"char": c, "time": t} for c, t in zip(chars, [1., 1.2, 1.4, 1.6])]
            return {"origin": start, "end": 5. if end is None else end, "text": chars,
                    "anchors": anchors, "units": self.units(chars, anchors)}
        checker.recognize = recognize
        with patch.object(worker, "units", self.units):
            result = checker.verify([{"start": 1., "end": 2., "text": "abcd"}])
        row = result["cues"][0]
        self.assertTrue(row["consistent"])
        self.assertEqual(row["coverage"], 1.)
        self.assertTrue(row["boundary_supported"])
        self.assertEqual(row["boundary_evidence"], "two_changed_acoustic_contexts")

    def test_low_coverage_tentative_cue_never_hides_uncovered_actual_speech(self):
        checker = object.__new__(worker.Checker)
        checker.duration = 5.
        def recognize(start=0., end=None):
            anchors = [{"char": c, "time": t} for c, t in zip("abc", [1., 1.3, 1.6])]
            return {"origin": start, "end": 5. if end is None else end, "text": "abc",
                    "anchors": anchors, "units": self.units("abc", anchors)}
        checker.recognize = recognize
        with patch.object(worker, "units", self.units):
            result = checker.verify([{"start": 0., "end": 3., "text": "xyz"}])
        self.assertFalse(result["cues"][0]["consistent"])
        self.assertTrue(result["contradictions"])
        self.assertEqual(result["contradictions"][0]["observed_text"], "abc")

    def test_missing_audio_addition_requires_both_word_and_original_clock_agreement(self):
        checker = object.__new__(sense.SenseVoiceChecker)
        checker.duration = 20.
        def hypothesis(start, end, shift=0.):
            anchors = [{"char": c, "time": t + shift} for c, t in zip("退了", [13.32, 13.5])]
            return {"origin": start, "end": end, "text": "退了", "units": self.units("退了", anchors)}
        checker.recognize = lambda start, end: hypothesis(start, end)
        def secondary(start, end):
            observed = hypothesis(start, end, .08)
            # Synthetic reproduction of the real decoder's context
            # sensitivity: a later recrop loses the first observed word.
            if start > 10.5:
                observed["text"], observed["units"] = "了", observed["units"][1:]
            return observed
        checker.coverage_checker = SimpleNamespace(recognize=secondary)
        issue = {"start": 12.8, "end": 13.7, "observed_text": "他退了",
                 "observed_anchors": [{"char": c, "time": t} for c, t in zip("他退了", [12.9, 13.32, 13.5])]}
        secondary = self.units("谈退了", [{"char": c, "time": t} for c, t in zip("谈退了", [13., 13.4, 13.58])])
        with patch.object(sense, "units", self.units):
            additions = checker._supported_additions(issue, secondary, [])
            self.assertEqual(len(additions), 1)
            self.assertEqual(additions[0]["text"], "退了", "unsupported prefix is not copied")
            self.assertEqual(len(additions[0]["checks"]), 4)
            self.assertEqual(checker._supported_additions(issue, secondary, [
                {"consistent": True, "start": 13., "end": 14.}]), [])
            wrong_clock = [{**u, "time": u["time"] + 1.} for u in secondary]
            self.assertEqual(checker._supported_additions(issue, wrong_clock, []), [])
            checker.coverage_checker = SimpleNamespace(recognize=lambda start, end: hypothesis(start, end, .8))
            self.assertEqual(checker._supported_additions(issue, secondary, []), [])


if __name__ == "__main__":
    unittest.main()
