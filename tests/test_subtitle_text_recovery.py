"""Synthetic independent acoustic evidence; no recognizer, media or API calls."""
from copy import deepcopy
import unittest
from unittest.mock import Mock

from norlan.subtitle_sync import caption_hash, report_matches
from norlan.subtitle_text_recovery import recover_audio_text


class SubtitleAudioTextRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.captions = [{"start": 1., "end": 2., "text": "账我看了。"},
                         {"start": 4., "end": 5., "text": "那些电话"},
                         {"start": 8., "end": 9., "text": "250万"}]
        self.raw = [{"start": .1, "end": .4, "text": "原识别中已删除的噪声"},
                    {"start": 1.1, "end": 2.1, "text": "让我看了"},
                    {"start": 4.1, "end": 5.1, "text": "那些电话"},
                    {"start": 8., "end": 9., "text": "250万"}]
        self.binding = {"duration": 20., "audio_sha256": "a" * 64,
                        "source_key": "synthetic-original-audio-v1",
                        "checker_identity": {"engine": "synthetic-independent-ctc", "model": "b" * 64}}

    def observations(self, cues):
        rows = []
        for index, cue in enumerate(cues):
            unsupported = cue["text"] in {"账我看了。", "不支持的替代文字"}
            rows.append({"cue_index": index, "text": cue["text"],
                         "start": cue["start"], "end": cue["end"],
                         "coverage": .75 if unsupported else 1.,
                         "consistent": not unsupported, "boundary_supported": not unsupported})
        return {"independent": True, "method": "independent_ctc", "full_audio_checked": True,
                "audio_sha256": self.binding["audio_sha256"],
                "checker_identity": deepcopy(self.binding["checker_identity"]),
                "cues": rows, "contradictions": []}

    def run_recovery(self, verify=None, captions=None, raw=None):
        return recover_audio_text(self.captions if captions is None else captions,
            self.raw if raw is None else raw,
            verify=verify or self.observations, **self.binding)

    def test_independent_support_recovers_original_words_and_keeps_timing_and_count(self):
        caps, raw = deepcopy(self.captions), deepcopy(self.raw)
        verify = Mock(side_effect=self.observations)
        output, audit = self.run_recovery(verify)
        self.assertEqual(output[0], {"start": 1., "end": 2., "text": "让我看了"})
        self.assertEqual(output[1:], caps[1:])
        self.assertEqual((self.captions, self.raw), (caps, raw))
        self.assertEqual(len(output), len(caps))
        self.assertEqual(audit["status"], "recovered")
        self.assertEqual(audit["verification_calls"], 3)
        self.assertEqual(verify.call_count, 3)
        self.assertEqual(audit["changes"][0]["raw_cue_index"], 1)
        self.assertEqual(audit["input_caption_hash"], caption_hash(caps))
        self.assertEqual(audit["output_caption_hash"], caption_hash(output))
        self.assertEqual(audit["raw_caption_hash"], caption_hash(raw))
        self.assertEqual(audit["source_audio_sha256"], self.binding["audio_sha256"])
        self.assertEqual(audit["checker_identity"], self.binding["checker_identity"])
        self.assertTrue(report_matches(audit["final_report"], output,
            audio_sha256=self.binding["audio_sha256"], source_key=self.binding["source_key"]))

    def test_passed_script_text_is_retained_without_extra_checks(self):
        def support_all(cues):
            result = self.observations(cues)
            result["cues"][0].update(coverage=1., consistent=True, boundary_supported=True)
            return result
        verify = Mock(side_effect=support_all)
        output, audit = self.run_recovery(verify)
        self.assertEqual(output, self.captions)
        self.assertEqual((audit["status"], verify.call_count), ("unchanged", 1))

    def test_timing_only_errors_never_authorize_text_replacement(self):
        def timing_only(cues):
            result = self.observations(cues)
            result["cues"][0].update(start=.3, end=1.5, coverage=1., consistent=True, boundary_supported=True)
            return result
        verify = Mock(side_effect=timing_only)
        output, audit = self.run_recovery(verify)
        self.assertEqual(output, self.captions)
        self.assertEqual((audit["proposals"], verify.call_count), ([], 1))

    def test_unsupported_raw_words_cannot_replace_current_words(self):
        raw = deepcopy(self.raw)
        raw[1]["text"] = "不支持的替代文字"
        verify = Mock(side_effect=self.observations)
        output, audit = self.run_recovery(verify, raw=raw)
        self.assertEqual(output, self.captions)
        self.assertEqual((audit["status"], verify.call_count), ("unchanged", 2))
        self.assertTrue(audit["rejections"])

    def test_full_coverage_and_explicit_boundary_are_required_even_when_normal_qa_tolerates_less(self):
        for field, value in [("coverage", .8), ("boundary_supported", None)]:
            with self.subTest(field=field):
                def incomplete(cues):
                    result = self.observations(cues)
                    if cues[0]["text"] == "让我看了":
                        result["cues"][0][field] = value
                    return result
                output, audit = self.run_recovery(incomplete)
                self.assertEqual(output, self.captions)
                self.assertEqual(audit["verification_calls"], 2)

    def test_mismatched_audio_identity_blocks_recovery(self):
        def wrong_audio(cues):
            result = self.observations(cues)
            result["audio_sha256"] = "c" * 64
            return result
        verify = Mock(side_effect=wrong_audio)
        output, audit = self.run_recovery(verify)
        self.assertEqual(output, self.captions)
        self.assertEqual((audit["status"], verify.call_count), ("verification_failed", 1))

    def test_original_vad_or_incomplete_whole_audio_checks_do_not_support_recovery(self):
        for field, value in [("method", "vad"), ("independent", False), ("full_audio_checked", False)]:
            with self.subTest(field=field):
                def not_independent(cues):
                    result = self.observations(cues)
                    result[field] = value
                    return result
                output, audit = self.run_recovery(not_independent)
                self.assertEqual(output, self.captions)
                self.assertEqual(audit["verification_calls"], 1)

    def test_overlapping_original_cues_are_rejected_as_invalid_not_guessed(self):
        raw = deepcopy(self.raw)
        raw.insert(2, {"start": 1.2, "end": 2.2, "text": "另一个原识别"})
        output, audit = self.run_recovery(raw=raw)
        self.assertEqual(output, self.captions)
        self.assertEqual(audit["verification_calls"], 1)
        self.assertTrue(audit["rejections"])

    def test_two_nonoverlapping_original_cues_cannot_be_joined_by_recovery(self):
        raw = deepcopy(self.raw)
        raw[1:2] = [{"start": 1., "end": 1.5, "text": "让我"},
                    {"start": 1.5, "end": 2., "text": "看了"}]
        output, audit = self.run_recovery(raw=raw)
        self.assertEqual(output, self.captions)
        self.assertEqual(audit["proposals"], [])

    def test_one_original_cue_cannot_be_copied_into_two_split_current_cues(self):
        captions = deepcopy(self.captions)
        captions[:1] = [{"start": 1., "end": 1.5, "text": "账我看了。"},
                       {"start": 1.5, "end": 2., "text": "账我看了。"}]
        raw = deepcopy(self.raw)
        raw[1].update(start=1., end=2.)
        output, audit = self.run_recovery(captions=captions, raw=raw)
        self.assertEqual(output, captions)
        self.assertEqual(audit["proposals"], [])

    def test_nearest_original_without_strong_time_overlap_is_not_used(self):
        raw = deepcopy(self.raw)
        raw[1].update(start=2.1, end=2.8)
        output, audit = self.run_recovery(raw=raw)
        self.assertEqual(output, self.captions)
        self.assertEqual(audit["verification_calls"], 1)

    def test_differing_digits_or_chinese_number_words_cannot_be_recovered(self):
        for current, original in [("250万", "200万"), ("一百万", "二百万"), ("250万", "二百五十万")]:
            with self.subTest(current=current, original=original):
                captions = [{"start": 1., "end": 2., "text": current}]
                raw = [{"start": 1., "end": 2., "text": original}]
                def unsupported(cues):
                    observed = self.observations(cues)
                    observed["cues"][0].update(coverage=.75, consistent=False, boundary_supported=False)
                    return observed
                output, audit = self.run_recovery(unsupported, captions=captions, raw=raw)
                self.assertEqual(output, captions)
                self.assertEqual(audit["proposals"], [])

    def test_added_or_removed_current_cues_do_not_shift_original_text_mapping(self):
        # There is a new current cue after a dropped noise row; indices differ
        # in both directions while unique original audio clocks remain stable.
        captions = [deepcopy(self.captions[0]), {"start": 2.5, "end": 3., "text": "新增对白"}]
        captions.extend(deepcopy(self.captions[1:]))
        output, audit = self.run_recovery(captions=captions)
        self.assertEqual(output[0]["text"], "让我看了")
        self.assertEqual(output[1:], captions[1:])
        self.assertEqual(audit["changes"][0]["raw_cue_index"], 1)

    def test_final_verification_regression_rolls_back_every_change(self):
        calls = 0
        def regression(cues):
            nonlocal calls
            calls += 1
            result = self.observations(cues)
            if calls == 3:
                result["cues"][1].update(coverage=.1, consistent=False)
            return result
        output, audit = self.run_recovery(regression)
        self.assertEqual(output, self.captions)
        self.assertEqual((audit["changes"], audit["verification_calls"]), ([], 3))

    def test_final_new_whole_audio_contradiction_rolls_back_change(self):
        calls = 0
        def contradiction(cues):
            nonlocal calls
            calls += 1
            result = self.observations(cues)
            if calls == 3:
                result["contradictions"] = [{"start": 10., "end": 11., "reason": "新增未覆盖人声"}]
            return result
        output, audit = self.run_recovery(contradiction)
        self.assertEqual(output, self.captions)
        self.assertEqual(audit["changes"], [])

    def test_final_coverage_regression_to_point_eight_still_rolls_back(self):
        calls = 0
        def weaker_final(cues):
            nonlocal calls
            calls += 1
            result = self.observations(cues)
            if calls == 3:
                result["cues"][0]["coverage"] = .8
            return result
        output, audit = self.run_recovery(weaker_final)
        self.assertEqual(output, self.captions)
        self.assertEqual(audit["changes"], [])

    def test_supported_subset_is_rechecked_without_hiding_remaining_old_failure(self):
        captions = deepcopy(self.captions)
        captions[1]["text"] = "不支持的替代文字"
        raw = deepcopy(self.raw)
        raw[2]["text"] = "另一个仍不支持的识别"
        def partial(cues):
            result = self.observations(cues)
            result["cues"][1].update(coverage=.2, consistent=False, boundary_supported=False)
            return result
        output, audit = self.run_recovery(partial, captions=captions, raw=raw)
        self.assertEqual(output[0]["text"], "让我看了")
        self.assertEqual(output[1]["text"], captions[1]["text"])
        self.assertEqual(audit["status"], "recovered")
        self.assertEqual(audit["final_report"]["status"], "needs_attention")
        self.assertEqual([change["cue_index"] for change in audit["changes"]], [0])

    def test_checker_exception_at_each_stage_does_not_adopt_or_expose_error(self):
        for failure_at in (1, 2, 3):
            with self.subTest(failure_at=failure_at):
                calls = 0
                def failure(cues):
                    nonlocal calls
                    calls += 1
                    if calls == failure_at:
                        raise RuntimeError("SECRET-DIAGNOSTIC")
                    return self.observations(cues)
                output, audit = self.run_recovery(failure)
                self.assertEqual(output, self.captions)
                self.assertEqual(audit["status"], "verification_failed")
                self.assertNotIn("SECRET-DIAGNOSTIC", str(audit))
                self.assertEqual(calls, failure_at)

    def test_raw_same_text_or_empty_text_evidence_never_invents_a_change(self):
        for raw in ([], [{"start": 1., "end": 2., "text": "账我看了。"}],
                    [{"start": 1., "end": 2., "text": ""}]):
            with self.subTest(raw=raw):
                output, audit = self.run_recovery(raw=raw)
                self.assertEqual(output, self.captions)
                self.assertEqual(audit["verification_calls"], 1)


if __name__ == "__main__":
    unittest.main()
