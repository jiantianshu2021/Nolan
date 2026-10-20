"""Synthetic independent evidence only; no audio/model/API calls."""
from copy import deepcopy
import unittest
from unittest.mock import Mock

from norlan import subtitle_sync as sync


class SubtitleSyncTests(unittest.TestCase):
    def setUp(self):
        self.captions = [{"start": 6.6, "end": 8.9, "text": "最后一车，全退了。"},
                         {"start": 9.3, "end": 10.7, "text": "还要一百万违约金。"}]
        self.binding = {"duration": 20.0, "audio_sha256": "a" * 64, "source_key": "dry-version-1",
                        "checker_identity": {"engine": "synthetic-independent-forced-alignment", "model_hash": "b" * 64}}

    def observations(self, timings=None):
        timings = timings or self.captions
        return {"independent": True, "method": "forced_alignment", "full_audio_checked": True,
                "audio_sha256": self.binding["audio_sha256"],
                "checker_identity": deepcopy(self.binding["checker_identity"]),
                "cues": [{"cue_index": i, "start": row["start"], "end": row["end"],
                          "coverage": 1.0, "consistent": True, "text": self.captions[i]["text"]}
                         for i, row in enumerate(timings)], "contradictions": []}

    def evaluate(self, captions=None, observations=None):
        return sync.evaluate_alignment(captions if captions is not None else self.captions,
                                       observations if observations is not None else self.observations(), **self.binding)

    def matches(self, report, captions=None, **kwargs):
        return sync.report_matches(report, captions if captions is not None else self.captions,
                                   audio_sha256=kwargs.get("audio_sha256", self.binding["audio_sha256"]),
                                   source_key=kwargs.get("source_key", self.binding["source_key"]))

    def call_engine(self, captions=None, **kwargs):
        return sync.run_sync(captions if captions is not None else self.captions, **self.binding, **kwargs)

    def test_complete_independent_check_can_pass_and_is_bound_to_text_and_audio(self):
        report = self.evaluate()
        self.assertEqual(report["status"], "passed")
        self.assertTrue(self.matches(report))
        changed = deepcopy(self.captions)
        changed[0]["text"] = "修改后的台词"
        self.assertFalse(self.matches(report, changed))
        changed = deepcopy(self.captions)
        changed[0]["start"] += 0.1
        self.assertFalse(self.matches(report, changed))
        self.assertFalse(self.matches(report, audio_sha256="c" * 64))
        self.assertFalse(self.matches(report, source_key="new-source"))

    def test_old_zero_second_timestamp_fails_by_actual_audio_evidence(self):
        early = deepcopy(self.captions)
        early[0].update(start=0, end=2.96)
        report = self.evaluate(early)
        self.assertEqual(report["status"], "needs_attention")
        self.assertIn("start_mismatch", {row["code"] for row in report["issues"]})
        self.assertIn("end_mismatch", {row["code"] for row in report["issues"]})
        self.assertFalse(self.matches(report, early))

    def test_single_global_shift_is_not_accepted_when_offsets_differ(self):
        early = [{"start": 0, "end": 2.3, "text": self.captions[0]["text"]},
                 {"start": 3.2, "end": 4.6, "text": self.captions[1]["text"]}]
        verify = Mock(side_effect=lambda cues: self.observations())
        def shift(cues, report, attempt):
            return [{**row, "start": row["start"] + (6.6 if attempt == 1 else 0),
                     "end": row["end"] + (6.6 if attempt == 1 else 0)} for row in cues]
        repair = Mock(side_effect=shift)
        current, report = self.call_engine(early, verify=verify, repair=repair)
        self.assertEqual(report["status"], "needs_attention")
        self.assertEqual(report["attempts"], 2)
        self.assertEqual(verify.call_count, 3)
        self.assertEqual(repair.call_count, 2)
        self.assertFalse(self.matches(report, current))

    def test_repair_targets_then_independent_recheck_before_pass(self):
        early = deepcopy(self.captions)
        early[0].update(start=0, end=2.96)
        verify = Mock(side_effect=lambda cues: self.observations())
        def apply(cues, report, attempt):
            for row in report["aligned_cues"]:
                cues[row["cue_index"]].update(start=row["start"], end=row["end"])
            return cues
        repair = Mock(side_effect=apply)
        progress = []
        progress_captions = []
        def on_progress(report, cues):
            progress.append(report)
            progress_captions.append(cues)
        current, report = self.call_engine(early, verify=verify, repair=repair, on_progress=on_progress)
        self.assertTrue(self.matches(report, current))
        self.assertEqual(report["attempts"], 1)
        self.assertEqual(report["input_caption_hash"], sync.caption_hash(early))
        self.assertEqual(verify.call_count, 2)
        self.assertEqual(repair.call_count, 1)
        self.assertEqual([r["status"] for r in progress], ["checking", "repairing", "checking", "passed"])
        self.assertEqual(progress[1]["attempts"], 1)
        self.assertEqual(progress_captions[1], early)
        self.assertEqual(progress_captions[2], current)
        for state, cues in zip(progress, progress_captions):
            self.assertEqual(state["caption_hash"], sync.caption_hash(cues))

    def test_independent_ctc_method_is_supported_but_still_needs_all_evidence(self):
        observed = self.observations()
        observed["method"] = "independent_ctc"
        self.assertTrue(self.matches(self.evaluate(observations=observed)))
        observed["cues"][0]["consistent"] = False
        self.assertFalse(self.matches(self.evaluate(observations=observed)))

    def test_missing_or_same_vad_or_dtw_evidence_never_passes(self):
        for method in (None, "vad", "dtw", "original_captions", "spelling_proofreading"):
            observed = self.observations()
            observed["method"] = method
            report = self.evaluate(observations=observed)
            self.assertEqual(report["status"], "needs_attention")
            self.assertFalse(report["independent_evidence"])
            self.assertFalse(self.matches(report))
        self.assertFalse(self.matches(self.evaluate(observations={})))

    def test_empty_subtitles_and_no_raw_evidence_are_not_automatic_pass(self):
        observed = self.observations()
        observed["cues"] = []
        self.assertEqual(self.evaluate([], observed)["status"], "needs_attention")

    def test_independent_whole_audio_silence_confirmation_can_pass_a_silent_episode(self):
        observed = self.observations()
        observed.update(cues=[], no_speech_confirmed=True)
        report = self.evaluate([], observed)
        self.assertTrue(self.matches(report, []))
        self.assertEqual((report["checked_cues"], report["total_cues"]), (0, 0))
        for key in ("independent", "full_audio_checked", "audio_sha256", "checker_identity"):
            invalid = deepcopy(observed)
            invalid.pop(key)
            self.assertFalse(self.matches(self.evaluate([], invalid), []))
        self.assertFalse(self.matches(self.evaluate(observations=observed)))

    def test_real_middle_character_omission_is_tolerated_only_at_explicit_policy_threshold(self):
        observed = self.observations()
        observed["cues"][0]["coverage"] = 0.8
        self.assertTrue(self.matches(self.evaluate(observations=observed)))
        observed["cues"][0]["coverage"] = 0.79
        self.assertFalse(self.matches(self.evaluate(observations=observed)))
        observed["cues"][0].update(coverage=0.8, consistent=False)
        self.assertFalse(self.matches(self.evaluate(observations=observed)))

    def test_coverage_checks_the_whole_track_not_only_existing_cues(self):
        observed = self.observations()
        observed["full_audio_checked"] = False
        self.assertIn("audio_coverage_unchecked", {r["code"] for r in self.evaluate(observations=observed)["issues"]})
        observed = self.observations()
        observed["contradictions"] = [{"start": 15, "end": 16, "code": "uncovered_speech",
                                      "reason": "合成测试：疑似人声缺少字幕"}]
        report = self.evaluate(observations=observed)
        self.assertFalse(self.matches(report))
        self.assertIn("uncovered_speech", {r["code"] for r in report["issues"]})

    def test_collapse_low_coverage_and_inconsistency_prevent_pass_and_targets(self):
        for change, code in (({"end": 6.62}, "alignment_collapsed"),
                             ({"coverage": 0.4}, "insufficient_coverage"),
                             ({"consistent": False}, "inconsistent_evidence"),
                             ({"boundary_supported": False}, "unsupported_boundary"),
                             ({"text": "别的台词"}, "evidence_text_mismatch")):
            observed = self.observations()
            observed["cues"][0].update(change)
            report = self.evaluate(observations=observed)
            self.assertIn(code, {r["code"] for r in report["issues"]})
            self.assertFalse(self.matches(report))
            self.assertNotIn(0, {r["cue_index"] for r in report["aligned_cues"]})

    def test_every_cue_must_be_checked_exactly_once(self):
        observed = self.observations()
        observed["cues"].pop()
        self.assertEqual(self.evaluate(observations=observed)["checked_cues"], 1)
        observed = self.observations()
        observed["cues"].append(deepcopy(observed["cues"][0]))
        self.assertIn("duplicate_cue_evidence", {r["code"] for r in self.evaluate(observations=observed)["issues"]})

    def test_policy_tolerances_are_explicit_and_report_invalidates_on_policy_change(self):
        observed = self.observations()
        observed["cues"][0].update(start=6.85, end=9.35)
        report = self.evaluate(observations=observed)
        self.assertTrue(self.matches(report))
        report["policy"]["start_tolerance"] = 1
        self.assertFalse(self.matches(report))

    def test_report_requires_identity_policy_evidence_and_complete_coverage(self):
        for key in ("schema", "policy", "checker_identity", "evidence_hash", "source_audio_sha256",
                    "source_key", "independent_evidence", "full_audio_checked", "checked_cues", "caption_hash"):
            report = self.evaluate()
            report.pop(key)
            self.assertFalse(self.matches(report), key)
        self.assertFalse(self.matches(None))
        for malformed in (None, False, 0, [], {}, "bad", {"policy": True}):
            for captions in (None, False, 0, [], {}, "bad"):
                self.assertFalse(self.matches(malformed, captions))
        report = self.evaluate()
        report["policy"]["version"] = True
        self.assertFalse(self.matches(report))

    def test_two_rounds_then_continue_cannot_create_an_infinite_repair_loop(self):
        verify = Mock(return_value={})
        repair = Mock(side_effect=lambda cues, report, attempt: cues)
        current, report = self.call_engine(verify=verify, repair=repair)
        self.assertEqual(report["attempts"], 2)
        self.assertEqual(report["status"], "needs_attention")
        self.assertEqual(repair.call_count, 2)
        for _ in range(3):
            current, report = self.call_engine(current, verify=verify, repair=repair, previous_report=report)
        self.assertEqual(repair.call_count, 2)
        self.assertEqual(verify.call_count, 3)

    def test_exhaustion_is_bound_to_initial_input_when_repair_changed_captions(self):
        verify = Mock(return_value={})
        def shift(cues, report, attempt):
            return [{**row, "start": row["start"] + 0.1, "end": row["end"] + 0.1} for row in cues]
        repair = Mock(side_effect=shift)
        current, report = self.call_engine(verify=verify, repair=repair)
        self.assertNotEqual(sync.caption_hash(current), sync.caption_hash(self.captions))
        self.call_engine(verify=verify, repair=repair, previous_report=report)
        self.assertEqual(repair.call_count, 2)

    def test_matching_pass_is_cached_without_rechecking_but_edited_text_is_not(self):
        report = self.evaluate()
        verify = Mock(side_effect=AssertionError("cached pass must not re-run"))
        repair = Mock(side_effect=AssertionError("cached pass must not repair"))
        self.call_engine(verify=verify, repair=repair, previous_report=report)
        verify.assert_not_called()
        changed = deepcopy(self.captions)
        changed[0]["text"] = "修改后的台词"
        _, report = self.call_engine(changed, verify=verify, repair=repair, previous_report=report)
        self.assertEqual(verify.call_count, 1)
        self.assertEqual(report["status"], "needs_attention")

    def test_inflight_repair_attempt_is_not_forgotten_on_restart(self):
        report = self.evaluate(observations={})
        report.update(status="repairing", attempts=2)
        verify = Mock(return_value={})
        repair = Mock(side_effect=AssertionError("must not launch third repair"))
        _, result = self.call_engine(verify=verify, repair=repair, previous_report=report)
        self.assertEqual(result["attempts"], 2)
        self.assertEqual(result["status"], "needs_attention")
        self.assertEqual(verify.call_count, 1)
        repair.assert_not_called()

    def test_checker_and_repair_errors_do_not_pass_or_expose_exception_text(self):
        _, report = self.call_engine(verify=Mock(side_effect=RuntimeError("SECRET")), repair=Mock())
        self.assertEqual(report["status"], "needs_attention")
        self.assertNotIn("SECRET", str(report))
        _, report = self.call_engine(verify=Mock(return_value={}), repair=Mock(side_effect=RuntimeError("SECRET")))
        self.assertEqual(report["attempts"], 1)
        self.assertNotIn("SECRET", str(report))

    def test_canonical_rejects_nonfinite_overlap_invalid_text_and_out_of_bounds(self):
        for row in ({"start": False, "end": 1, "text": "字"}, {"start": 0, "end": float("nan"), "text": "字"},
                    {"start": 1, "end": 1, "text": "字"}, {"start": 0, "end": 1, "text": ""},
                    {"start": 0, "end": 21, "text": "字"}):
            with self.assertRaises(ValueError):
                sync.canonical_captions([row], 20)
        with self.assertRaises(ValueError):
            sync.canonical_captions([{"start": 0, "end": 2, "text": "一"}, {"start": 1, "end": 3, "text": "二"}])


if __name__ == "__main__":
    unittest.main()
