"""Synthetic per-video model consent; no network, keys or paid requests."""
import copy
import unittest

from norlan import core, pipeline, prompt_gate
import test_prompt_gate as prompt_fixture


FAST = "doubao-seedance-2-0-fast-260128"
STANDARD = "doubao-seedance-2-0-260128"
MINI = "doubao-seedance-2-0-mini-260615"
V25 = "doubao-seedance-2-5-260628"


class VideoModelGateTests(unittest.TestCase):
    answer = prompt_fixture.PromptGateTests.answer
    amendment = prompt_fixture.PromptGateTests.amendment
    shot = prompt_fixture.PromptGateTests.shot

    def setUp(self):
        prompt_fixture.PromptGateTests.setUp(self)
        core.configure(self.root, {"ark": {"video_model": FAST}})

    def prepare(self):
        self.assertEqual(pipeline.run(self.root)["status"], "awaiting_prompt_approval")
        return copy.deepcopy(self.shot()["prompt_candidate"])

    def approve(self, candidate, model=STANDARD):
        return core.approve_prompt(self.root, self.shot()["id"], candidate["id"],
                                   candidate["request_hash"], video_model=model)

    def edit(self, candidate, text, model=STANDARD):
        return core.edit_prompt_and_approve(self.root, self.shot()["id"], candidate["id"],
                                           candidate["request_hash"], text, video_model=model)

    def assert_current_binding(self, model):
        state = core.load_project(self.root)
        shot = state["episodes"][0]["shots"][0]
        candidate = shot["prompt_candidate"]
        approval = shot["prompt_approval"]
        self.assertEqual(shot["video_model"], model)
        self.assertEqual(candidate["video_model"], model)
        self.assertEqual(approval["video_model"], model)
        self.assertEqual(approval["candidate_id"], candidate["id"])
        self.assertEqual(approval["request_hash"], candidate["request_hash"])
        specification = prompt_gate.specification(state, shot, candidate["prompt"],
            candidate["reference_manifest"], candidate["refined"])
        self.assertEqual(specification["provider"]["video_model"], model)
        self.assertEqual(prompt_gate._hash(specification), candidate["request_hash"])
        return candidate

    def test_model_choice_rebinds_exact_prompt_and_references_without_gpt_or_payment(self):
        original = self.prepare()
        calls = self.brain.call_count
        self.approve(original)
        candidate = self.assert_current_binding(STANDARD)
        self.assertNotEqual(candidate["id"], original["id"])
        self.assertNotEqual(candidate["request_hash"], original["request_hash"])
        for key in ("prompt", "refined", "reference_manifest", "version", "revision"):
            self.assertEqual(candidate[key], original[key])
        archived = self.shot()["prompt_history"][-1]
        self.assertEqual(archived["id"], original["id"])
        self.assertEqual(archived["video_model"], FAST)
        self.assertEqual(archived["status"], "superseded_by_model_selection")
        self.assertEqual(core.load_project(self.root)["settings"]["ark"]["video_model"], FAST)
        self.assertEqual(self.brain.call_count, calls)
        self.api.submit_video.assert_not_called()

    def test_same_model_preserves_candidate_identity_and_pins_default(self):
        original = self.prepare()
        self.approve(original, FAST)
        candidate = self.assert_current_binding(FAST)
        self.assertEqual(candidate["id"], original["id"])
        self.assertEqual(candidate["request_hash"], original["request_hash"])
        self.assertNotIn("prompt_history", self.shot())

    def test_manual_prompt_and_model_are_adopted_together_without_gpt_rewriting(self):
        original = self.prepare()
        raw = "  用户完整定稿。\n原文、换行和空格全部保留。  "
        calls = self.brain.call_count
        self.edit(original, raw, MINI)
        candidate = self.assert_current_binding(MINI)
        self.assertEqual(candidate["prompt"], raw)
        self.assertEqual(candidate["reference_manifest"], original["reference_manifest"])
        self.assertEqual(candidate["manual_prompt"]["source_candidate_id"], original["id"])
        self.assertEqual(candidate["manual_prompt"]["source_request_hash"], original["request_hash"])
        archived = self.shot()["prompt_history"][-1]
        self.assertEqual(archived["prompt"], original["prompt"])
        self.assertEqual(archived["video_model"], FAST)
        self.assertEqual(archived["status"], "superseded_by_manual_edit")
        self.assertEqual(prompt_gate.approved(self.root, self.shot()), candidate)
        self.assertEqual(self.brain.call_count, calls)
        self.api.submit_video.assert_not_called()

    def test_later_default_change_cannot_retarget_approved_model(self):
        original = self.prepare()
        self.approve(original)
        approved = copy.deepcopy(self.shot()["prompt_candidate"])
        core.configure(self.root, {"ark": {"video_model": MINI}})
        self.assertEqual(prompt_gate.approved(self.root, self.shot()), approved)
        self.assert_current_binding(STANDARD)
        self.api.submit_video.assert_not_called()

    def test_model_and_approval_tampering_cannot_submit(self):
        original = self.prepare()
        self.approve(original)
        with core.transaction(self.root) as state:
            state["episodes"][0]["shots"][0]["prompt_approval"]["video_model"] = MINI
        with self.assertRaises(core.GateError):
            prompt_gate.approved(self.root, self.shot())
        self.api.submit_video.assert_not_called()

    def test_candidate_model_metadata_is_checked_against_bound_specification(self):
        original = self.prepare()
        with core.transaction(self.root) as state:
            state["episodes"][0]["shots"][0]["prompt_candidate"]["video_model"] = MINI
        with self.assertRaises(core.GateError):
            self.approve(original)
        self.assertNotIn("prompt_approval", self.shot())
        self.api.submit_video.assert_not_called()

    def test_invalid_choices_are_atomic_for_normal_and_manual_generation(self):
        original = self.prepare()
        for model in ("unknown-new-model", "doubao-seedance-1-0-pro-250528", "", True, []):
            for manual in (False, True):
                with self.subTest(model=model, manual=manual):
                    before = core.load_project(self.root)
                    with self.assertRaises(core.GateError):
                        (self.edit(original, "用户尚未提交的文本", model) if manual
                         else self.approve(original, model))
                    self.assertEqual(core.load_project(self.root), before)
        self.api.submit_video.assert_not_called()

    def test_incompatible_duration_is_blocked_without_changing_storyboard_or_prompt(self):
        with core.transaction(self.root) as state:
            state["episodes"][0]["shots"][0]["duration"] = 20
        original = self.prepare()
        before = core.load_project(self.root)
        with self.assertRaisesRegex(core.GateError, "时长"):
            self.approve(original, FAST)
        self.assertEqual(core.load_project(self.root), before)
        self.approve(original, V25)
        self.assert_current_binding(V25)
        self.assertEqual(self.shot()["duration"], 20)
        self.assertEqual(self.shot()["prompt_candidate"]["prompt"], original["prompt"])
        self.api.submit_video.assert_not_called()

    def test_incompatible_resolution_is_blocked_without_silent_downgrade(self):
        core.configure(self.root, {"ark": {"video_model": STANDARD, "resolution": "1080p"}})
        original = self.prepare()
        before = core.load_project(self.root)
        with self.assertRaisesRegex(core.GateError, "分辨率"):
            self.approve(original, FAST)
        self.assertEqual(core.load_project(self.root), before)
        self.assertEqual(core.load_project(self.root)["settings"]["ark"]["resolution"], "1080p")
        self.api.submit_video.assert_not_called()

    def test_stale_second_submission_cannot_change_already_chosen_model(self):
        original = self.prepare()
        self.approve(original)
        before = core.load_project(self.root)
        with self.assertRaises(core.GateError):
            self.approve(original, MINI)
        with self.assertRaises(core.GateError):
            self.edit(original, "过期页面的新文本", MINI)
        self.assertEqual(core.load_project(self.root), before)
        self.api.submit_video.assert_not_called()

    def test_legacy_candidate_without_model_metadata_remains_usable_without_gpt(self):
        original = self.prepare()
        with core.transaction(self.root) as state:
            state["episodes"][0]["shots"][0]["prompt_candidate"].pop("video_model")
        calls = self.brain.call_count
        self.approve(original, None)
        candidate = self.assert_current_binding(FAST)
        self.assertEqual(candidate["id"], original["id"])
        self.assertEqual(candidate["request_hash"], original["request_hash"])
        self.assertEqual(self.brain.call_count, calls)
        self.api.submit_video.assert_not_called()


if __name__ == "__main__":
    unittest.main()
