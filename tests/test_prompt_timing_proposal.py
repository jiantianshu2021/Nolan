"""Draft UI/audio schedules share one exact, human-approved video request."""
import copy
from pathlib import Path
import tempfile
import unittest

from norlan import core, pipeline, prompt_gate
from norlan.segments import timeline_prompt


class PromptTimingProposalTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        core.create_project(self.root, "合成声音时间安排测试。", "live")
        core.install_plan(self.root, pipeline.demo_plan())
        core.approve(self.root, "plan", "plan", 1)
        with core.transaction(self.root) as state:
            for index, asset in enumerate(state["assets"]):
                relative = f"media/approved-{index}.png"
                path = self.root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(f"SYNTHETIC APPROVED REFERENCE {index}".encode())
                sha = core.digest(path)
                asset.update(status="approved", version=1, file=relative, sha256=sha,
                             approved_sha256=sha, approved_at=core.now())
            shot = state["episodes"][0]["shots"][0]
            shot.update(version=0, revision=2, status="planned", duration=15,
                        dialogue="系统播报员：危机响应，系统绑定成功。\n楚天行：知道了。",
                        subshots=[{"start": 0, "end": 5, "prompt": "固定机位，角色走近柜体；此时没有任务框。"},
                                  {"start": 5, "end": 15, "prompt": "5秒后任务框出现，系统对白在第二镜内完整说完。"}])
        self.shot_id = self.shot()["id"]

    def shot(self):
        return core.load_project(self.root)["episodes"][0]["shots"][0]

    def refined(self):
        return {"prompt": "柜体保持原位置，摄影切点不变；声音按本次连续时窗安排。",
                "dialogue": self.shot()["dialogue"], "continuity_notes": "只调整任务框与播音起点，不改变对白。",
                "timing_proposal": {
                    "summary": "任务框在0.5秒出现，系统播音跨过5秒摄影切点连续说完。",
                    "subshots": [
                        {"start": 0, "end": 5, "prompt": "固定机位，角色走近柜体；0.5秒任务框出现并开始画外播音。"},
                        {"start": 5, "end": 15, "prompt": "按原5秒切镜，任务框保留，系统播音不断句继续；结尾楚天行回答。"}],
                    "audio_windows": [
                        {"start": .5, "end": 13.5, "speaker": "系统播报员", "text": "危机响应，系统绑定成功。"},
                        {"start": 13.6, "end": 15, "speaker": "楚天行", "text": "知道了。"}]}}

    def manifest(self):
        state = core.load_project(self.root)
        assets = [core._find(state, "asset", identifier) for identifier in self.shot()["asset_ids"]]
        return {"images": [asset["file"] for asset in assets], "audio": [], "observations": [],
                "context": {"image_references": [
                    {"index": index, "role": "approved_asset", "asset_id": asset["id"], "sha256": asset["sha256"]}
                    for index, asset in enumerate(assets, 1)], "audio_references": []}}

    def final_prompt(self, refined):
        return pipeline._final_shot_prompt(core.load_project(self.root), self.shot(), refined,
                                           self.manifest()["context"])

    def prepare(self):
        refined = self.refined()
        return prompt_gate.prepare(self.root, self.shot_id, refined, self.final_prompt(refined), self.manifest())

    def test_cross_cut_audio_draft_is_one_candidate_without_changing_canonical_plan(self):
        original = copy.deepcopy(self.shot())
        candidate = self.prepare()
        self.assertEqual(self.shot()["subshots"], original["subshots"])
        self.assertEqual(self.shot()["duration"], original["duration"])
        self.assertEqual(self.shot()["asset_ids"], original["asset_ids"])
        self.assertEqual(candidate["timing_proposal"], self.refined()["timing_proposal"])
        self.assertEqual(candidate["timing_summary"], candidate["timing_proposal"]["summary"])
        self.assertEqual(candidate["effective_subshots"], candidate["timing_proposal"]["subshots"])
        self.assertEqual(candidate["version"], 1)
        self.assertEqual(candidate["revision"], 2)
        self.assertEqual(core.load_project(self.root)["jobs"], {})

        prompt_gate.approve(self.root, self.shot_id, candidate["id"], candidate["request_hash"])
        approved = prompt_gate.approved(self.root, self.shot())
        self.assertEqual(approved["prompt"], candidate["prompt"])
        self.assertEqual(approved["timing_proposal"], candidate["timing_proposal"])
        self.assertEqual(self.shot()["subshots"], original["subshots"])
        self.assertEqual(core.load_project(self.root)["jobs"], {})

    def test_final_prompt_uses_one_draft_schedule_without_old_cut_bound_speech_rule(self):
        refined = self.refined()
        prompt = self.final_prompt(refined)
        timeline = prompt_gate.candidate_timeline_prompt(self.shot(), refined)
        self.assertEqual(prompt.count(timeline), 1)
        self.assertIn("0.5–13.5秒，系统播报员", prompt)
        self.assertIn("音频可以跨摄影切点连续说完", prompt)
        self.assertNotIn("此时没有任务框", prompt)
        self.assertNotIn("系统对白在第二镜内完整说完", prompt)
        self.assertNotIn("各内部镜头的对白必须在所属时段内完整", prompt)
        self.assertIn(self.shot()["dialogue"], prompt)

    def test_prepare_rejects_missing_duplicated_or_conflicting_timeline(self):
        refined = self.refined()
        final = self.final_prompt(refined)
        timeline = prompt_gate.candidate_timeline_prompt(self.shot(), refined)
        for text in (refined["prompt"], final + "\n" + timeline,
                     final + "\n" + timeline_prompt(self.shot())):
            with self.subTest(prompt=text[:30]), self.assertRaises(core.GateError):
                prompt_gate.prepare(self.root, self.shot_id, refined, text, self.manifest())
        self.assertNotIn("prompt_candidate", self.shot())
        self.assertEqual(core.load_project(self.root)["jobs"], {})

    def test_proposal_cannot_change_camera_count_cuts_duration_references_or_dialogue(self):
        mutations = {
            "camera_count": lambda value: value["timing_proposal"]["subshots"].pop(),
            "camera_cut": lambda value: (value["timing_proposal"]["subshots"][0].update(end=4),
                                          value["timing_proposal"]["subshots"][1].update(start=4)),
            "duration_field": lambda value: value["timing_proposal"].update(duration=20),
            "asset_refs": lambda value: value.update(asset_ids=["new-unapproved-person"]),
            "dialogue": lambda value: value.update(dialogue="系统播报员：改写的内容。"),
        }
        before = copy.deepcopy(self.shot())
        for label, mutate in mutations.items():
            with self.subTest(label=label):
                refined = self.refined()
                mutate(refined)
                with self.assertRaises(core.GateError):
                    prompt_gate.validate_timing_proposal(self.shot(), refined)
        self.assertEqual(self.shot(), before)

    def test_audio_windows_cannot_add_omit_repeat_or_reassign_dialogue(self):
        mutations = {
            "new_speaker": lambda value: value[0].update(speaker="新角色"),
            "added_text": lambda value: value[0].update(text=value[0]["text"] + "新增一句。"),
            "missing_hero": lambda value: value.pop(),
            "repeated_text": lambda value: value[0].update(text=value[0]["text"] * 2),
            "swapped_attribution": lambda value: value[0].update(speaker="楚天行"),
        }
        for label, mutate in mutations.items():
            with self.subTest(label=label):
                refined = self.refined()
                mutate(refined["timing_proposal"]["audio_windows"])
                with self.assertRaises(core.GateError):
                    prompt_gate.validate_timing_proposal(self.shot(), refined)

    def test_audio_windows_must_stay_finite_ordered_and_inside_original_duration(self):
        mutations = {
            "negative_start": {"start": -.1}, "past_end": {"end": 15.1},
            "zero_length": {"end": .5}, "nan": {"start": float("nan")},
            "infinity": {"end": float("inf")}, "bool": {"start": False},
        }
        for label, changes in mutations.items():
            with self.subTest(label=label):
                refined = self.refined()
                refined["timing_proposal"]["audio_windows"][0].update(changes)
                with self.assertRaises(core.GateError):
                    prompt_gate.validate_timing_proposal(self.shot(), refined)

    def test_split_sentence_can_continue_across_cut_without_changing_its_words(self):
        refined = self.refined()
        refined["timing_proposal"]["audio_windows"] = [
            {"start": .5, "end": 5, "speaker": "系统播报员", "text": "危机响应，"},
            {"start": 5, "end": 13.5, "speaker": "系统播报员", "text": "系统绑定成功。"},
            {"start": 13.6, "end": 15, "speaker": "楚天行", "text": "知道了。"}]
        self.assertEqual(prompt_gate.validate_timing_proposal(self.shot(), refined), refined["timing_proposal"])
        refined["timing_proposal"]["audio_windows"][1]["start"] = 4.9
        with self.assertRaisesRegex(core.GateError, "重叠"):
            prompt_gate.validate_timing_proposal(self.shot(), refined)

    def test_system_broadcast_cannot_overlap_another_characters_line(self):
        refined = self.refined()
        refined["timing_proposal"]["audio_windows"][1]["start"] = 13
        with self.assertRaisesRegex(core.GateError, "覆盖其他角色"):
            prompt_gate.validate_timing_proposal(self.shot(), refined)

    def test_hash_binds_timing_metadata_even_when_final_prompt_stays_identical(self):
        state, shot, manifest = core.load_project(self.root), self.shot(), self.manifest()
        refined = self.refined()
        prompt = self.final_prompt(refined)
        original = prompt_gate._hash(prompt_gate.specification(state, shot, prompt, manifest, refined))
        changed = copy.deepcopy(refined)
        changed["timing_proposal"]["summary"] += "另一个说明。"
        self.assertNotEqual(original, prompt_gate._hash(prompt_gate.specification(state, shot, prompt, manifest, changed)))
        changed = copy.deepcopy(refined)
        changed["continuity_notes"] += "另一个连续性条件。"
        self.assertNotEqual(original, prompt_gate._hash(prompt_gate.specification(state, shot, prompt, manifest, changed)))

    def test_changed_draft_metadata_cannot_reuse_confirmation(self):
        candidate = self.prepare()
        with core.transaction(self.root) as state:
            stored = core._find(state, "shot", self.shot_id)["prompt_candidate"]
            stored["timing_summary"] += "已改变。"
        with self.assertRaises(core.GateError):
            prompt_gate.approve(self.root, self.shot_id, candidate["id"], candidate["request_hash"])
        self.assertNotIn("prompt_approval", self.shot())
        self.assertEqual(core.load_project(self.root)["jobs"], {})

    def test_changed_refinement_after_approval_invalidates_authorization(self):
        candidate = self.prepare()
        prompt_gate.approve(self.root, self.shot_id, candidate["id"], candidate["request_hash"])
        with core.transaction(self.root) as state:
            stored = core._find(state, "shot", self.shot_id)["prompt_candidate"]
            stored["refined"]["continuity_notes"] += "未经确认的新条件。"
            stored["continuity_notes"] = stored["refined"]["continuity_notes"]
        with self.assertRaises(core.GateError):
            prompt_gate.approved(self.root, self.shot())
        self.assertEqual(self.shot()["status"], "planned")
        self.assertNotIn("prompt_approval", self.shot())
        self.assertEqual(self.shot()["prompt_history"][-1]["status"], "invalidated")

    def test_reference_or_canonical_timeline_change_invalidates_approved_draft(self):
        for field in ("asset_version", "canonical_subshots", "prompt"):
            with self.subTest(field=field):
                candidate = self.prepare()
                prompt_gate.approve(self.root, self.shot_id, candidate["id"], candidate["request_hash"])
                with core.transaction(self.root) as state:
                    shot = core._find(state, "shot", self.shot_id)
                    if field == "asset_version":
                        core._find(state, "asset", shot["asset_ids"][0])["version"] += 1
                    elif field == "canonical_subshots":
                        shot["subshots"][0]["prompt"] += "未经确认的原计划变动。"
                    else:
                        shot["prompt_candidate"]["prompt"] += "未经确认的新最终提示。"
                with self.assertRaises(core.GateError):
                    prompt_gate.approved(self.root, self.shot())
                self.assertNotIn("prompt_approval", self.shot())
                self.assertEqual(core.load_project(self.root)["jobs"], {})

    def test_legacy_no_proposal_keeps_original_timeline_and_request_hash(self):
        state, shot, manifest = core.load_project(self.root), self.shot(), self.manifest()
        legacy = {"prompt": "旧流程的最终提示。", "dialogue": "", "continuity_notes": "旧流程。"}
        for refined in (legacy, {**legacy, "timing_proposal": None}):
            with self.subTest(nullable="timing_proposal" in refined):
                self.assertEqual(prompt_gate.candidate_timeline_prompt(shot, refined), timeline_prompt(shot))
                self.assertEqual(prompt_gate.effective_shot(shot, refined), shot)
                prompt = self.final_prompt(refined)
                self.assertEqual(prompt_gate.specification(state, shot, prompt, manifest, refined),
                                 prompt_gate.specification(state, shot, prompt, manifest))
                candidate = prompt_gate.prepare(self.root, self.shot_id, refined, prompt, manifest)
                self.assertNotIn("timing_proposal", candidate)


if __name__ == "__main__":
    unittest.main()
