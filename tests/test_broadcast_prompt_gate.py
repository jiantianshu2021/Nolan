"""Synthetic system-voice edits never grant media generation permission."""
import copy
from pathlib import Path
import tempfile
import unittest

from norlan import broadcast, core, pipeline, prompt_gate


FULL = "技术须以民用产品发布。普通售价2999元。震惊值升级加工台；真实使用满意值兑换新图纸。"
SHORT = "技术须以民用产品发布。"


def shot_and_refined():
    shot = {"id": "s01", "duration": 15, "scene_id": "room", "asset_ids": ["hero", "system", "system_voice"],
            "dialogue": "楚阳：要我喂你？\n系统播报员：" + FULL,
            "subshots": [{"start": 0, "end": 5, "prompt": "人物问话，女声逐字读完通知。"},
                         {"start": 5, "end": 15, "prompt": "系统完整播读规则，人物反应。"}]}
    refined = {"prompt": "保持原动作和切点，系统简播，全文分屏显示，留出人物反应。",
               "dialogue": "楚阳：要我喂你？\n系统播报员：" + SHORT, "continuity_notes": "同一声音，人物对白不改。",
               "broadcast_proposal": {"speaker": "系统播报员", "source_text": FULL, "spoken_text": SHORT,
                                      "display_text": FULL, "summary": "只播发布条件，价格与兑换细则留画面。"},
               "timing_proposal": {"summary": "人物问话后简播，界面全文分屏，后半段保留反应。",
                                   "subshots": [{"start": 0, "end": 5, "prompt": "人物问原句，系统简播，完整规则界面开始展示。"},
                                                {"start": 5, "end": 15, "prompt": "切镜承接人物反应，规则继续分屏展示。"}],
                                   "audio_windows": [{"start": .4, "end": 1.6, "speaker": "楚阳", "text": "要我喂你？"},
                                                     {"start": 2, "end": 5.5, "speaker": "系统播报员", "text": SHORT}]}}
    return shot, refined


def context():
    return {"assets": [{"id": "hero", "kind": "character", "name": "楚阳"},
                       {"id": "system", "kind": "character", "name": "系统播报员（仅画外声源）", "voice_only": True},
                       {"id": "system_voice", "kind": "voice", "character_id": "system", "audition_speaker": "系统"}],
            "audio_policy": {"system_broadcast": {"enabled": True, "character_id": "system", "voice_id": "system_voice",
                                                   "instruction": "每次完整逐字朗读所有系统规则。"}}}


class BroadcastPromptGateTests(unittest.TestCase):
    def test_system_selection_keeps_source_plan_and_other_dialogue_immutable(self):
        shot, refined = shot_and_refined()
        before = copy.deepcopy(shot)
        self.assertEqual(prompt_gate.validate_broadcast_proposal(shot, refined, context()), refined["broadcast_proposal"])
        effective = prompt_gate.effective_shot(shot, refined)
        self.assertEqual(effective["dialogue"], refined["dialogue"])
        self.assertEqual(shot, before)
        self.assertEqual(effective["duration"], 15)
        self.assertEqual(effective["asset_ids"], shot["asset_ids"])

    def test_characters_sources_visual_rules_and_numbers_are_protected(self):
        mutations = {
            "hero": lambda r: r.update(dialogue=r["dialogue"].replace("要我喂你？", "新增台词。")),
            "missing_visual": lambda r: r["broadcast_proposal"].update(display_text=SHORT),
            "forged_source": lambda r: r["broadcast_proposal"].update(source_text=SHORT),
            "new_number": lambda r: (r["broadcast_proposal"].update(spoken_text="售价3999元。"),
                                     r.update(dialogue="楚阳：要我喂你？\n系统播报员：售价3999元。")),
            "missing_timing": lambda r: r.update(timing_proposal=None),
            "old_full_read_windows": lambda r: r["timing_proposal"]["audio_windows"][1].update(text=FULL),
        }
        for label, mutate in mutations.items():
            with self.subTest(label=label):
                shot, refined = shot_and_refined()
                mutate(refined)
                with self.assertRaises(core.GateError):
                    prompt_gate.validate_timing_proposal(shot, refined)

    def test_arbitrary_protagonist_cannot_be_designated_system_by_model(self):
        shot, refined = shot_and_refined()
        refined["broadcast_proposal"].update(speaker="楚阳", source_text="要我喂你？", spoken_text="喂你？", display_text="要我喂你？")
        refined["dialogue"] = "楚阳：喂你？\n系统播报员：" + FULL
        with self.assertRaisesRegex(core.GateError, "系统声源"):
            prompt_gate.validate_broadcast_proposal(shot, refined, context())
        shot, refined = shot_and_refined()
        disabled = context()
        disabled["audio_policy"]["system_broadcast"]["enabled"] = False
        with self.assertRaises(core.GateError):
            prompt_gate.validate_broadcast_proposal(shot, refined, disabled)

    def test_legacy_full_read_policy_is_normalized_without_modifying_saved_policy(self):
        original = context()
        before = copy.deepcopy(original)
        policy = broadcast.policy_for(original)
        self.assertEqual(policy["editorial_mode"], "key_points")
        self.assertIn("系统播报员", policy["speaker_labels"])
        self.assertIn("只读剧情必需", policy["instruction"])
        self.assertNotIn("每次完整逐字朗读", policy["instruction"])
        self.assertEqual(original, before)

    def test_legacy_single_camera_candidate_cannot_show_different_dialogue(self):
        shot, refined = shot_and_refined()
        shot.pop("subshots")
        refined["timing_proposal"] = None
        candidate = {"refined": copy.deepcopy(refined), "broadcast_proposal": refined["broadcast_proposal"],
                     "dialogue": refined["dialogue"], "continuity_notes": refined["continuity_notes"],
                     "prompt": prompt_gate.candidate_timeline_prompt(shot, refined)}
        prompt_gate._validate_candidate_timing(shot, candidate)
        candidate["dialogue"] += "未经确认的内容"
        with self.assertRaises(core.GateError):
            prompt_gate._validate_candidate_timing(shot, candidate)

    def test_source_and_selected_words_are_bound_to_exact_single_confirmation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            core.create_project(root, "SYNTHETIC system broadcast test", "live")
            core.install_plan(root, pipeline.demo_plan())
            core.approve(root, "plan", "plan", 1)
            shot, refined = shot_and_refined()
            with core.transaction(root) as state:
                state["assets"].extend(context()["assets"][1:])
                state["audio_policy"] = context()["audio_policy"]
                actual = state["episodes"][0]["shots"][0]
                shot.update(id=actual["id"], scene_id=actual["scene_id"])
                actual.update(shot)
                shot = copy.deepcopy(actual)
                for index, asset in enumerate(state["assets"]):
                    if core.is_internal_voice_source(asset):
                        core._register_voice_source(asset)
                        continue
                    path = root / f"media/reference-{index}.png"
                    path.parent.mkdir(exist_ok=True)
                    path.write_bytes(b"SYNTHETIC REFERENCE")
                    asset.update(status="approved", version=1, file=str(path.relative_to(root)), sha256=core.digest(path))
                    if asset.get("kind") == "voice":
                        asset.update(audio_file=str(path.relative_to(root)), audio_sha256=core.digest(path))
            state = core.load_project(root)
            prompt = prompt_gate.candidate_timeline_prompt(shot, refined)
            candidate = prompt_gate.prepare(root, shot["id"], refined, prompt, {"images": [], "audio": [], "context": {}})
            self.assertEqual(core.load_project(root)["jobs"], {})
            with self.assertRaises(core.GateError):
                prompt_gate.approved(root, core._find(core.load_project(root), "shot", shot["id"]))
            prompt_gate.approve(root, shot["id"], candidate["id"], candidate["request_hash"])
            approved = prompt_gate.approved(root, core._find(core.load_project(root), "shot", shot["id"]))
            self.assertEqual(approved["broadcast_proposal"]["display_text"], FULL)
            self.assertEqual(core.load_project(root)["jobs"], {})
            changed = copy.deepcopy(refined)
            changed["broadcast_proposal"]["summary"] += "changed"
            self.assertNotEqual(candidate["request_hash"], prompt_gate._hash(
                prompt_gate.specification(state, shot, prompt, {"images": [], "audio": [], "context": {}}, changed)))
            with core.transaction(root) as current:
                core._find(current, "shot", shot["id"])["prompt_candidate"]["broadcast_proposal"]["spoken_text"] += "未经确认"
            with self.assertRaises(core.GateError):
                prompt_gate.approved(root, core._find(core.load_project(root), "shot", shot["id"]))


if __name__ == "__main__":
    unittest.main()
