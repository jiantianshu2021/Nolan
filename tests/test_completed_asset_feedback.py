"""Synthetic completed-asset cascade recovery; never call paid services."""
import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from norlan import core, pipeline


class CompletedAssetFeedbackTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        core.create_project(self.root, "合成资产声音返工记录。")
        core.install_plan(self.root, pipeline.demo_plan())
        self.reason = "合成男声太急，需要选更稳定的原文试音。"
        self.machine = "资产变更：" + self.reason
        self.source_ids = ["old-S01", "old-S02"]
        self.bases = ["人物拿起工具，保持构图。", "人物看向门口，保持动作方向。"]
        for relative, data in (("media/voice-v3.mp4", b"SYNTHETIC VOICE V3"),
                               ("media/voice-v3.wav", b"SYNTHETIC AUDIO V3")):
            (self.root / relative).write_bytes(data)
        with core.transaction(self.root) as state:
            video_sha = core.digest(self.root / "media/voice-v3.mp4")
            state["assets"].append({"id": "voice", "kind": "voice", "character_id": "hero", "name": "合成已批准男声",
                "version": 3, "status": "approved", "file": "media/voice-v3.mp4", "sha256": video_sha,
                "approved_sha256": video_sha, "approved_at": core.now(),
                "audio_file": "media/voice-v3.wav", "audio_sha256": core.digest(self.root / "media/voice-v3.wav")})
            core.event(state, "human_rejected", kind="asset", id="voice", version=2, feedback=self.reason)
            core.event(state, "human_approved", kind="asset", id="voice", version=3, sha256=video_sha)
            shot = state["episodes"][0]["shots"][0]
            self.shot_id = shot["id"]
            shot.update(duration=10, dialogue="主角：保持这句原台词。", asset_ids=["hero", "station", "voice"],
                source_shot_ids=self.source_ids, feedback="\n".join(f"{identifier}：{self.machine}" for identifier in self.source_ids),
                source_shot_instructions=[{"id": identifier, "revision": 1, "feedback": self.machine} for identifier in self.source_ids],
                subshots=[{"start": i * 5, "end": (i + 1) * 5,
                    "prompt": base + self.trailer(identifier, i * 5, (i + 1) * 5)}
                    for i, (identifier, base) in enumerate(zip(self.source_ids, self.bases))],
                revision_analysis={"signature": "SYNTHETIC OLD DIAGNOSIS", "blockers": ["重复要求新增同一声音。"]},
                prompt_candidate={"id": "SYNTHETIC UNSUBMITTED", "status": "candidate"})

    def trailer(self, identifier, start, end):
        return (f"\n原生成片段 {identifier} 的返工意见（仅适用于本生成片段 {start}–{end} 秒的来源段；"
                "覆盖该来源段的内部镜头，保留各句对白的归属，不作为每镜新增对白）：\n" + self.machine)

    def shot(self):
        return core._find(core.load_project(self.root), "shot", self.shot_id)

    def resolve(self):
        with core.run_lock(self.root):
            return core.resolve_completed_asset_feedback(self.root, self.shot_id)

    def test_exact_proof_consumes_only_bookkeeping_and_preserves_full_original_history(self):
        original = self.shot()
        with patch.object(pipeline, "_brain", side_effect=AssertionError("No paid calls")), \
                patch.object(pipeline, "_ark", side_effect=AssertionError("No paid calls")):
            self.assertTrue(self.resolve())
        current = self.shot()
        self.assertEqual(current["feedback"], "")
        self.assertEqual([item["prompt"] for item in current["subshots"]], self.bases)
        self.assertTrue(all(item["feedback"] == "" for item in current["source_shot_instructions"]))
        for key in ("version", "file", "job_id", "dialogue", "duration", "scene_id", "asset_ids", "revision"):
            self.assertEqual(current.get(key), original.get(key), key)
        self.assertEqual([(item["start"], item["end"]) for item in current["subshots"]], [(0, 5), (5, 10)])
        archive = current["resolved_asset_feedback_history"][0]
        for key in ("feedback", "subshots", "source_shot_instructions", "revision_analysis"):
            self.assertEqual(archive[key], original[key])
        self.assertEqual(archive["proof"][0]["approved_event"]["version"], 3)
        self.assertEqual(current["revision_analysis_history"][-1]["resolution"], "completed_asset_feedback")
        self.assertEqual(current["prompt_history"][-1]["status"], "invalidated")
        self.assertFalse(self.resolve())
        self.assertEqual(len(self.shot()["resolved_asset_feedback_history"]), 1)

    def test_raw_unregrouped_machine_feedback_can_be_resolved(self):
        with core.transaction(self.root) as state:
            shot = core._find(state, "shot", self.shot_id)
            shot["feedback"] = self.machine
            for key in ("source_shot_ids", "source_shot_instructions", "subshots"):
                shot.pop(key)
        self.assertTrue(self.resolve())
        self.assertEqual(self.shot()["feedback"], "")

    def test_pipeline_resolves_the_completed_cascade_before_any_revision_planning(self):
        def inspect_revision(root, shot):
            self.assertEqual(shot["feedback"], "")
            self.assertEqual([item["prompt"] for item in shot["subshots"]], self.bases)
            raise core.GateError("SYNTHETIC STOP BEFORE MODEL OR PROVIDER")

        with core.run_lock(self.root), patch.object(pipeline, "_revision_analysis", side_effect=inspect_revision) as planner, \
                patch.object(pipeline, "_brain") as brain, patch.object(pipeline, "_ark") as provider:
            with self.assertRaisesRegex(core.GateError, "SYNTHETIC STOP"):
                pipeline._shot(self.root, "E001", self.shot_id, None)
        planner.assert_called_once()
        brain.assert_not_called()
        provider.assert_not_called()

    def test_missing_current_sha_bound_approval_or_changed_asset_is_never_consumed(self):
        baseline = core.load_project(self.root)
        changes = {
            "candidate": lambda state: state["assets"][-1].update(status="candidate"),
            "approved_hash": lambda state: state["assets"][-1].update(approved_sha256="0" * 64),
            "event_hash": lambda state: state["events"][-1].update(sha256="0" * 64),
            "event_version": lambda state: state["events"][-1].update(version=2),
            "missing_event": lambda state: state["events"].pop(),
            "unknown_feedback": lambda state: core._find(state, "shot", self.shot_id).update(feedback="资产变更：未知指令。"),
        }
        for label, change in changes.items():
            with self.subTest(label=label):
                with core.transaction(self.root) as state:
                    state.clear(); state.update(copy.deepcopy(baseline)); change(state)
                original = self.shot()
                self.assertFalse(self.resolve())
                self.assertEqual(self.shot(), original)
        with core.transaction(self.root) as state:
            state.clear(); state.update(copy.deepcopy(baseline))
        (self.root / "media/voice-v3.wav").write_bytes(b"SYNTHETIC TAMPERED AUDIO")
        self.assertFalse(self.resolve())

    def test_mixed_unknown_source_or_modified_terminal_block_is_not_consumed(self):
        baseline = core.load_project(self.root)
        for label in ("mixed", "unknown_source", "changed_trailer"):
            with self.subTest(label=label):
                with core.transaction(self.root) as state:
                    state.clear(); state.update(copy.deepcopy(baseline))
                    shot = core._find(state, "shot", self.shot_id)
                    if label == "mixed":
                        shot["feedback"] += "\n另请修改人物站位。"
                    elif label == "unknown_source":
                        shot["feedback"] = "UNKNOWN：" + self.machine
                    else:
                        shot["subshots"][0]["prompt"] += "\n用户后来补充的新动作。"
                original = self.shot()
                self.assertFalse(self.resolve())
                self.assertEqual(self.shot(), original)

    def test_fresh_human_shot_prompt_or_planning_feedback_even_same_words_is_protected(self):
        baseline = core.load_project(self.root)
        for action, kind in (("human_rejected", "shot"), ("human_rejected_prompt", None),
                             ("human_planning_feedback", None)):
            with self.subTest(action=action):
                with core.transaction(self.root) as state:
                    state.clear(); state.update(copy.deepcopy(baseline))
                    core.event(state, action, kind=kind, id=self.shot_id, version=0, feedback=self.machine)
                original = self.shot()
                self.assertFalse(self.resolve())
                self.assertEqual(self.shot(), original)

    def test_fresh_same_asset_rejection_and_any_video_attempt_are_protected(self):
        baseline = core.load_project(self.root)
        for label in ("fresh_asset", "version", "file", "job", "historic_job", "attempt", "prompt_approval"):
            with self.subTest(label=label):
                with core.transaction(self.root) as state:
                    state.clear(); state.update(copy.deepcopy(baseline))
                    shot = core._find(state, "shot", self.shot_id)
                    if label == "fresh_asset":
                        core.event(state, "human_rejected", kind="asset", id="voice", version=3, feedback=self.reason)
                    elif label == "historic_job":
                        state["jobs"]["synthetic"] = {"kind": "video", "item_id": self.shot_id,
                            "accounting": "spent", "state": "completed", "amount": 0}
                    elif label == "attempt":
                        shot["attempts"] = [{"revision": 0}]
                    else:
                        shot[{"version": "version", "file": "file", "job": "job_id", "prompt_approval": "prompt_approval"}[label]] = 1
                original = self.shot()
                self.assertFalse(self.resolve())
                self.assertEqual(self.shot(), original)


if __name__ == "__main__":
    unittest.main()
