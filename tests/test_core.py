import copy
from pathlib import Path
import tempfile
import unittest

from norlan import core, pipeline


class CoreTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        core.create_project(self.root, "原创小说")
        core.install_plan(self.root, pipeline.demo_plan())

    def artifact(self, relative, body=b"media"):
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(body)
        return {"file": relative, "sha256": core.digest(path)}

    def test_transaction_rolls_back_invalid_mutations(self):
        with self.assertRaises(ValueError):
            core.configure(self.root, {"caps": {"image": -1}}, limit=100)
        state = core.load_project(self.root)
        self.assertEqual(state["settings"]["caps"]["image"], 0)
        self.assertEqual(state["budget"]["limit"], 0)

    def test_new_project_preserves_unrelated_existing_files(self):
        other = self.root / "unrelated"
        other.mkdir()
        original = other / "source"
        original.mkdir()
        novel = original / "novel.txt"
        novel.write_text("original", encoding="utf-8")
        with self.assertRaises(ValueError):
            core.create_project(other, "replacement", "demo")
        self.assertEqual(novel.read_text(encoding="utf-8"), "original")

    def test_budget_counts_unknown_reserved_spent_and_duplicate_updates(self):
        core.configure(self.root, limit=10)
        core.reserve(self.root, "one", "video", "E001-S001", 3)
        core.update_job(self.root, "one", state="submitted", task_id="remote-1")
        core.update_job(self.root, "one", state="completed", accounting="spent")
        core.update_job(self.root, "one", state="completed", accounting="spent")
        core.reserve(self.root, "two", "video", "E001-S002", 4)
        core.update_job(self.root, "two", state="uncertain", accounting="unknown")
        state = core.load_project(self.root)
        self.assertEqual(state["budget"]["spent"], 3)
        self.assertEqual(state["budget"]["unknown"], 4)
        self.assertEqual(state["budget"]["available"], 3)
        with self.assertRaises(core.GateError):
            core.reserve(self.root, "three", "video", "E002-S001", 1)
        with self.assertRaises(core.GateError):
            core.reserve(self.root, "one", "video", "E001-S001", 3)
        with self.assertRaises(ValueError):
            core.update_job(self.root, "one", amount=0)
        with self.assertRaises(core.GateError):
            core.update_job(self.root, "one", task_id="different-id")
        with self.assertRaises(core.GateError):
            core.update_job(self.root, "one", accounting="released")
        core.reconcile(self.root, "two", "resume", "控制台查到任务", task_id="remote-2")
        state = core.load_project(self.root)
        self.assertEqual(state["budget"]["reserved"], 4)
        self.assertEqual(core._find(state, "shot", "E001-S002")["job_id"], "two")
        with self.assertRaises(core.GateError):
            core.reserve(self.root, "too-much", "video", "E002-S001", 4)

    def test_version_and_file_hash_approval_gates(self):
        with self.assertRaises(core.GateError):
            core.approve(self.root, "plan", "plan", True)
        with self.assertRaises(core.GateError):
            core.approve(self.root, "plan", "plan", 0)
        core.approve(self.root, "plan", "plan", 1)
        record = self.artifact("media/hero.png")
        with core.transaction(self.root) as state:
            state["assets"][0].update(record, status="candidate", version=1)
        (self.root / record["file"]).write_bytes(b"changed")
        with self.assertRaises(core.GateError):
            core.approve(self.root, "asset", "hero", 1)

    def test_voice_audio_hash_is_separate_from_preview_video(self):
        record = self.artifact("media/voice.mp4")
        audio = self.artifact("media/voice.wav", b"approved voice")
        with core.transaction(self.root) as state:
            state["assets"].append({"id": "voice", "kind": "voice", "name": "试音", "prompt": "声音",
                                    "character_id": "hero",
                                    **record, "audio_file": audio["file"], "audio_sha256": audio["sha256"],
                                    "status": "candidate", "version": 1})
        (self.root / audio["file"]).write_bytes(b"different voice")
        with self.assertRaises(core.GateError):
            core.approve(self.root, "asset", "voice", 1)

    def test_episode_rejection_preserves_unselected_video_and_rechecks_continuity(self):
        with core.transaction(self.root) as state:
            ep = state["episodes"][0]
            third = copy.deepcopy(ep["shots"][0])
            fourth = copy.deepcopy(third)
            third.update(id="E001-S003", scene_id="street")
            fourth.update(id="E001-S004")
            ep["shots"].extend([third, fourth])
            ep.update(status="awaiting_review", version=1, verified_subtitles=[{"start": 0, "end": 1, "text": "旧字幕"}])
            for shot in ep["shots"]:
                shot.update(self.artifact(f"media/{shot['id']}.mp4"), status="accepted", version=1)
        core.reject(self.root, "episode", "E001", 1, "第二镜服装改回蓝色", ["E001-S002"])
        ep = core.load_project(self.root)["episodes"][0]
        self.assertEqual([s["status"] for s in ep["shots"]], ["accepted", "planned", "accepted", "review_required"])
        self.assertEqual(ep["shots"][3]["version"], 1)
        self.assertIn("file", ep["shots"][3])
        self.assertEqual(ep["shots"][3]["continuity_review"]["source_shots"][0]["id"], "E001-S002")
        self.assertNotIn("verified_subtitles", ep)
        self.assertEqual(ep["subtitle_status"], "needs_retiming")

    def test_reject_during_run_does_not_mutate_state(self):
        with core.run_lock(self.root):
            with self.assertRaises(core.GateError):
                core.reject(self.root, "plan", "plan", 1, "修改")
        self.assertEqual(core.load_project(self.root)["plan"]["status"], "candidate")

    def test_resume_music_reconciliation(self):
        core.configure(self.root, limit=1)
        core.reserve(self.root, "music-one", "music", "E001", .5)
        core.update_job(self.root, "music-one", state="uncertain", accounting="unknown")
        core.reconcile(self.root, "music-one", "resume", "已查到配乐任务", task_id="music-remote")
        state = core.load_project(self.root)
        self.assertEqual(state["episodes"][0]["music_job_id"], "music-one")
        self.assertEqual(state["jobs"]["music-one"]["task_id"], "music-remote")

    def test_voice_plan_mapping_and_legacy_nonvoice_compatibility(self):
        plan = pipeline.demo_plan()
        core.validate_plan(plan)
        plan["assets"].append({"id": "voice", "kind": "voice", "name": "声音", "prompt": "声音说明"})
        with self.assertRaises(ValueError):
            core.validate_plan(plan)
        plan["assets"][-1]["character_id"] = "station"
        with self.assertRaises(ValueError):
            core.validate_plan(plan)
        plan["assets"][-1]["character_id"] = "hero"
        core.validate_plan(plan)


if __name__ == "__main__":
    unittest.main()
