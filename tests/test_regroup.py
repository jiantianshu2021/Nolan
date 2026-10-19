"""Synthetic paused projects for regrouping; all APIs stay offline."""
import ast
import copy
import json
from pathlib import Path
import tempfile
import unittest

from norlan import core, pipeline, regroup


class RegroupTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        core.create_project(self.root, "合成连续片段测试；原台词保持不变。")
        plan = pipeline.demo_plan()
        plan["episodes"] = plan["episodes"][:1]
        original = plan["episodes"][0]["shots"][0]
        plan["episodes"][0]["shots"] = [{**copy.deepcopy(original), "id": f"S{index}", "duration": 5,
            "prompt": f"00:{(index-1)*5:02}–00:{index*5:02}。角色站在窗边，说原句。", "dialogue": "等等。"}
            for index in range(1, 7)]
        core.install_plan(self.root, plan)
        core.approve(self.root, "plan", "plan", 1)
        core.configure(self.root, limit=100)
        core.reserve(self.root, "paid-one", "video", "S1", 3)
        core.update_job(self.root, "paid-one", state="completed", accounting="spent", task_id="synthetic-one")
        core.reserve(self.root, "failed-two", "video", "S2", 2)
        core.update_job(self.root, "failed-two", state="failed", accounting="spent", task_id="synthetic-two")
        with core.transaction(self.root) as state:
            episode = state["episodes"][0]
            episode["shots"][0].update(status="accepted", version=1, file="media/old.mp4", sha256="a" * 64,
                                       job_id="paid-one", history=[{"version": 0, "note": "keep"}])
            episode["shots"][1].update(version=2, feedback="保留之前的返工要求")
            episode["progress_stage"] = {"stage": "shot_generation", "shot_id": "S2"}
            episode["progress_started_at"] = "2026-01-01T00:00:00+00:00"
        core.set_paused(self.root, True)
        self.groups = [["S3", "S4"], ["S5"], ["S6"]]

    def propose(self, groups=None):
        return regroup.propose_segments(self.root, "E001", groups or self.groups)

    def proposal(self):
        return core.load_project(self.root)["episodes"][0]["segment_proposal"]

    def restore(self, state):
        with core.transaction(self.root) as current:
            current.clear()
            current.update(copy.deepcopy(state))

    def test_proposal_is_inert_and_preserves_repeated_dialogue_and_local_timing(self):
        before = core.load_project(self.root)
        state = self.propose()
        proposal = state["episodes"][0]["segment_proposal"]
        self.assertEqual(proposal["source_ids"], ["S3", "S4", "S5", "S6"])
        self.assertEqual(proposal["status"], "candidate")
        merged = proposal["shots"][0]
        self.assertEqual(merged["duration"], 10)
        self.assertEqual(merged["source_shot_ids"], ["S3", "S4"])
        self.assertEqual(merged["dialogue"], "等等。\n等等。", "intentional original repetition is not an editing permission")
        self.assertEqual([(sub["start"], sub["end"]) for sub in merged["subshots"]], [(0, 5), (5, 10)])
        for sub in merged["subshots"]:
            self.assertIn("等等。", sub["prompt"])
            self.assertIn("仅说一次", sub["prompt"])
            self.assertNotIn("00:", sub["prompt"])
        self.assertNotIn("00:", merged["prompt"])
        self.assertEqual(state["episodes"][0]["shots"], before["episodes"][0]["shots"])
        for key in ("assets", "jobs", "budget", "paused", "plan"):
            self.assertEqual(state[key], before[key])
        self.assertEqual(len(list((self.root / "plans").glob("segments-*-before.json"))), 0)

    def test_human_approval_only_replaces_pristine_tail_and_archives_original(self):
        before = core.load_project(self.root)
        self.propose()
        proposal = self.proposal()
        state = regroup.decide_segments(self.root, "E001", proposal["id"], True)
        episode = state["episodes"][0]
        self.assertEqual(episode["shots"][:2], before["episodes"][0]["shots"][:2])
        self.assertEqual(episode["shots"][2:], proposal["shots"])
        self.assertEqual(episode["shots"][-2:], before["episodes"][0]["shots"][-2:])
        for key in ("assets", "jobs", "budget", "paused", "plan"):
            self.assertEqual(state[key], before[key])
        self.assertNotIn("progress_stage", episode)
        self.assertNotIn("progress_started_at", episode)
        record = episode["segment_plan_history"][0]["snapshot"]
        path = self.root / record["file"]
        self.assertEqual(core.digest(path), record["sha256"])
        snapshot = json.loads(path.read_text())
        self.assertEqual(snapshot["episodes"][0]["shots"], before["episodes"][0]["shots"])
        self.assertIn("00:10–00:15", snapshot["episodes"][0]["shots"][2]["prompt"])
        with self.assertRaises(core.GateError):
            regroup.decide_segments(self.root, "E001", proposal["id"], True)

    def test_cancel_preserves_plan_and_allows_later_new_proposal(self):
        before = core.load_project(self.root)
        self.propose()
        old_id = self.proposal()["id"]
        state = regroup.decide_segments(self.root, "E001", old_id, False)
        self.assertEqual(state["episodes"][0]["shots"], before["episodes"][0]["shots"])
        self.assertEqual(state["episodes"][0]["segment_proposal"]["status"], "rejected")
        self.propose([["S3", "S4", "S5"], ["S6"]])
        self.assertNotEqual(self.proposal()["id"], old_id)
        self.assertEqual(core.load_project(self.root)["episodes"][0]["segment_proposal_history"][0]["id"], old_id)

    def test_pause_approval_run_lock_and_inflight_gates(self):
        initial = core.load_project(self.root)
        core.set_paused(self.root, False)
        with self.assertRaises(core.GateError):
            self.propose()
        self.restore(initial)
        with core.run_lock(self.root), self.assertRaises(core.GateError):
            self.propose()
        for state, accounting in (("submitted", "reserved"), ("download_pending", "spent"), ("uncertain", "unknown")):
            self.restore(initial)
            with core.transaction(self.root) as current:
                current["jobs"]["paid-one"].update(state=state, accounting=accounting)
            with self.assertRaises(core.GateError):
                self.propose()
        self.restore(initial)
        with core.transaction(self.root) as current:
            current["plan"]["status"] = "candidate"
        with self.assertRaises(core.GateError):
            self.propose()
        self.restore(initial)
        self.propose()
        # A manual resume without an actual worker must not trap the decision.
        core.set_paused(self.root, False)
        regroup.decide_segments(self.root, "E001", self.proposal()["id"], False)

    def test_original_order_coverage_and_actual_change_are_required(self):
        for groups in ([["S3", "S4"]], [["S4", "S3"], ["S5"], ["S6"]],
                       [["S3", "S4"], ["S4", "S5"], ["S6"]], [["S2", "S3"], ["S4", "S5"], ["S6"]],
                       [["S3"], ["S4"], ["S5"], ["S6"]], [["S3", "S4", "S5", "S6"]]):
            before = (self.root / "project.json").read_bytes()
            with self.subTest(groups=groups), self.assertRaises(core.GateError):
                self.propose(groups)
            self.assertEqual((self.root / "project.json").read_bytes(), before)

    def test_any_prior_attempt_or_prepared_result_locks_the_prefix(self):
        initial = core.load_project(self.root)
        variants = [{"version": 1}, {"refined_prompt": {}},
                    {"evidence": {}}, {"history": [{"file": "old"}]}, {"attempts": [{"job_id": "old"}]},
                    {"file": "media/old.mp4"}, {"status": "review_required"}]
        for fields in variants:
            self.restore(initial)
            with core.transaction(self.root) as current:
                current["episodes"][0]["shots"][3].update(fields)
            with self.assertRaises(core.GateError):
                self.propose()
            state = self.propose([["S5", "S6"]])
            self.assertEqual(state["episodes"][0]["segment_proposal"]["source_ids"], ["S5", "S6"])
        self.restore(initial)
        with core.transaction(self.root) as current:
            current["jobs"]["previous-s4"] = {"id": "previous-s4", "kind": "video", "item_id": "S4", "amount": 0,
                                              "accounting": "released", "state": "failed"}
        self.assertEqual(self.propose([["S5", "S6"]])["episodes"][0]["segment_proposal"]["source_ids"], ["S5", "S6"])

    def test_unstarted_inherited_revisions_keep_scoped_feedback_and_audit_metadata(self):
        first_feedback = "资产变更：角色声线改为稳定清晰；保留原句。"
        second_feedback = "这一个原片段保持人物右侧站位。"
        with core.transaction(self.root) as state:
            first, second = state["episodes"][0]["shots"][2:4]
            first.update(revision=2, feedback=first_feedback, subshots=[
                {"start": 0, "end": 2, "prompt": "左侧中景，说第一句。"},
                {"start": 2, "end": 5, "prompt": "同场景近景，说第二句。"}])
            second.update(revision=3, feedback=second_feedback)
        state = self.propose()
        candidate = state["episodes"][0]["segment_proposal"]["shots"][0]
        self.assertEqual(candidate["source_shot_instructions"], [
            {"id": "S3", "revision": 2, "feedback": first_feedback},
            {"id": "S4", "revision": 3, "feedback": second_feedback}])
        self.assertEqual(candidate["feedback"], "S3：" + first_feedback + "\nS4：" + second_feedback)
        self.assertIn(first_feedback, candidate["subshots"][0]["prompt"])
        self.assertIn("0–5 秒", candidate["subshots"][0]["prompt"])
        self.assertNotIn(first_feedback, candidate["subshots"][1]["prompt"])
        self.assertIn(second_feedback, candidate["subshots"][2]["prompt"])
        self.assertIn("5–10 秒", candidate["subshots"][2]["prompt"])
        self.assertIn("等等。", candidate["subshots"][2]["prompt"])
        regroup.decide_segments(self.root, "E001", self.proposal()["id"], True)
        applied = core.load_project(self.root)["episodes"][0]["shots"][2]
        self.assertEqual(applied["source_shot_instructions"], candidate["source_shot_instructions"])

    def test_overlong_feedback_is_rejected_without_truncating_original(self):
        with core.transaction(self.root) as state:
            state["episodes"][0]["shots"][2].update(revision=2, feedback="原反馈" * 1000)
        before = (self.root / "project.json").read_bytes()
        with self.assertRaises(core.GateError):
            self.propose()
        self.assertEqual((self.root / "project.json").read_bytes(), before)

    def test_scene_total_duration_and_flattened_internal_shot_limit(self):
        initial = core.load_project(self.root)
        for field, value in (("scene_id", "other-scene"), ("duration", 4), ("duration", 11)):
            self.restore(initial)
            with core.transaction(self.root) as current:
                current["episodes"][0]["shots"][3][field] = value
            with self.assertRaises(core.GateError):
                self.propose()
        self.restore(initial)
        with core.transaction(self.root) as current:
            for shot in current["episodes"][0]["shots"][2:4]:
                shot["subshots"] = [{"start": 0, "end": 2, "prompt": "第一内部镜头，说第一句。"},
                                    {"start": 2, "end": 5, "prompt": "第二内部镜头，说第二句。"}]
        with self.assertRaises(core.GateError):
            self.propose()
        with core.transaction(self.root) as current:
            current["episodes"][0]["shots"][3].pop("subshots")
            current["episodes"][0]["shots"][2]["dialogue"] = "整段对白不可重复附每个已有内部镜头"
            current["episodes"][0]["shots"][3]["dialogue"] = None
        proposal = self.propose()["episodes"][0]["segment_proposal"]
        merged = proposal["shots"][0]
        self.assertEqual([(row["start"], row["end"]) for row in merged["subshots"]], [(0, 2), (2, 5), (5, 10)])
        self.assertNotIn("整段对白", merged["subshots"][0]["prompt"])
        self.assertNotIn("整段对白", merged["subshots"][1]["prompt"])

    def test_reference_limits_and_voice_character_binding_are_preserved(self):
        initial = core.load_project(self.root)
        with core.transaction(self.root) as state:
            for index in range(7):
                state["assets"].append({"id": f"extra{index}", "kind": "prop", "name": "合成道具", "prompt": "道具"})
            state["episodes"][0]["shots"][2]["asset_ids"] = [asset["id"] for asset in state["assets"]]
        with self.assertRaises(core.GateError):
            self.propose()
        self.restore(initial)
        with core.transaction(self.root) as state:
            for index in range(4):
                state["assets"].append({"id": f"voice{index}", "kind": "voice", "name": "合成声音", "prompt": "中年男声", "character_id": "hero"})
            state["episodes"][0]["shots"][2]["asset_ids"] += [f"voice{index}" for index in range(4)]
        with self.assertRaises(core.GateError):
            self.propose()
        self.restore(initial)
        with core.transaction(self.root) as state:
            state["assets"].append({"id": "voice", "kind": "voice", "name": "声音", "prompt": "中年男声", "character_id": "hero"})
            for shot in state["episodes"][0]["shots"][2:4]:
                shot["asset_ids"] = ["station", "voice"]
        with self.assertRaises(core.GateError):
            self.propose()

    def test_changed_source_or_candidate_cannot_be_approved(self):
        initial = core.load_project(self.root)
        for change in ("plan", "asset", "shot", "proposal"):
            self.restore(initial)
            self.propose()
            proposal_id = self.proposal()["id"]
            with core.transaction(self.root) as state:
                if change == "plan":
                    state["plan"]["version"] += 1
                elif change == "asset":
                    state["assets"][0]["sha256"] = "new asset hash"
                elif change == "shot":
                    state["episodes"][0]["shots"][2]["prompt"] += "新增剧情"
                else:
                    state["episodes"][0]["segment_proposal"]["shots"][0]["dialogue"] = "篡改对白"
            before = (self.root / "project.json").read_bytes()
            with self.subTest(change=change), self.assertRaises(core.GateError):
                regroup.decide_segments(self.root, "E001", proposal_id, True)
            self.assertEqual((self.root / "project.json").read_bytes(), before)
            regroup.decide_segments(self.root, "E001", proposal_id, False)

    def test_existing_snapshot_is_never_overwritten_and_symlink_is_rejected(self):
        self.propose()
        proposal_id = self.proposal()["id"]
        snapshot = self.root / "plans" / (proposal_id + "-before.json")
        snapshot.write_bytes(b"keep existing archive")
        with self.assertRaises(core.GateError):
            regroup.decide_segments(self.root, "E001", proposal_id, True)
        self.assertEqual(snapshot.read_bytes(), b"keep existing archive")
        snapshot.unlink()
        target = self.root / "existing.json"
        target.write_bytes(b"keep target")
        snapshot.symlink_to(target)
        with self.assertRaises(core.GateError):
            regroup.decide_segments(self.root, "E001", proposal_id, True)
        self.assertEqual(target.read_bytes(), b"keep target")
        self.assertEqual(self.proposal()["status"], "candidate")

    def test_public_errors_are_exact_fixed_messages_without_data_echo(self):
        source = Path(regroup.__file__).read_text()
        tree = ast.parse(source)
        messages = {node.args[0].value for node in ast.walk(tree) if isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute) and node.func.attr == "GateError"
                    and node.args and isinstance(node.args[0], ast.Constant)}
        self.assertEqual(messages, regroup.PUBLIC_ERRORS)


if __name__ == "__main__":
    unittest.main()
