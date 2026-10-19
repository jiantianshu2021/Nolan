"""Synthetic internal-shot planning checks; never contact generation engines."""
import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from norlan import core, mcp_server, pipeline
from norlan.segments import MAX_SUBSHOT_PROMPT_CHARS, timeline_prompt, validate_subshots


def segment(duration=10):
    return {"duration": duration, "subshots": [{"start": 0, "end": 4.5, "prompt": "中景，角色转身，完整说出：我现在就去。"},
                                              {"start": 4.5, "end": duration, "prompt": "近景，另一个角色回应，环境声连续。"}]}


class SegmentValidationTests(unittest.TestCase):
    def test_legacy_is_unchanged_and_valid_timeline_is_pure(self):
        self.assertIsNone(validate_subshots({"duration": 4}))
        self.assertEqual(timeline_prompt({"duration": 4}), "")
        value = segment()
        before = copy.deepcopy(value)
        self.assertIsNone(validate_subshots(value))
        prompt = timeline_prompt(value)
        self.assertEqual(value, before)
        self.assertIn("0–4.5秒", prompt)
        self.assertIn("4.5–10秒", prompt)
        self.assertIn("生成目标", prompt)
        self.assertIn("不代表模型能精确", prompt)
        self.assertIn("完整、自然说完", prompt)
        self.assertIn("不生成背景配乐", prompt)
        self.assertIn("真实人物/场景及声音参考", prompt)
        self.assertIsNone(validate_subshots({"duration": 4.0, "subshots": [{"start": 0, "end": 4, "prompt": "连续长镜头"}]}))

    def test_rejects_empty_extra_fields_and_invalid_text(self):
        candidates = [None, [], {}, segment()["subshots"] * 2,
                      [{"start": 0, "end": 10, "prompt": "ok", "file": "secret"}],
                      [{"start": 0, "end": 10}], [{"start": 0, "end": 10, "prompt": " "}],
                      [{"start": 0, "end": 10, "prompt": "x" * (MAX_SUBSHOT_PROMPT_CHARS + 1)}],
                      [{"start": 0, "end": 10, "prompt": "x\x00y"}]]
        for subshots in candidates:
            with self.subTest(subshots=str(subshots)[:50]), self.assertRaises(ValueError):
                validate_subshots({"duration": 10, "subshots": subshots})

    def test_rejects_nonfinite_bool_gap_overlap_reversal_and_uncovered_duration(self):
        cases = [(-1, 4.5, 4.5, 10), (0, 4.5, 4.500000001, 10), (0, 4.5, 4.4, 10),
                 (0, 0, 0, 10), (0, 4.5, 4.5, 9.99), (0, 4.5, 4.5, 10.01),
                 (False, 4.5, 4.5, 10), (0, True, 1, 10), (0, float("nan"), 4.5, 10),
                 (0, 4.5, 4.5, float("inf")), (0, 4.5, "4.5", 10)]
        for first_start, first_end, second_start, second_end in cases:
            value = segment()
            value["subshots"][0].update(start=first_start, end=first_end)
            value["subshots"][1].update(start=second_start, end=second_end)
            with self.subTest(case=str(cases)), self.assertRaises(ValueError):
                validate_subshots(value)
        for duration in (True, 3, 16, 10.5, float("nan"), float("inf"), 10 ** 1000):
            with self.subTest(duration=str(duration)[:30]), self.assertRaises(ValueError):
                validate_subshots(segment(duration))

    def test_core_reports_fixed_generation_segment_diagnostic_and_keeps_reference_gates(self):
        plan = pipeline.demo_plan()
        shot = plan["episodes"][0]["shots"][0]
        shot.update(segment())
        core.validate_plan(plan)
        shot["subshots"][1]["start"] = 5
        with self.assertRaises(core.PlanValidationError) as raised:
            core.validate_plan(plan)
        self.assertEqual(raised.exception.code, "subshots_invalid")
        self.assertEqual(raised.exception.path, "plan.episodes[0].shots[0].subshots")
        self.assertIn("生成片段", str(raised.exception))
        shot.update(segment())
        shot["asset_ids"] = ["missing"]
        with self.assertRaises(core.PlanValidationError) as raised:
            core.validate_plan(plan)
        self.assertEqual(raised.exception.code, "asset_references_invalid")
        plan["assets"].append({"id": "voice", "name": "声线", "kind": "voice", "prompt": "中年男声", "character_id": "hero"})
        shot["asset_ids"] = ["station", "voice"]
        with self.assertRaises(core.PlanValidationError) as raised:
            core.validate_plan(plan)
        self.assertEqual(raised.exception.code, "voice_character_not_referenced")


class SegmentMCPTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.workspace = Path(self.temp.name).resolve() / "workspace"
        last = patch.object(mcp_server, "_last_project", return_value=None)
        last.start()
        self.addCleanup(last.stop)
        self.server = mcp_server.MCPServer(self.workspace)
        self.addCleanup(self.server.close)
        self.root = self.server.projects / "segment-test"
        self.server.call_tool("create_project", {"project": str(self.root), "novel_text": "合成分段测试。", "mode": "demo"})

    def test_mcp_persists_and_returns_optional_subshots_without_starting_tasks(self):
        plan = pipeline.demo_plan()
        plan["episodes"][0]["shots"][0].update(segment())
        with patch.object(pipeline, "run", side_effect=AssertionError("must not generate")):
            result = self.server.call_tool("save_plan", {"project": str(self.root), "plan": plan})
        self.assertEqual(result["episodes"][0]["shots"][0]["subshots"], plan["episodes"][0]["shots"][0]["subshots"])
        self.assertEqual(result["plan"]["status"], "candidate")
        self.assertEqual(core.load_project(self.root)["jobs"], {})
        self.assertNotIn("subshots", mcp_server.SHOT["required"])

    def test_mcp_number_schema_rejects_nonfinite_bool_unknown_keys_and_empty_list(self):
        for value in (True, float("nan"), float("inf"), -1, 16):
            with self.assertRaises(mcp_server.ArgumentValidationError):
                mcp_server._validate(value, mcp_server.SUBSHOT["properties"]["start"])
        mcp_server._validate(4.5, mcp_server.SUBSHOT["properties"]["start"])
        for value in ([], [{"start": 0, "end": 4, "prompt": "测试", "approved": True}]):
            plan = pipeline.demo_plan()
            plan["episodes"][0]["shots"][0]["subshots"] = value
            before = (self.root / "project.json").read_bytes()
            with self.assertRaises(mcp_server.ArgumentValidationError):
                self.server.call_tool("save_plan", {"project": str(self.root), "plan": plan})
            self.assertEqual((self.root / "project.json").read_bytes(), before)

    def test_existing_running_plan_cannot_be_replaced_by_subshot_plan(self):
        plan = pipeline.demo_plan()
        self.server.call_tool("save_plan", {"project": str(self.root), "plan": plan})
        plan["episodes"][0]["shots"][0].update(segment())
        with core.run_lock(self.root), self.assertRaises(core.GateError):
            self.server.call_tool("save_plan", {"project": str(self.root), "plan": plan})
        core.reserve(self.root, "existing-call", "video", "E001-S001", 0)
        core.update_job(self.root, "existing-call", state="submitted", task_id="synthetic-existing")
        before = (self.root / "project.json").read_bytes()
        with self.assertRaises(core.GateError):
            core.install_plan(self.root, plan)
        self.assertEqual((self.root / "project.json").read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
