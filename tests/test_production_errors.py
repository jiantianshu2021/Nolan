"""Synthetic bounded production-error persistence; no services or credentials."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from norlan import core, pipeline, production_errors


class ProductionErrorTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        core.create_project(self.root, "SYNTHETIC：仅测试安全诊断存储。")
        core.install_plan(self.root, pipeline.demo_plan())
        core.approve(self.root, "plan", "plan", 1)

    def test_message_is_bounded_and_credentials_urls_are_redacted(self):
        key = "SYNTHETIC_PRIVATE_CREDENTIAL_123456789"
        message = f"合成请求错误 Bearer {key} https://private.example.test/task?secret=x " + "长" * 2000
        with patch.dict("os.environ", {"SYNTHETIC_API_KEY": key}):
            production_errors.record(self.root, message)
        state = core.load_project(self.root)
        self.assertLessEqual(len(state["last_error"]), 801)
        self.assertEqual(state["production_failure"]["message"], state["last_error"])
        stored = json.dumps({name: state[name] for name in
            ("last_error", "production_failure", "production_failure_history")}, ensure_ascii=False)
        self.assertNotIn(key, stored)
        self.assertNotIn("https://", stored)
        self.assertIn("合成请求错误", stored)
        self.assertEqual(set(state["production_failure"]),
            {"schema", "time", "message", "stage", "asset_id", "asset_version"})

    def test_active_asset_binding_precedes_pending_and_approved_prompt(self):
        with core.transaction(self.root) as state:
            state["assets"][0].update(status="asset_prompt_approved", version=0)
            state["assets"][1].update(status="generating", version=2)
        production_errors.record(self.root, "合成本机校验失败")
        state = core.load_project(self.root)
        failure = state["production_failure"]
        self.assertEqual((failure["stage"], failure["asset_id"], failure["asset_version"]),
                         ("assets", "station", 2))
        self.assertEqual(state["jobs"], {})
        with core.transaction(self.root) as state:
            state["assets"][1]["status"] = "planned"
        production_errors.record(self.root, "合成待提交检查失败")
        self.assertEqual(core.load_project(self.root)["production_failure"]["asset_id"], "hero")

    def test_history_is_bounded_and_repeated_continue_does_not_grow_it(self):
        for index in range(25):
            production_errors.record(self.root, f"合成错误 {index}")
        state = core.load_project(self.root)
        self.assertEqual(len(state["production_failure_history"]), 20)
        self.assertEqual(state["production_failure_history"][0]["message"], "合成错误 5")
        for _ in range(30):
            production_errors.record(self.root, "合成错误 24")
        state = core.load_project(self.root)
        self.assertEqual(len(state["production_failure_history"]), 20)
        self.assertEqual(sum(item["message"] == "合成错误 24"
                             for item in state["production_failure_history"]), 1)
        with core.transaction(self.root) as state:
            state["assets"][0]["version"] = 1
        production_errors.record(self.root, "合成错误 24")
        state = core.load_project(self.root)
        self.assertEqual(len(state["production_failure_history"]), 20)
        self.assertEqual([item["asset_version"] for item in state["production_failure_history"][-2:]], [0, 1])

    def test_durable_failure_survives_transient_last_error_clear(self):
        production_errors.record(self.root, "合成图片素材准备失败", status="blocked")
        before = core.load_project(self.root)["production_failure"]
        with core.transaction(self.root) as state:
            state["last_error"] = None
        state = core.load_project(self.root)
        self.assertEqual(state["production_failure"], before)
        self.assertEqual(state["production_failure_history"], [before])
        self.assertIsNone(state["last_error"])

    def test_history_reprojection_drops_unapproved_fields_and_invalid_entries(self):
        production_errors.record(self.root, "合成历史错误")
        with core.transaction(self.root) as state:
            state["production_failure_history"][0].update(trace="SYNTHETIC TRACE", prompt="SYNTHETIC PROMPT",
                                                          config={"secret": "SYNTHETIC PRIVATE"})
            state["production_failure_history"].append({"schema": 1, "time": "unsafe", "message": "坏记录"})
        production_errors.record(self.root, "合成新错误")
        state = core.load_project(self.root)
        history = state["production_failure_history"]
        self.assertEqual(len(history), 2)
        encoded = json.dumps(history)
        for value in ("trace", "prompt", "config", "unsafe", "SYNTHETIC PRIVATE"):
            self.assertNotIn(value, encoded)

    def test_planning_and_current_episode_phase_inference(self):
        with core.transaction(self.root) as state:
            state["plan"]["status"] = "pending"
        production_errors.record(self.root, "合成策划错误")
        self.assertEqual(core.load_project(self.root)["production_failure"]["stage"], "planning")
        with core.transaction(self.root) as state:
            state["plan"]["status"] = "approved"
            for asset in state["assets"]:
                asset["status"] = "approved"
        production_errors.record(self.root, "合成提示词错误")
        self.assertEqual(core.load_project(self.root)["production_failure"]["stage"], "shot_preparation")
        with core.transaction(self.root) as state:
            state["episodes"][0]["shots"][0]["status"] = "generating"
        production_errors.record(self.root, "合成生成错误", status="paused")
        state = core.load_project(self.root)
        self.assertEqual(state["production_failure"]["stage"], "shot_generation")
        self.assertEqual(state["status"], "paused")


if __name__ == "__main__":
    unittest.main()
