"""Synthetic local asset-input error retention; no real services or media."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from norlan import asset_inputs, core, dashboard, pipeline


class AssetInputDiagnosticsTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        core.create_project(self.root, "SYNTHETIC：用于核对资产输入错误保留。", mode="live")
        core.install_plan(self.root, pipeline.demo_plan())
        core.approve(self.root, "plan", "plan", 1)
        self.message = "合成图片模型配置尚未完成；请先核对本机设置。"
        with core.transaction(self.root) as state:
            state.update(status="blocked", last_error=self.message)

    def assert_blocker_retained(self, state):
        self.assertEqual(state["status"], "blocked")
        self.assertEqual(state["last_error"], self.message)
        stop = dashboard.public_state(state)["production_stop"]
        self.assertEqual(stop["stage"], "assets")
        self.assertEqual(stop["message"], self.message)
        self.assertEqual(state["jobs"], {})
        self.assertTrue(all(asset["version"] == 0 for asset in state["assets"]))

    def test_single_input_confirmation_preserves_current_blocked_reason(self):
        state = core.set_asset_input(self.root, "hero", 0, 0, "gpt", "")
        self.assert_blocker_retained(state)
        self.assertTrue(asset_inputs.confirmed(core._find(state, "asset", "hero")))
        self.assertFalse(asset_inputs.confirmed(core._find(state, "asset", "station")))

    def test_bulk_input_confirmation_preserves_current_blocked_reason(self):
        state = core.set_remaining_asset_inputs_gpt(self.root, 1)
        self.assert_blocker_retained(state)
        self.assertTrue(all(asset_inputs.confirmed(asset) for asset in state["assets"]))
        again = core.set_remaining_asset_inputs_gpt(self.root, 1)
        self.assert_blocker_retained(again)
        self.assertEqual(again["assets"], state["assets"])
        self.assertEqual(again["events"], state["events"])

    def test_nonblocked_input_confirmation_keeps_existing_clear_behavior(self):
        for bulk in (False, True):
            with self.subTest(bulk=bulk):
                with core.transaction(self.root) as state:
                    state.update(status="ready", last_error=self.message)
                if bulk:
                    state = core.set_remaining_asset_inputs_gpt(self.root, 1)
                else:
                    state = core.set_asset_input(self.root, "hero", 0, 0, "gpt", "")
                self.assertEqual(state["status"], "ready")
                self.assertIsNone(state["last_error"])
                self.assertEqual(state["jobs"], {})

    def test_successful_prompt_preparation_clears_error_only_after_actual_advance(self):
        state = core.set_remaining_asset_inputs_gpt(self.root, 1)
        self.assert_blocker_retained(state)
        with patch.object(pipeline, "_ark", side_effect=AssertionError("No provider access before consent")), \
                patch.object(pipeline, "_brain", side_effect=AssertionError("No model call for input choices")):
            state = pipeline.run(self.root)
        self.assertEqual(state["status"], "awaiting_asset_prompt_approval")
        self.assertIsNone(state["last_error"])
        self.assertNotIn("production_stop", dashboard.public_state(state))
        self.assertEqual(state["jobs"], {})
        for asset in state["assets"]:
            self.assertEqual(asset["version"], 0)
            self.assertEqual(asset["status"], "awaiting_asset_prompt_approval")
            self.assertEqual(asset["generation_prompt"]["status"], "candidate")
            self.assertNotIn("generation_prompt_approval", asset)


if __name__ == "__main__":
    unittest.main()
