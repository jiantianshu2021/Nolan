"""Offline transport outcomes retain unknown accounting until reconciled."""
import copy
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from norlan import core, pipeline
from norlan.brain import BrainError, _codex_failure


class BrainTransportUncertaintyTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        core.create_project(self.root, "合成传输边界测试。", "live")
        core.install_plan(self.root, pipeline.demo_plan())
        core.configure(self.root, {"caps": {"llm": 2}}, limit=20)
        self.shot = copy.deepcopy(core.load_project(self.root)["episodes"][0]["shots"][0])
        self.config = {"provider": "codex", "billing": "subscription", "model": "synthetic-model"}
        self.enterContext(patch("norlan.connection.production_brain_config", side_effect=lambda _config: dict(self.config)))
        self.enterContext(patch("norlan.brain.resolve_codex", return_value="/synthetic/codex"))

    def assert_unknown_blocks_repeat(self, *, returncode, stderr="", stdout="", output_text=None):
        def invoke(args, **_kwargs):
            reserved = core.load_project(self.root)
            self.assertEqual(reserved["llm_calls"], 1)
            self.assertEqual(reserved["budget"]["reserved"], 2)
            job = next(iter(reserved["jobs"].values()))
            self.assertEqual((job["state"], job["accounting"]), ("submitting", "reserved"))
            if output_text is not None:
                Path(args[args.index("--output-last-message") + 1]).write_text(output_text)
            return subprocess.CompletedProcess(args, returncode, stdout, stderr)

        with patch("norlan.brain.subprocess.run", side_effect=invoke) as transport:
            with self.assertRaises(BrainError) as raised:
                pipeline._brain(self.root, "refine_shot", self.shot, {}, [])
            self.assertTrue(raised.exception.uncertain)
            first = core.load_project(self.root)
            self.assertEqual(len(first["jobs"]), 1)
            job = next(iter(first["jobs"].values()))
            self.assertEqual((job["state"], job["accounting"]), ("uncertain", "unknown"))
            self.assertTrue(job["failure"]["uncertain"])
            self.assertEqual(first["budget"]["unknown"], 2)
            self.assertEqual(first["budget"]["spent"], 0)
            self.assertEqual(first["budget"]["reserved"], 0)
            self.assertEqual(len(core.blocking_unknown_jobs(first)), 1)
            with self.assertRaisesRegex(core.GateError, "结果不明"):
                pipeline._brain(self.root, "refine_shot", self.shot, {}, [])
            self.assertEqual(transport.call_count, 1)
            after = core.load_project(self.root)
            for field in ("jobs", "llm_calls", "budget", "events"):
                self.assertEqual(after[field], first[field])

    def test_process_signal_exit_is_unknown_even_with_auth_warning(self):
        self.assert_unknown_blocks_repeat(returncode=-15, stderr="earlier warning: refresh token expired")

    def test_stream_loss_overrides_an_earlier_auth_warning(self):
        self.assert_unknown_blocks_repeat(returncode=1,
            stderr="old auth warning: token has expired\nstream disconnected before completion")

    def test_network_failure_in_stdout_overrides_auth_text_in_stderr(self):
        self.assert_unknown_blocks_repeat(returncode=1, stderr="earlier warning: unauthorized",
                                          stdout="connection reset while reading response")

    def test_generic_unclassified_exit_is_unknown(self):
        self.assert_unknown_blocks_repeat(returncode=1, stderr="synthetic unexpected worker exit")

    def test_explicit_auth_model_quota_rejections_are_definite(self):
        for text in ("authentication required; please log in", "model synthetic-model is not supported",
                     "quota exceeded", "rate limit reached"):
            with self.subTest(text=text):
                error = _codex_failure(1, text)
                self.assertFalse(error.uncertain)
                self.assertFalse(error.failure["uncertain"])

    def test_success_exit_without_structured_output_is_unknown(self):
        self.assert_unknown_blocks_repeat(returncode=0)

    def test_success_exit_with_unparseable_output_is_unknown(self):
        self.assert_unknown_blocks_repeat(returncode=0, output_text="not a JSON model result")

    def test_reservation_refusal_prevents_both_cli_and_api_transports(self):
        core.configure(self.root, limit=1)
        before = core.load_project(self.root)
        reserve = core.reserve
        with patch.dict(os.environ, {"OPENAI_API_KEY": "SYNTHETIC-LOCAL-TEST-KEY"}), \
                patch("norlan.brain.subprocess.run", side_effect=AssertionError("No CLI dispatch")) as cli, \
                patch("norlan.brain._api_open", side_effect=AssertionError("No API POST")) as api:
            for provider, billing in (("codex", "subscription"), ("openai", "api")):
                with self.subTest(provider=provider):
                    self.config.update(provider=provider, billing=billing)
                    with patch("norlan.pipeline.core.reserve", wraps=reserve) as reservation:
                        with self.assertRaisesRegex(core.GateError, "预算不足"):
                            pipeline._brain(self.root, "refine_shot", self.shot, {}, [])
                        reservation.assert_called_once()
                    after = core.load_project(self.root)
                    for field in ("jobs", "llm_calls", "budget", "events"):
                        self.assertEqual(after[field], before[field])
            cli.assert_not_called()
            api.assert_not_called()


if __name__ == "__main__":
    unittest.main()
