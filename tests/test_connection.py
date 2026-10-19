from contextlib import redirect_stdout
import io
import json
import os
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from norlan import connection, core, pipeline
from norlan.cli import main


class ConnectionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.environ = patch.dict(os.environ, {"XDG_CONFIG_HOME": str(self.root / "config"),
                                               "XDG_DATA_HOME": str(self.root / "data")})
        self.environ.start()

    def tearDown(self):
        self.environ.stop()
        self.temp.cleanup()

    def test_default_preserves_subscription_and_config_contains_no_key(self):
        self.assertEqual(connection.load_connection(), {"provider": "chatgpt"})
        connection.save_connection({"provider": "openai", "model": "chosen-model"})
        self.assertEqual(connection.load_connection(), {"provider": "openai", "model": "chosen-model"})
        self.assertEqual(connection.connection_path().stat().st_mode & 0o777, 0o600)
        with self.assertRaises(ValueError):
            connection.save_connection({"provider": "openai", "model": "chosen-model", "api_key": "not-a-key"})
        with self.assertRaises(ValueError):
            connection.save_connection({"provider": "openai"})

    def test_failed_setup_keeps_old_connection_and_key_never_in_args(self):
        connection.save_connection({"provider": "chatgpt"})
        failed = SimpleNamespace(returncode=1)
        output = io.StringIO()
        with patch("norlan.interactive.prepare_launch", return_value=(["/codex"], {})), \
                patch("norlan.connection.subprocess.run", return_value=failed) as run, \
                patch("norlan.connection.getpass.getpass", return_value="local-test-placeholder"), \
                patch("norlan.connection.sys.stdin.isatty", return_value=True), \
                patch.dict(os.environ, {"OPENAI_API_KEY": ""}), redirect_stdout(output):
            with self.assertRaises(ValueError):
                connection.setup("openai", "chosen-model")
        self.assertEqual(connection.load_connection(), {"provider": "chatgpt"})
        self.assertNotIn("local-test-placeholder", output.getvalue())
        self.assertNotIn("local-test-placeholder", repr([call.args for call in run.call_args_list]))
        self.assertEqual(run.call_args.kwargs["input"], "local-test-placeholder\n")

    def test_successful_api_setup_then_subscription_switch(self):
        ok = SimpleNamespace(returncode=0)
        missing = SimpleNamespace(returncode=1)
        with patch("norlan.interactive.prepare_launch", return_value=(["/codex"], {"CODEX_HOME": "/api"})), \
                patch("norlan.connection.subprocess.run", side_effect=[missing, ok]) as run, \
                patch.dict(os.environ, {"OPENAI_API_KEY": "local-test-placeholder"}), redirect_stdout(io.StringIO()):
            self.assertEqual(connection.setup("openai", "chosen-model"), 0)
        self.assertEqual(connection.load_connection()["provider"], "openai")
        self.assertIn('forced_login_method="api"', run.call_args.args[0])
        self.assertNotIn("local-test-placeholder", connection.connection_path().read_text())
        with patch("norlan.interactive.prepare_launch", return_value=(["/codex"], {"CODEX_HOME": "/subscription"})), \
                patch("norlan.connection.subprocess.run", return_value=ok) as run, redirect_stdout(io.StringIO()):
            self.assertEqual(connection.setup("chatgpt"), 0)
        self.assertEqual(connection.load_connection()["provider"], "chatgpt")
        self.assertIn('forced_login_method="chatgpt"', run.call_args.args[0])

    def test_running_session_connection_does_not_change_underneath_it(self):
        connection.save_connection({"provider": "chatgpt", "model": "another-session-model"})
        with patch.dict(os.environ, {"NORLAN_GPT_BILLING": "api", "CODEX_HOME": "/pinned/api"}):
            config = connection.production_brain_config({"provider": "codex"})
            info = connection.production_connection_info({"provider": "codex"})
        self.assertEqual(config["billing"], "api")
        self.assertEqual(config["codex_home"], "/pinned/api")
        self.assertEqual(info["billing"], "api")
        self.assertNotEqual(info["model"], "another-session-model")

    def test_cli_production_subscription_is_also_pinned_and_forced(self):
        with patch.dict(os.environ, {"NORLAN_GPT_BILLING": ""}), \
                patch("norlan.interactive.prepare_launch", return_value=([], {
                    "CODEX_HOME": "/isolated/subscription", "NORLAN_GPT_BILLING": "subscription"})):
            config = connection.production_brain_config({"provider": "codex"})
        self.assertEqual(config["billing"], "subscription")
        self.assertEqual(config["codex_home"], "/isolated/subscription")

    def test_direct_api_project_cannot_silently_override_subscription_choice(self):
        with patch.dict(os.environ, {"NORLAN_GPT_BILLING": "subscription"}):
            with self.assertRaisesRegex(ValueError, "固定使用 OpenAI API"):
                connection.production_brain_config({"provider": "openai", "model": "chosen"})

    def test_non_tty_cannot_fall_back_to_echoed_key_input(self):
        with patch("norlan.interactive.prepare_launch", return_value=(["/codex"], {})), \
                patch("norlan.connection.subprocess.run", return_value=SimpleNamespace(returncode=1)), \
                patch("norlan.connection.sys.stdin.isatty", return_value=False), \
                patch("norlan.connection.getpass.getpass") as key_prompt, \
                patch.dict(os.environ, {"OPENAI_API_KEY": ""}):
            with self.assertRaisesRegex(ValueError, "交互终端"):
                connection.setup("openai", "chosen")
        key_prompt.assert_not_called()

    def test_api_budget_required_before_production_request(self):
        project = self.root / "story"
        core.create_project(project, "A synthetic test story", "live")
        with patch("norlan.connection.production_brain_config", return_value={"provider": "codex", "billing": "api"}), \
                patch("norlan.brain.Brain") as brain:
            with self.assertRaises(core.GateError):
                pipeline._brain(project, "plan", "test")
        brain.assert_not_called()
        self.assertEqual(core.load_project(project)["jobs"], {})

    def test_api_production_preserves_reservation_and_selected_connection(self):
        project = self.root / "story"
        core.create_project(project, "A synthetic test story", "live")
        core.configure(project, {"caps": {"llm": 1}}, limit=3)
        config = {"provider": "codex", "billing": "api", "codex_home": "/isolated/api", "model": "chosen"}
        def check_reserved(_):
            before = core.load_project(project)
            self.assertFalse(before["jobs"])
            self.assertEqual(before["llm_calls"], 0)
            # Fake transport explicitly reaches the real pre-submit callback;
            # construction/local validation must not reserve budget by itself.
            brain.return_value.before_submit()
            state = core.load_project(project)
            self.assertEqual(state["budget"]["reserved"], 1)
            self.assertEqual(state["llm_calls"], 1)
            job = list(state["jobs"].values())[-1]
            self.assertEqual((job["state"], job["accounting"]), ("submitting", "reserved"))
            return {"ok": True}
        with patch("norlan.connection.production_brain_config", return_value=config), patch("norlan.brain.Brain") as brain:
            brain.return_value.plan.side_effect = check_reserved
            self.assertEqual(pipeline._brain(project, "plan", "test"), {"ok": True})
            brain.assert_called_once_with(config)
        self.assertEqual(core.load_project(project)["budget"]["spent"], 1)


class NativeEntryTests(unittest.TestCase):
    def test_default_launch_does_not_force_last_demo_or_create_project(self):
        with patch("norlan.interactive.launch", return_value=0) as launch, patch("norlan.cli.last_project") as last:
            self.assertEqual(main([]), 0)
        launch.assert_called_once_with(None, resume=False)
        last.assert_not_called()

    def test_resume_and_explicit_project(self):
        with patch("norlan.interactive.launch", return_value=0) as launch:
            self.assertEqual(main(["--resume"]), 0)
            launch.assert_called_once_with(None, resume=True)
        with patch("norlan.interactive.launch", return_value=0) as launch:
            self.assertEqual(main(["chat", "/a/project", "--resume"]), 0)
            launch.assert_called_once_with(Path("/a/project"), resume=True)

    def test_offline_console_is_explicit(self):
        with patch("norlan.chat.chat", return_value=0) as chat:
            self.assertEqual(main(["console", "/demo"]), 0)
        chat.assert_called_once_with(Path("/demo"))

    def test_setup_is_available_without_project(self):
        with patch("norlan.connection.setup", return_value=0) as setup:
            self.assertEqual(main(["setup", "--provider", "openai", "--model", "chosen"]), 0)
        setup.assert_called_once_with("openai", "chosen")


if __name__ == "__main__":
    unittest.main()
