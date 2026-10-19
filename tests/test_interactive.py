import os
from pathlib import Path
import tempfile
import tomllib
import unittest
from unittest.mock import patch

from norlan.brain import BrainError
from norlan.interactive import _CONFIG_HEADER, launch, native_home, prepare_launch


class InteractiveTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.original = self.root / "original-codex"
        self.original.mkdir()
        (self.original / "config.toml").write_text(
            'model = "chosen-gpt"\nmodel_reasoning_effort = "high"\n'
            '[mcp_servers.unrelated]\ncommand = "never-run-this"\n'
            '[features]\nhooks = true\n', encoding="utf-8")
        # Synthetic sentinel, deliberately not valid credentials.
        (self.original / "auth.json").write_text("synthetic-auth-not-a-key", encoding="utf-8")
        self.env = patch.dict(os.environ, {
            "XDG_DATA_HOME": str(self.root / "data"), "CODEX_HOME": str(self.original),
            "CODEX_SQLITE_HOME": str(self.root / "unrelated-database"),
            "CODEX_REMOTE_SERVER": "must-not-inherit", "SECRET_UNRELATED": "sentinel",
            "PYTHONPATH": str(self.root / "unrelated-code"),
        }, clear=False)
        self.env.start()
        self.resolve = patch("norlan.interactive.resolve_codex", return_value="/fake/codex")
        self.resolve.start()

    def tearDown(self):
        self.resolve.stop()
        self.env.stop()
        self.temp.cleanup()

    def config(self, env):
        return tomllib.loads((Path(env["CODEX_HOME"]) / "config.toml").read_text())

    def test_subscription_isolated_without_reading_or_copying_auth(self):
        original_auth = (self.original / "auth.json").read_bytes()
        original_config = (self.original / "config.toml").read_bytes()
        real_read_text = Path.read_text

        def guarded_read(path, *args, **kwargs):
            if path.name == "auth.json":
                raise AssertionError("launcher must not read authentication")
            return real_read_text(path, *args, **kwargs)

        with patch.object(Path, "read_text", guarded_read):
            argv, env = prepare_launch(connection={"provider": "chatgpt"})
        home = Path(env["CODEX_HOME"])
        self.assertNotEqual(home, self.original)
        self.assertTrue((home / "auth.json").is_symlink())
        self.assertEqual((home / "auth.json").resolve(), (self.original / "auth.json").resolve())
        self.assertEqual((self.original / "auth.json").read_bytes(), original_auth)
        self.assertEqual((self.original / "config.toml").read_bytes(), original_config)
        config = self.config(env)
        self.assertEqual(config["model"], "chosen-gpt")
        self.assertEqual(config["model_reasoning_effort"], "high")
        self.assertEqual(config["forced_login_method"], "chatgpt")
        self.assertEqual(env["NORLAN_GPT_BILLING"], "subscription")
        self.assertEqual(set(config["mcp_servers"]), {"norlan"})
        self.assertNotIn("SECRET_UNRELATED", env)
        self.assertNotIn("CODEX_REMOTE_SERVER", env)
        self.assertEqual(env["CODEX_SQLITE_HOME"], env["CODEX_HOME"])
        self.assertNotEqual(env["PYTHONPATH"], os.environ["PYTHONPATH"])
        self.assertIn("--no-daemon", argv)
        self.assertNotIn("--ignore-user-config", argv)
        self.assertNotIn("--ignore-rules", argv)
        self.assertNotIn("resume", argv)
        self.assertNotIn("--last", argv)
        self.assertNotIn("DEMO", config["developer_instructions"])
        for name in ("shell_tool", "unified_exec", "apps", "plugins", "hooks", "browser_use", "computer_use"):
            self.assertFalse(config["features"][name])
        self.assertTrue(config["features"]["code_mode"])
        self.assertTrue(config["features"]["code_mode_host"])
        self.assertEqual(config["sandbox_mode"], "read-only")
        self.assertEqual(config["approval_policy"], "never")
        self.assertEqual(config["project_doc_max_bytes"], 0)
        self.assertTrue(config["features"]["skip_host_skill_discovery"])
        self.assertNotIn("approve", config["mcp_servers"]["norlan"]["enabled_tools"])
        self.assertNotIn("budget", config["mcp_servers"]["norlan"]["enabled_tools"])

    def test_api_requires_explicit_model_and_never_links_subscription(self):
        with self.assertRaises(BrainError):
            prepare_launch(connection={"provider": "openai"})
        argv, env = prepare_launch(connection={"provider": "openai", "model": "api-chosen-model"})
        self.assertFalse((Path(env["CODEX_HOME"]) / "auth.json").exists())
        config = self.config(env)
        self.assertEqual(config["model"], "api-chosen-model")
        self.assertEqual(config["forced_login_method"], "api")
        self.assertNotIn("model_reasoning_effort", config)
        self.assertEqual(env["NORLAN_GPT_BILLING"], "api")
        self.assertNotEqual(native_home("chatgpt"), native_home("openai"))

    def test_api_refuses_auth_symlink(self):
        home = native_home("openai")
        (home / "auth.json").symlink_to(self.original / "auth.json")
        with self.assertRaises(BrainError):
            prepare_launch(connection={"provider": "openai", "model": "api-model"})
        self.assertTrue((home / "auth.json").is_symlink())

    def test_resume_and_user_prompt_preserve_native_history(self):
        argv, env = prepare_launch(resume=True, connection={"provider": "chatgpt"})
        self.assertEqual(argv[1], "resume")
        self.assertIn("--last", argv)
        self.assertNotIn("--ephemeral", argv)
        self.assertEqual(self.config(env)["history"]["persistence"], "save-all")
        argv, env = prepare_launch(prompt="讨论这个故事，不开始制作", connection={"provider": "chatgpt"})
        self.assertEqual(argv[-2:], ["--", "讨论这个故事，不开始制作"])
        with self.assertRaises(BrainError):
            prepare_launch(resume=True, prompt="a new message", connection={"provider": "chatgpt"})

    def test_selected_project_is_only_data_and_does_not_become_cwd(self):
        project = self.root / "小说项目"
        project.mkdir()
        (project / "project.json").write_text("{}")
        argv, env = prepare_launch(project, connection={"provider": "chatgpt"})
        self.assertNotEqual(Path(argv[argv.index("--cd") + 1]), project)
        self.assertIn(str(project), self.config(env)["developer_instructions"])
        self.assertEqual(self.config(env)["mcp_servers"]["norlan"]["args"][-2:], ["--project", str(project.resolve())])
        with self.assertRaises(BrainError):
            prepare_launch(self.root / "missing", connection={"provider": "chatgpt"})

    def test_existing_unrelated_directories_and_configs_are_not_overwritten(self):
        existing = self.root / "data" / "norlan" / "native"
        existing.mkdir(parents=True)
        keep = existing / "precious.txt"
        keep.write_text("keep me")
        with self.assertRaises(BrainError):
            native_home("chatgpt")
        self.assertEqual(keep.read_text(), "keep me")

    def test_managed_home_refuses_unrelated_config_and_wrong_auth_link(self):
        home = native_home("chatgpt")
        config = home / "config.toml"
        config.write_text('model = "handwritten"\n')
        with self.assertRaises(BrainError):
            prepare_launch(connection={"provider": "chatgpt"})
        self.assertEqual(config.read_text(), 'model = "handwritten"\n')
        config.unlink()
        (home / "auth.json").unlink()
        (home / "auth.json").symlink_to(self.root / "other-auth")
        with self.assertRaises(BrainError):
            prepare_launch(connection={"provider": "chatgpt"})

    def test_managed_config_is_reusable_and_no_profile_is_imported(self):
        argv, env = prepare_launch(connection={"provider": "chatgpt"})
        first = (Path(env["CODEX_HOME"]) / "config.toml").read_text()
        _, env2 = prepare_launch(connection={"provider": "chatgpt"})
        self.assertEqual(env["CODEX_HOME"], env2["CODEX_HOME"])
        self.assertEqual(first, (Path(env2["CODEX_HOME"]) / "config.toml").read_text())
        self.assertTrue(first.startswith(_CONFIG_HEADER))
        self.assertNotIn("profile", self.config(env))

    def test_launcher_attaches_tty_and_does_not_capture_or_send_initial_prompt(self):
        with patch("norlan.interactive.subprocess.call", return_value=17) as run:
            self.assertEqual(launch(connection={"provider": "chatgpt"}), 17)
        args, kwargs = run.call_args
        self.assertNotIn("--", args[0])
        self.assertNotIn("stdin", kwargs)
        self.assertNotIn("stdout", kwargs)
        self.assertNotIn("stderr", kwargs)
        self.assertNotIn("shell", kwargs)

    def test_linked_native_directory_is_refused(self):
        parent = self.root / "data" / "norlan"
        parent.mkdir(parents=True)
        (parent / "native").symlink_to(self.original, target_is_directory=True)
        with self.assertRaises(BrainError):
            native_home()

    def test_matching_bundled_code_mode_host_is_discoverable_without_extra_tools(self):
        bundle = self.root / "desktop" / "bin"
        bundle.mkdir(parents=True)
        executable = bundle / "codex"
        executable.write_text("synthetic wrapper")
        host = bundle / "codex-code-mode-host"
        host.write_text("synthetic host")
        host.chmod(0o700)
        with patch("norlan.interactive.resolve_codex", return_value=str(executable)):
            argv, env = prepare_launch(connection={"provider": "chatgpt"})
        self.assertEqual(Path(env["PATH"].split(os.pathsep)[0]), bundle.resolve())
        self.assertEqual(argv[0], str(executable))
        config = self.config(env)
        self.assertTrue(config["features"]["code_mode_host"])
        self.assertTrue(config["features"]["code_mode"])
        self.assertFalse(config["features"]["shell_tool"])
        self.assertFalse(config["features"]["unified_exec"])
        self.assertEqual(config["sandbox_mode"], "read-only")
        self.assertEqual(set(config["mcp_servers"]), {"norlan"})


if __name__ == "__main__":
    unittest.main()
