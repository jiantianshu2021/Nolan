import os
from pathlib import Path
import tempfile
import tomllib
import unittest
from unittest.mock import patch

from norlan.brain import BrainError
from norlan.interactive import _CONFIG_HEADER, launch, native_home, prepare_launch
from norlan.segments import SEGMENT_PLAN_RULES


class InteractiveTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name).resolve()
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
        self.assertIn(SEGMENT_PLAN_RULES, config["developer_instructions"])
        for name in ("apps", "plugins", "hooks", "browser_use", "computer_use"):
            self.assertFalse(config["features"][name])
        for name in ("shell_tool", "unified_exec", "view_image"):
            self.assertTrue(config["features"][name])
        self.assertTrue(config["features"]["code_mode"])
        self.assertTrue(config["features"]["code_mode_host"])
        self.assertEqual(config["sandbox_mode"], "workspace-write")
        self.assertEqual(config["approval_policy"], "on-request")
        self.assertEqual(config["approvals_reviewer"], "user")
        self.assertEqual(config["project_doc_max_bytes"], 32768)
        self.assertFalse(config["features"]["skip_host_skill_discovery"])
        self.assertTrue(config["skills"]["include_instructions"])
        self.assertTrue(config["skills"]["bundled"]["enabled"])
        self.assertNotIn("approve", config["mcp_servers"]["norlan"]["enabled_tools"])
        self.assertNotIn("budget", config["mcp_servers"]["norlan"]["enabled_tools"])
        self.assertIn("propose_segments", config["mcp_servers"]["norlan"]["enabled_tools"])
        self.assertIn("propose_asset_amendment", config["mcp_servers"]["norlan"]["enabled_tools"])
        from norlan.mcp_server import TOOLS
        self.assertEqual(set(config["mcp_servers"]["norlan"]["enabled_tools"]),
                         {tool["name"] for tool in TOOLS})
        self.assertIn("不能用 save_plan 重写旧计划", config["developer_instructions"])
        self.assertIn("等用户亲自点击确认后才提交视频", config["developer_instructions"])
        self.assertNotIn("approve_segments", config["mcp_servers"]["norlan"]["enabled_tools"])
        mcp_args = config["mcp_servers"]["norlan"]["args"]
        production_workspace = Path(mcp_args[mcp_args.index("--workspace") + 1])
        self.assertIn(str(production_workspace / "projects"), config["developer_instructions"])
        self.assertNotEqual(production_workspace, Path(argv[argv.index("--cd") + 1]))

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
        self.assertIn("--all", argv)
        self.assertNotIn("--ephemeral", argv)
        self.assertEqual(self.config(env)["history"]["persistence"], "save-all")
        argv, env = prepare_launch(prompt="讨论这个故事，不开始制作", connection={"provider": "chatgpt"})
        self.assertEqual(argv[-2:], ["--", "讨论这个故事，不开始制作"])
        with self.assertRaises(BrainError):
            prepare_launch(resume=True, prompt="a new message", connection={"provider": "chatgpt"})

    def test_explicit_selected_project_is_native_workdir_and_stays_mcp_selected(self):
        project = self.root / "小说项目"
        project.mkdir()
        (project / "project.json").write_text("{}")
        argv, env = prepare_launch(project, connection={"provider": "chatgpt"})
        self.assertEqual(Path(argv[argv.index("--cd") + 1]), project)
        self.assertIn(str(project), self.config(env)["developer_instructions"])
        arguments = self.config(env)["mcp_servers"]["norlan"]["args"]
        self.assertEqual(arguments[arguments.index("--project") + 1], str(project.resolve()))
        self.assertEqual(arguments[arguments.index("--workdir") + 1], str(project.resolve()))
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
        self.assertTrue(config["features"]["shell_tool"])
        self.assertTrue(config["features"]["unified_exec"])
        self.assertEqual(config["sandbox_mode"], "workspace-write")
        self.assertEqual(set(config["mcp_servers"]), {"norlan"})

    def test_explicit_workdir_is_writable_without_adopting_or_modifying_source_folder(self):
        directory = self.root / "御剑飞行"
        directory.mkdir()
        script = directory / "剧本.md"
        script.write_text("合成测试剧本，不调用模型。")
        argv, env = prepare_launch(workdir=directory, connection={"provider": "chatgpt"})
        self.assertEqual(Path(argv[argv.index("--cd") + 1]), directory)
        arguments = self.config(env)["mcp_servers"]["norlan"]["args"]
        self.assertEqual(arguments[arguments.index("--workdir") + 1], str(directory))
        self.assertFalse((directory / ".norlan-managed").exists())
        self.assertEqual(list(directory.iterdir()), [script])
        self.assertIn("--add-dir", argv)
        self.assertIn("不能通过 shell", self.config(env)["developer_instructions"])
        self.assertIn("list_directory", self.config(env)["mcp_servers"]["norlan"]["enabled_tools"])

    def test_resume_retains_user_workdir_without_granting_entire_home(self):
        directory = self.root / "chosen"
        directory.mkdir()
        prepare_launch(workdir=directory, connection={"provider": "chatgpt"})
        argv, env = prepare_launch(resume=True, connection={"provider": "chatgpt"})
        self.assertEqual(Path(argv[argv.index("--cd") + 1]), directory)
        self.assertNotEqual(Path(argv[argv.index("--cd") + 1]), Path.home())
        self.assertNotIn("danger-full-access", argv)
        self.assertEqual(self.config(env)["sandbox_mode"], "workspace-write")

    def test_shell_has_tools_path_but_not_service_keys(self):
        with patch.dict(os.environ, {"ARK_API_KEY": "synthetic-ark-secret", "OPENAI_API_KEY": "synthetic-openai-secret",
                                    "VOLC_SECRET_ACCESS_KEY": "synthetic-volc-secret"}):
            _, env = prepare_launch(connection={"provider": "chatgpt"})
        config = self.config(env)
        shell = config["shell_environment_policy"]
        self.assertEqual(shell["inherit"], "none")
        self.assertEqual(shell["set"]["PATH"], env["PATH"])
        self.assertEqual(shell["set"]["HOME"], env["HOME"])
        for key in ("ARK_API_KEY", "OPENAI_API_KEY", "VOLC_SECRET_ACCESS_KEY", "SECRET_UNRELATED"):
            self.assertNotIn(key, shell["set"])
        self.assertNotIn("synthetic-ark-secret", (Path(env["CODEX_HOME"]) / "config.toml").read_text())

    def test_new_managed_session_replaces_previous_workdir_before_resume(self):
        directory = self.root / "old-folder"
        directory.mkdir()
        prepare_launch(workdir=directory, connection={"provider": "chatgpt"})
        new, _ = prepare_launch(connection={"provider": "chatgpt"})
        resumed, _ = prepare_launch(resume=True, connection={"provider": "chatgpt"})
        self.assertEqual(new[new.index("--cd") + 1], resumed[resumed.index("--cd") + 1])
        self.assertNotEqual(Path(resumed[resumed.index("--cd") + 1]), directory)

    def test_first_launch_additional_write_root_already_exists(self):
        argv, _ = prepare_launch(connection={"provider": "chatgpt"})
        for index, flag in enumerate(argv):
            if flag == "--add-dir":
                self.assertTrue(Path(argv[index + 1]).is_dir())

    def test_existing_video_skill_is_discovered_without_importing_other_source_skills(self):
        skill = self.original / "skills" / "deepwhite-short-drama-video-prompts"
        skill.mkdir(parents=True)
        (skill / "SKILL.md").write_text("synthetic existing skill fixture")
        unrelated = self.original / "skills" / "unrelated"
        unrelated.mkdir()
        (unrelated / "SKILL.md").write_text("must not import")
        _, env = prepare_launch(connection={"provider": "chatgpt"})
        target = Path(env["CODEX_HOME"]) / "skills" / skill.name
        self.assertTrue(target.is_symlink())
        self.assertEqual(target.resolve(), skill)
        self.assertFalse((target.parent / unrelated.name).exists())
        prepare_launch(connection={"provider": "chatgpt"})
        self.assertEqual(target.resolve(), skill)

    def test_bad_workdir_and_linked_resume_record_do_not_broaden_access(self):
        for value in (self.root / "missing", "", 5):
            with self.assertRaises(BrainError):
                prepare_launch(workdir=value, connection={"provider": "chatgpt"})
        linked = self.root / "linked"
        linked.symlink_to(self.root, target_is_directory=True)
        with self.assertRaises(BrainError):
            prepare_launch(workdir=linked, connection={"provider": "chatgpt"})
        home = native_home("chatgpt")
        (home / "working-directory.json").symlink_to(self.original / "auth.json")
        with self.assertRaises(BrainError):
            prepare_launch(resume=True, connection={"provider": "chatgpt"})


if __name__ == "__main__":
    unittest.main()
