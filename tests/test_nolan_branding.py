"""Synthetic rename/upgrade tests; no real installation, browser or paid calls."""
from contextlib import redirect_stdout
import io
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import tomllib
import unittest
from unittest.mock import patch

from norlan import __version__, cli, connection, core, interactive, mcp_server


class NolanEntryTests(unittest.TestCase):
    def test_primary_and_legacy_commands_share_the_entry_point(self):
        metadata = tomllib.loads((Path(__file__).parents[1] / "pyproject.toml").read_text())
        self.assertEqual(metadata["project"]["scripts"], {
            "nolan": "norlan.cli:main", "norlan": "norlan.cli:main"})
        self.assertEqual(metadata["project"]["name"], "norlan-agent")
        self.assertEqual(metadata["project"]["version"], __version__)
        with redirect_stdout(io.StringIO()) as output, self.assertRaises(SystemExit) as raised:
            cli.main(["--help"])
        self.assertEqual(raised.exception.code, 0)
        self.assertIn("usage: nolan", output.getvalue())
        self.assertIn("Nolan", output.getvalue())

    def test_connection_and_resume_use_existing_config_and_data_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            config, data = root / "config", root / "data"
            old_connection = config / "norlan/connection.json"
            old_connection.parent.mkdir(parents=True)
            old_connection.write_text('{"provider": "chatgpt"}')
            with patch.dict(os.environ, {"XDG_CONFIG_HOME": str(config), "XDG_DATA_HOME": str(data)}):
                self.assertEqual(connection.load_connection()["provider"], "chatgpt")
                self.assertEqual(interactive.native_home(), data / "norlan/native/chatgpt")
                self.assertEqual(interactive.default_workspace(), data / "norlan/workspace")
                project = root / "existing-project"
                core.create_project(project, "synthetic fixture", "demo")
                cli.remember(project)
                self.assertEqual(cli.last_project(), project)
                self.assertTrue((config / "norlan/last-project").is_file())
                self.assertFalse((data / "nolan").exists())
                self.assertFalse((config / "nolan").exists())
        self.assertEqual(interactive._CONFIG_HEADER, "# Managed by Norlan native conversation v1\n")
        self.assertIn("你是 Nolan", interactive._DIRECTOR_INSTRUCTIONS)

    def test_cli_asset_gate_holds_review_server_without_extra_generation(self):
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory).resolve() / "project"
            core.create_project(project, "synthetic fixture", "demo")
            state = core.load_project(project)
            state["status"] = "awaiting_asset_input"
            for command in ("run", "resume"):
                with self.subTest(command=command), patch("norlan.cli.remember"), \
                        patch("norlan.pipeline.run", return_value=state) as run, \
                        patch("norlan.core.set_paused", return_value=state), \
                        patch("norlan.dashboard.serve") as serve, redirect_stdout(io.StringIO()) as output:
                    self.assertEqual(cli.main([command, str(project)]), 0)
                    serve.assert_called_once_with(project, port=0, open_browser=True)
                    self.assertEqual(run.call_count, 1 if command == "run" else 0)
                    self.assertIn("等待资产创作选择", output.getvalue())
            with patch("norlan.cli.remember"), patch("norlan.dashboard.serve") as serve, \
                    redirect_stdout(io.StringIO()):
                self.assertEqual(cli.main(["status", str(project)]), 0)
                serve.assert_not_called()


class NolanInstallerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.prefix, self.bin = self.root / "install", self.root / "bin"
        self.script = Path(__file__).parents[1] / "scripts/install.sh"
        # Stub venv/pip in this synthetic test; never use network or install packages.
        self.python = self.root / "synthetic-python"
        self.python.write_text(
            "#!" + sys.executable + "\n"
            "from pathlib import Path\nimport subprocess,sys\n"
            "if sys.argv[1] == '-c':\n"
            "    sys.exit(subprocess.run([sys.executable, *sys.argv[1:]]).returncode)\n"
            "if sys.argv[1:3] == ['-m', 'venv']:\n"
            "    target = Path(sys.argv[3]) / 'bin'\n"
            "    target.mkdir(parents=True, exist_ok=True)\n"
            "    python = target / 'python'\n"
            "    python.write_text('#!' + sys.executable + '\\nimport os,sys\\n'\n"
            "                       + 'sys.exit(1 if os.environ.get(\"NOLAN_TEST_FAIL_PIP\") else 0)\\n')\n"
            "    python.chmod(0o700)\n"
            "    for name in ('nolan', 'norlan'):\n"
            "        executable = target / name\n"
            f"        executable.write_text('#!/bin/sh\\nprintf \"%s\\\\n\" \"{__version__}\"\\n')\n"
            "        executable.chmod(0o700)\n"
        )
        self.python.chmod(0o700)
        self.env = {key: value for key, value in os.environ.items()
                    if not key.startswith(("NOLAN_", "NORLAN_"))}
        self.env.update(NOLAN_INSTALL_DIR=str(self.prefix), NOLAN_BIN_DIR=str(self.bin),
                        NOLAN_PYTHON=str(self.python))

    def run_installer(self):
        return subprocess.run(["bash", str(self.script)], env=self.env, capture_output=True,
                              text=True, timeout=10)

    def assert_entry_points(self):
        runtime = None
        for name in ("nolan", "norlan"):
            self.assertTrue((self.bin / name).is_symlink())
            target = Path(os.readlink(self.bin / name))
            current = target.parent.parent.parent
            self.assertEqual(current.parent, self.prefix / "runtimes")
            self.assertRegex(current.name, r"^" + __version__.replace(".", r"\.") + r"-[A-Za-z0-9]{8}$")
            self.assertEqual(target, current / "venv/bin" / name)
            if runtime is not None:
                self.assertEqual(current, runtime)
            runtime = current
        return runtime

    def test_fresh_install_creates_primary_and_legacy_aliases(self):
        result = self.run_installer()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("启动对话：nolan", result.stdout)
        self.assert_entry_points()

    def test_legacy_install_upgrade_keeps_state_and_both_commands(self):
        self.prefix.mkdir()
        self.bin.mkdir()
        (self.prefix / ".norlan-install").write_text("norlan-install-v1\n")
        existing = self.prefix / "workspace/projects/story/project.json"
        existing.parent.mkdir(parents=True)
        existing.write_text('{"synthetic": "unchanged"}')
        legacy_module = self.prefix / "venv/lib/existing-module.py"
        legacy_module.parent.mkdir(parents=True)
        legacy_module.write_text("# synthetic active old process code\n")
        (self.bin / "norlan").symlink_to(self.prefix / "venv/bin/norlan")
        # Old environment variable names remain accepted for existing install commands.
        for key in ("INSTALL_DIR", "BIN_DIR", "PYTHON"):
            self.env["NORLAN_" + key] = self.env.pop("NOLAN_" + key)
        result = self.run_installer()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(existing.read_text(), '{"synthetic": "unchanged"}')
        self.assertEqual(legacy_module.read_text(), "# synthetic active old process code\n")
        self.assert_entry_points()
        self.assertEqual(self.run_installer().returncode, 0)

    def test_same_version_reinstall_preserves_previous_runtime(self):
        self.assertEqual(self.run_installer().returncode, 0)
        previous = self.assert_entry_points()
        marker = previous / "synthetic-active-module.py"
        marker.write_text("# an active process may lazily read this module\n")
        before = {path.relative_to(previous): path.read_bytes() for path in previous.rglob("*") if path.is_file()}
        self.assertEqual(self.run_installer().returncode, 0)
        current = self.assert_entry_points()
        self.assertNotEqual(previous, current)
        self.assertEqual(before, {path.relative_to(previous): path.read_bytes() for path in previous.rglob("*") if path.is_file()})

    def test_failed_install_does_not_switch_either_launcher(self):
        self.assertEqual(self.run_installer().returncode, 0)
        previous = self.assert_entry_points()
        links = {name: os.readlink(self.bin / name) for name in ("nolan", "norlan")}
        self.env["NOLAN_TEST_FAIL_PIP"] = "1"
        self.assertNotEqual(self.run_installer().returncode, 0)
        self.assertEqual(links, {name: os.readlink(self.bin / name) for name in links})
        self.assertEqual(self.assert_entry_points(), previous)

    def test_linked_runtime_root_and_disguised_runtime_targets_are_rejected(self):
        self.prefix.mkdir()
        self.bin.mkdir()
        (self.prefix / ".norlan-install").write_text("norlan-install-v1\n")
        outside = self.root / "outside"
        outside.mkdir()
        runtimes = self.prefix / "runtimes"
        runtimes.symlink_to(outside, target_is_directory=True)
        self.assertNotEqual(self.run_installer().returncode, 0)
        self.assertEqual(list(outside.iterdir()), [])
        runtimes.unlink()
        runtimes.mkdir()
        disguised = runtimes / "0.3.0-AbCd1234"
        disguised.symlink_to(outside, target_is_directory=True)
        (self.bin / "nolan").symlink_to(disguised / "venv/bin/nolan")
        self.assertNotEqual(self.run_installer().returncode, 0)
        self.assertEqual(os.readlink(self.bin / "nolan"), str(disguised / "venv/bin/nolan"))
        self.assertEqual(list(outside.iterdir()), [])

    def test_existing_unrelated_new_command_is_not_overwritten(self):
        self.bin.mkdir()
        entry = self.bin / "nolan"
        entry.write_text("unrelated executable")
        result = self.run_installer()
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(entry.read_text(), "unrelated executable")
        self.assertFalse(self.prefix.exists())

    def test_unrelated_symlinks_for_either_name_are_not_overwritten(self):
        self.bin.mkdir()
        unrelated = self.root / "other-install"
        for name in ("nolan", "norlan"):
            with self.subTest(command=name):
                entry = self.bin / name
                entry.symlink_to(unrelated)
                result = self.run_installer()
                self.assertNotEqual(result.returncode, 0)
                self.assertEqual(os.readlink(entry), str(unrelated))
                self.assertFalse(self.prefix.exists())
                entry.unlink()


class AssetGateReviewTests(unittest.TestCase):
    def test_review_server_failure_is_separate_from_production_and_retry_recovers(self):
        with tempfile.TemporaryDirectory() as directory:
            server = mcp_server.MCPServer(Path(directory) / "workspace")
            self.addCleanup(server.close)
            root = server.projects / "story"
            core.create_project(root, "synthetic fixture", "demo")
            with core.transaction(root) as gated:
                gated["status"] = "awaiting_asset_input"
                gated["plan"]["version"] = 1
            gated = core.load_project(root)
            with patch("norlan.pipeline.run", return_value=gated), \
                    patch("norlan.dashboard.make_server", side_effect=OSError("private credential-like failure")):
                server._launch(root)
                server.workers[root].join(2)
            state = server._status(root)
            self.assertEqual(state["status"], "awaiting_asset_input")
            self.assertIsNone(state["background"]["error"])
            self.assertEqual(state["review_error"], mcp_server.REVIEW_ERROR_MESSAGE)
            self.assertIn("open_review", state["operator_actions"]["review_retry"])
            self.assertNotIn("private credential-like failure", str(state))
            self.assertFalse(server.reviews)
            with patch("norlan.mcp_server.webbrowser.open", return_value=False):
                result = server.call_tool("open_review", {"project": str(root)})
            self.assertTrue(result["url"].startswith("http://127.0.0.1:"))
            self.assertFalse(result["browser_opened"])
            self.assertIsNone(server._status(root)["review_error"])
            self.assertEqual(core.load_project(root)["jobs"], {})

    def test_review_thread_start_failure_closes_socket_and_uses_public_message(self):
        with tempfile.TemporaryDirectory() as directory:
            server = mcp_server.MCPServer(Path(directory) / "workspace")
            self.addCleanup(server.close)
            root = server.projects / "story"
            core.create_project(root, "synthetic fixture", "demo")
            review = mcp_server.dashboard.make_server(root, port=0)
            with patch("norlan.dashboard.make_server", return_value=review), \
                    patch("norlan.mcp_server.threading.Thread.start", side_effect=RuntimeError("private failure")):
                with self.assertRaisesRegex(mcp_server.ToolError, "open_review") as failure:
                    server.call_tool("open_review", {"project": str(root)})
            self.assertNotIn("private failure", str(failure.exception))
            self.assertEqual(review.fileno(), -1)
            self.assertFalse(server.reviews)
            self.assertIsNone(server._status(root)["background"]["error"])

    def test_mcp_gate_opens_once_per_plan_and_reuses_existing_server(self):
        with tempfile.TemporaryDirectory() as directory:
            server = mcp_server.MCPServer(Path(directory) / "workspace")
            self.addCleanup(server.close)
            root = server.projects / "story"
            core.create_project(root, "synthetic fixture", "demo")
            gated = core.load_project(root)
            gated.update(status="awaiting_asset_input")
            gated["plan"]["version"] = 1
            with patch("norlan.mcp_server.webbrowser.open", return_value=True) as browser, \
                    patch("norlan.pipeline.run", return_value=gated):
                first = server._open_review(root)
                self.assertEqual(browser.call_count, 1)
                for _ in range(2):
                    self.assertTrue(server._launch(root)["started"])
                    server.workers[root].join(2)
                    self.assertFalse(server.workers[root].is_alive())
                self.assertEqual(browser.call_count, 2)
                self.assertEqual(len(server.reviews), 1)
                self.assertEqual(server._status(root)["review_url"], first["url"])
                gated["plan"]["version"] = 2
                server._launch(root)
                server.workers[root].join(2)
                self.assertEqual(browser.call_count, 3)
                self.assertEqual(core.load_project(root)["jobs"], {})
                server.close()
                self.assertFalse(server.reviews[root][1].is_alive())

    def test_closed_mcp_session_cannot_spawn_a_review_server(self):
        with tempfile.TemporaryDirectory() as directory:
            server = mcp_server.MCPServer(Path(directory) / "workspace")
            server.close()
            with patch("norlan.dashboard.make_server") as make, self.assertRaises(mcp_server.ToolError):
                server._open_review(server.projects / "story", asset_plan_version=1)
            make.assert_not_called()


if __name__ == "__main__":
    unittest.main()
