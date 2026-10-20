"""Synthetic keys only: mock storage, prompts and launches, never use Keychain."""
from contextlib import redirect_stderr, redirect_stdout
import getpass
import io
import json
import os
from pathlib import Path
import unittest
from unittest.mock import patch

from norlan import ark_setup, cli
from norlan.credentials import CredentialError


class ArkSetupTests(unittest.TestCase):
    def test_hidden_key_is_stored_and_child_receives_it_without_logging(self):
        output = io.StringIO()
        with patch.dict(os.environ, {}, clear=True), patch("sys.stdin.isatty", return_value=True), \
                patch("getpass.getpass", return_value=" SYNTHETIC-ARK-KEY ") as prompt, \
                patch("norlan.credentials.save_ark_key") as save, \
                patch("norlan.interactive.launch") as launch, redirect_stdout(output):
            def inspect(project, *, resume, workdir):
                self.assertEqual(os.environ["ARK_API_KEY"], "SYNTHETIC-ARK-KEY")
                self.assertEqual(project, Path("/synthetic/project"))
                self.assertTrue(resume)
                self.assertEqual(workdir, Path("/synthetic/story"))
                return 7
            launch.side_effect = inspect
            self.assertEqual(ark_setup.launch_with_ark(Path("/synthetic/project"), resume=True,
                                                       workdir=Path("/synthetic/story")), 7)
            save.assert_called_once_with("SYNTHETIC-ARK-KEY")
            prompt.assert_called_once()
            self.assertNotIn("ARK_API_KEY", os.environ)
        self.assertNotIn("SYNTHETIC-ARK-KEY", output.getvalue())

    def test_explicit_login_can_store_inherited_environment_without_prompt(self):
        with patch.dict(os.environ, {"ARK_API_KEY": "SYNTHETIC-ENV-KEY"}, clear=True), \
                patch("getpass.getpass") as prompt, patch("norlan.credentials.save_ark_key") as save, \
                patch("norlan.interactive.launch", return_value=0) as launch, redirect_stdout(io.StringIO()):
            self.assertEqual(ark_setup.launch_with_ark(), 0)
            save.assert_called_once_with("SYNTHETIC-ENV-KEY")
            prompt.assert_not_called()
            launch.assert_called_once_with(None, resume=False)
            self.assertEqual(os.environ["ARK_API_KEY"], "SYNTHETIC-ENV-KEY")

    def test_save_only_does_not_launch_or_leave_key_in_environment(self):
        with patch.dict(os.environ, {}, clear=True), patch("sys.stdin.isatty", return_value=True), \
                patch("getpass.getpass", return_value="SYNTHETIC-ARK-KEY"), \
                patch("norlan.credentials.save_ark_key") as save, \
                patch("norlan.interactive.launch") as launch, redirect_stdout(io.StringIO()):
            self.assertEqual(ark_setup.launch_with_ark(no_launch=True), 0)
            save.assert_called_once_with("SYNTHETIC-ARK-KEY")
            launch.assert_not_called()
            self.assertNotIn("ARK_API_KEY", os.environ)

    def test_noninteractive_entry_never_reads_or_saves_a_key(self):
        with patch.dict(os.environ, {}, clear=True), patch("sys.stdin.isatty", return_value=False), \
                patch("getpass.getpass") as prompt, patch("norlan.credentials.save_ark_key") as save, \
                patch("norlan.interactive.launch") as launch:
            with self.assertRaisesRegex(ValueError, "自己的交互终端"):
                ark_setup.launch_with_ark()
            prompt.assert_not_called()
            save.assert_not_called()
            launch.assert_not_called()

    def test_empty_or_control_characters_do_not_save_or_launch(self):
        for key in ("", " ", "SYNTHETIC\nKEY", "SYNTHETIC\rKEY", "SYNTHETIC\x00KEY"):
            with self.subTest(key=repr(key)), patch.dict(os.environ, {}, clear=True), \
                    patch("sys.stdin.isatty", return_value=True), patch("getpass.getpass", return_value=key), \
                    patch("norlan.credentials.save_ark_key") as save, \
                    patch("norlan.interactive.launch") as launch:
                with self.assertRaisesRegex(ValueError, "为空或格式无效"):
                    ark_setup.launch_with_ark()
                save.assert_not_called()
                launch.assert_not_called()

    def test_unsafe_getpass_fallback_and_eof_fail_without_echo_or_storage(self):
        for error in (getpass.GetPassWarning("SYNTHETIC unsafe fallback"), EOFError()):
            with self.subTest(error=type(error).__name__), patch.dict(os.environ, {}, clear=True), \
                    patch("sys.stdin.isatty", return_value=True), patch("getpass.getpass", side_effect=error), \
                    patch("norlan.credentials.save_ark_key") as save, \
                    patch("norlan.interactive.launch") as launch:
                with self.assertRaisesRegex(ValueError, "无法安全隐藏输入"):
                    ark_setup.launch_with_ark()
                save.assert_not_called()
                launch.assert_not_called()

    def test_keychain_error_does_not_fall_back_to_plaintext_or_start(self):
        with patch.dict(os.environ, {"ARK_API_KEY": "SYNTHETIC-ENV-KEY"}, clear=True), \
                patch("norlan.credentials.save_ark_key", side_effect=CredentialError("钥匙串已锁定，请先解锁")), \
                patch("norlan.interactive.launch") as launch, redirect_stdout(io.StringIO()) as output:
            with self.assertRaisesRegex(ValueError, "钥匙串已锁定"):
                ark_setup.launch_with_ark()
            launch.assert_not_called()
            self.assertEqual(os.environ["ARK_API_KEY"], "SYNTHETIC-ENV-KEY")
            self.assertNotIn("SYNTHETIC-ENV-KEY", output.getvalue())

    def test_launch_failure_restores_environment(self):
        with patch.dict(os.environ, {}, clear=True), patch("sys.stdin.isatty", return_value=True), \
                patch("getpass.getpass", return_value="SYNTHETIC-ARK-KEY"), \
                patch("norlan.credentials.save_ark_key"), \
                patch("norlan.interactive.launch", side_effect=ValueError("SYNTHETIC launch failure")), \
                redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(ValueError, "launch failure"):
                ark_setup.launch_with_ark()
            self.assertNotIn("ARK_API_KEY", os.environ)


class ArkLoginRoutingTests(unittest.TestCase):
    def test_cli_routes_credential_entry_without_launching_itself(self):
        cases = (
            (["ark-login"], None, False, None, False),
            (["ark-login", "--no-launch"], None, False, None, True),
            (["ark-login", "--resume", "-C", "./story"], None, True, Path("./story"), False),
            (["--resume", "-C", "./story", "ark-login"], None, True, Path("./story"), False),
            (["ark-login", "/synthetic/project", "--cd", "./story"], Path("/synthetic/project"), False, Path("./story"), False),
            (["-C", "./story", "ark-login", "--cd", "story"], None, False, Path("./story"), False),
        )
        for args, project, resume, workdir, no_launch in cases:
            with self.subTest(args=args), patch.object(ark_setup, "launch_with_ark", return_value=7) as run:
                self.assertEqual(cli.main(args), 7)
                options = {"workdir": workdir} if workdir is not None else {}
                run.assert_called_once_with(project, resume=resume, no_launch=no_launch, **options)

    def test_conflicting_workdirs_do_not_prompt_or_save(self):
        with patch.object(ark_setup, "launch_with_ark") as run, redirect_stderr(io.StringIO()) as output:
            self.assertEqual(cli.main(["-C", "./one", "ark-login", "-C", "./two"]), 2)
            run.assert_not_called()
            self.assertIn("工作目录冲突", output.getvalue())

    def test_setup_error_is_safe_cli_failure(self):
        with patch.object(ark_setup, "launch_with_ark", side_effect=ValueError("请先解锁本机钥匙串")), \
                redirect_stderr(io.StringIO()) as output:
            self.assertEqual(cli.main(["ark-login"]), 2)
            self.assertIn("请先解锁本机钥匙串", output.getvalue())

    def test_secret_cannot_be_supplied_as_command_line_option(self):
        with patch.object(ark_setup, "launch_with_ark") as run, redirect_stderr(io.StringIO()), \
                self.assertRaises(SystemExit) as error:
            cli.main(["ark-login", "--key", "SYNTHETIC-NOT-ACCEPTED"])
        self.assertEqual(error.exception.code, 2)
        run.assert_not_called()


class ArkDoctorStatusTests(unittest.TestCase):
    def test_doctor_uses_safe_keychain_status_without_printing_a_key(self):
        from norlan.brain import BrainError
        for source, available, message in (
            ("keychain", True, "已从本机钥匙串读取"),
            ("keychain", False, "钥匙串已锁定，请先解锁"),
            ("missing", False, "尚未配置，运行 nolan ark-login"),
        ):
            status = {"available": available, "source": source, "message": message,
                      "unrelated_private_field": "SYNTHETIC-DO-NOT-PRINT"}
            output = io.StringIO()
            with self.subTest(source=source, available=available), \
                    patch.dict(os.environ, {"ARK_API_KEY": "SYNTHETIC-ENV-DO-NOT-PRINT"}, clear=True), \
                    patch("norlan.credentials.ark_key_status", return_value=status) as inspect, \
                    patch("norlan.cli.shutil.which", return_value="/synthetic/tool"), \
                    patch("norlan.local_alignment.LocalSubtitleVerifier.check_ready"), \
                    patch("norlan.brain.resolve_codex", side_effect=BrainError("SYNTHETIC CLI unavailable")), \
                    redirect_stdout(output):
                self.assertEqual(cli.doctor(), 0)
                inspect.assert_called_once_with()
            findings, _ = json.JSONDecoder().raw_decode(output.getvalue())
            self.assertEqual(findings["ARK_API_KEY"], {
                "available": available, "source": source, "message": message})
            self.assertNotIn("DO-NOT-PRINT", output.getvalue())


if __name__ == "__main__":
    unittest.main()
