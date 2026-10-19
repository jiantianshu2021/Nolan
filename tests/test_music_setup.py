"""Synthetic credentials only: never contact a provider or store secrets."""
import io
import os
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

from norlan import cli, music_setup


class MusicSetupTests(unittest.TestCase):
    def test_hidden_input_reaches_child_environment_without_file_or_stdout_storage(self):
        with patch.dict(os.environ, {}, clear=True), patch("sys.stdin.isatty", return_value=True), \
                patch("getpass.getpass", side_effect=["TEST-AK", "TEST-SK"]) as secret_input, \
                patch("norlan.interactive.launch") as launch, redirect_stdout(io.StringIO()) as output:
            def inspect(project, *, resume):
                self.assertEqual(os.environ["VOLC_ACCESS_KEY_ID"], "TEST-AK")
                self.assertEqual(os.environ["VOLC_SECRET_ACCESS_KEY"], "TEST-SK")
                self.assertTrue(resume)
                return 0
            launch.side_effect = inspect
            self.assertEqual(music_setup.launch_with_music(resume=True), 0)
            self.assertEqual(secret_input.call_count, 2)
            self.assertNotIn("TEST-AK", output.getvalue())
            self.assertNotIn("TEST-SK", output.getvalue())
            self.assertNotIn("VOLC_SECRET_ACCESS_KEY", os.environ)

    def test_existing_environment_is_preserved_without_prompt(self):
        with patch.dict(os.environ, {"VOLC_ACCESS_KEY_ID": "TEST-AK", "VOLC_SECRET_ACCESS_KEY": "TEST-SK"}, clear=True), \
                patch("getpass.getpass") as prompt, patch("norlan.interactive.launch", return_value=0):
            music_setup.launch_with_music()
            prompt.assert_not_called()
            self.assertEqual(os.environ["VOLC_SECRET_ACCESS_KEY"], "TEST-SK")

    def test_noninteractive_input_and_invalid_secret_do_not_launch(self):
        with patch.dict(os.environ, {}, clear=True), patch("sys.stdin.isatty", return_value=False), \
                patch("norlan.interactive.launch") as launch:
            with self.assertRaisesRegex(ValueError, "交互终端"):
                music_setup.launch_with_music()
            launch.assert_not_called()

    def test_cli_routes_to_human_setup_without_provider_submission(self):
        with patch.object(music_setup, "launch_with_music", return_value=0) as run:
            self.assertEqual(cli.main(["music-login", "--resume"]), 0)
            run.assert_called_once_with(None, resume=True)


if __name__ == "__main__":
    unittest.main()
