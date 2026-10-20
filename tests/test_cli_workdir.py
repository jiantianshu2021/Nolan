"""Synthetic CLI routing tests; no GPT or production calls."""
from contextlib import redirect_stderr
import io
from pathlib import Path
import unittest
from unittest.mock import patch

from norlan.cli import main


class NativeWorkdirEntryTests(unittest.TestCase):
    def test_workdir_options_in_both_positions(self):
        for args in (["-C", "./story"], ["--cd", "/synthetic/story"],
                     ["chat", "-C", "./story"], ["-C", "./story", "chat"],
                     ["chat", "--cd", "/synthetic/story"]):
            with self.subTest(args=args), patch("norlan.interactive.launch", return_value=7) as launch:
                self.assertEqual(main(args), 7)
                directory = Path("/synthetic/story") if "/synthetic/story" in args else Path("./story")
                launch.assert_called_once_with(None, resume=False, workdir=directory)

    def test_resume_with_explicit_workdir(self):
        for args in (["--resume", "-C", "./story"],
                     ["chat", "--resume", "-C", "./story"],
                     ["-C", "./story", "chat", "--resume"],
                     ["--resume", "chat", "--cd", "./story"]):
            with self.subTest(args=args), patch("norlan.interactive.launch", return_value=0) as launch:
                self.assertEqual(main(args), 0)
                launch.assert_called_once_with(None, resume=True, workdir=Path("./story"))

    def test_project_is_distinct_from_workdir(self):
        for args in (["chat", "/synthetic/project", "-C", "./story"],
                     ["-C", "./story", "chat", "/synthetic/project"]):
            with self.subTest(args=args), patch("norlan.interactive.launch", return_value=0) as launch:
                self.assertEqual(main(args), 0)
                launch.assert_called_once_with(Path("/synthetic/project"), resume=False, workdir=Path("./story"))

    def test_unspecified_workdir_preserves_launch_interface(self):
        for args, project, resume in (([], None, False), (["--resume"], None, True),
                                      (["chat"], None, False),
                                      (["chat", "/synthetic/project"], Path("/synthetic/project"), False)):
            with self.subTest(args=args), patch("norlan.interactive.launch", return_value=0) as launch:
                self.assertEqual(main(args), 0)
                launch.assert_called_once_with(project, resume=resume)

    def test_conflicting_directories_are_rejected_before_launch(self):
        for args in (["-C", "./one", "-C", "./two"],
                     ["chat", "--cd", "./one", "-C", "./two"],
                     ["-C", "./one", "chat", "-C", "./two"]):
            error = io.StringIO()
            with self.subTest(args=args), patch("norlan.interactive.launch") as launch, redirect_stderr(error):
                self.assertEqual(main(args), 2)
                launch.assert_not_called()
                self.assertIn("工作目录冲突", error.getvalue())

    def test_same_directory_in_both_positions_is_unambiguous(self):
        with patch("norlan.interactive.launch", return_value=0) as launch:
            self.assertEqual(main(["-C", "./story", "chat", "--cd", "story"]), 0)
        launch.assert_called_once_with(None, resume=False, workdir=Path("./story"))

    def test_global_workdir_does_not_change_other_commands(self):
        for command in (["doctor"], ["setup"], ["connection"], ["console"], ["music-login"],
                        ["status", "/synthetic/project"], ["run", "/synthetic/project"]):
            error = io.StringIO()
            with self.subTest(command=command), patch("norlan.interactive.launch") as launch, \
                    patch("norlan.cli.doctor") as doctor, patch("norlan.connection.setup") as setup, \
                    patch("norlan.music_setup.launch_with_music") as music, redirect_stderr(error):
                self.assertEqual(main(["-C", "./story", *command]), 2)
                launch.assert_not_called()
                doctor.assert_not_called()
                setup.assert_not_called()
                music.assert_not_called()
                self.assertIn("--cd 只用于", error.getvalue())

    def test_non_chat_subcommands_do_not_accept_local_workdir_option(self):
        for command in (["doctor"], ["setup"], ["music-login"], ["status", "/synthetic/project"]):
            with self.subTest(command=command), redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
                main([*command, "-C", "./story"])
            self.assertEqual(error.exception.code, 2)

    def test_missing_workdir_value_is_parse_error(self):
        for args in (["-C"], ["chat", "--cd"]):
            with self.subTest(args=args), redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
                main(args)
            self.assertEqual(error.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
