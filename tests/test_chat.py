from contextlib import redirect_stdout
import io
import os
import subprocess
import sys
import tempfile
import threading
from types import SimpleNamespace
import unittest
from pathlib import Path
from unittest.mock import patch
from norlan import core, pipeline
from norlan.chat import _blank, _model_command, _new_from_message, chat, context, execute_intent, quick_command
from norlan.cli import doctor, main, preview


class ChatTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        core.create_project(self.root, "demo", "demo")
        core.install_plan(self.root, pipeline.demo_plan())
        self.state = core.load_project(self.root)

    def tearDown(self):
        self.temp.cleanup()

    def test_clear_approval(self):
        seen = context(self.state)
        cmd = quick_command("规划通过", seen)
        result = execute_intent(self.root, "规划通过", cmd, seen)
        self.assertEqual(result["plan"]["status"], "approved")

    def test_llm_cannot_approve_question_or_negation(self):
        seen = context(self.state)
        cmd = quick_command("规划通过", seen)
        for msg in ("规划可以吗？", "规划不要通过", "这张图不错", "规划不可以通过", "规划不能通过", "规划能通过吗", "如果规划通过就开始", "你说过规划通过", "规划通过后会做什么", '演员说“规划通过”', "规划通过？"):
            with self.assertRaises(ValueError):
                execute_intent(self.root, msg, cmd, seen)
        self.assertEqual(core.load_project(self.root)["plan"]["status"], "candidate")

    def test_intent_bound_to_seen_version(self):
        seen = context(self.state)
        core.install_plan(self.root, pipeline.demo_plan())
        with self.assertRaises(ValueError):
            execute_intent(self.root, "规划通过", quick_command("规划通过", seen), seen)

    def test_budget_must_appear_in_human_message(self):
        seen = context(self.state)
        cmd = quick_command("总预算设为 30 元", seen)
        self.assertEqual(execute_intent(self.root, "总预算设为 30 元", cmd, seen)["budget"]["limit"], 30)
        with self.assertRaises(ValueError):
            execute_intent(self.root, "查看预算", {**cmd, "limit": 1000}, seen)

    def test_help_never_calls_model(self):
        self.assertEqual(quick_command("查看进度", context(self.state))["action"], "status")

    def test_budget_questions_and_currency_mismatch_cannot_change_cap(self):
        seen = context(self.state)
        cmd = _blank("budget", limit=30)
        for message in ("预算还能 30 吗", "预算 30 元", "如果预算设为 30 元", "不要把预算设为 30 元", "预算设为 30 美元", "预算设为30元还是40元", "我说过预算设为30元"):
            with self.subTest(message=message), self.assertRaises(ValueError):
                execute_intent(self.root, message, cmd, seen)
        self.assertEqual(core.load_project(self.root)["budget"]["limit"], 0)
        self.assertIsNone(quick_command("预算 30 元", seen))
        self.assertEqual(execute_intent(self.root, "请把预算上限调整到 30 元", cmd, seen)["budget"]["limit"], 30)

    def test_run_and_resume_questions_do_not_unpause(self):
        core.set_paused(self.root, True)
        seen = context(core.load_project(self.root))
        for message, action in (("继续会怎么样", "resume"), ("如果开始制作", "run"), ("先不生成", "run"), ("不要恢复", "resume")):
            with self.subTest(message=message), self.assertRaises(ValueError):
                execute_intent(self.root, message, _blank(action), seen)
        self.assertTrue(core.load_project(self.root)["paused"])

    def test_negated_rework_never_rejects(self):
        seen = context(self.state)
        cmd = _blank("reject", kind="plan", id="plan", feedback="重新规划")
        for message in ("规划不要修改", "规划需要返工吗", "如果重新规划会怎样"):
            with self.subTest(message=message), self.assertRaises(ValueError):
                execute_intent(self.root, message, cmd, seen)
        self.assertEqual(core.load_project(self.root)["plan"]["status"], "candidate")

    def test_conversation_brain_respects_cross_process_run_lock(self):
        with core.run_lock(self.root), patch("norlan.chat.pipeline._brain") as brain:
            with self.assertRaises(core.GateError):
                _model_command(self.root, "为什么这里需要修改")
            brain.assert_not_called()
        with patch("norlan.chat.pipeline._brain", return_value=_blank("answer", reply="已读取")) as brain:
            intent, seen = _model_command(self.root, "解释当前规划")
            self.assertEqual(intent["action"], "answer")
            self.assertEqual(seen["title"], self.state["title"])
            brain.assert_called_once()

    def test_help_and_status_work_without_creating_project(self):
        output = io.StringIO()
        with patch("builtins.input", side_effect=["/help", "状态", "/quit"]), patch("norlan.chat._new_from_message") as create, redirect_stdout(output):
            self.assertEqual(chat(), 0)
        create.assert_not_called()
        self.assertIn("总预算设为", output.getvalue())
        self.assertIn("尚未创建项目", output.getvalue())

    def test_offline_episode_preview_uses_context_order(self):
        cmd = quick_command("播放第一集", context(self.state))
        self.assertEqual(cmd["kind"], "episode")
        self.assertEqual(cmd["id"], self.state["episodes"][0]["id"])
        with self.assertRaisesRegex(ValueError, "尚未生成"):
            preview(self.root, "episode", cmd["id"])

    def test_model_cannot_redirect_explicit_approval(self):
        seen = context(self.state)
        seen["items"] = [
            {"kind": "asset", "id": "hero", "name": "男主", "status": "candidate", "version": 1},
            {"kind": "asset", "id": "villain", "name": "反派", "status": "candidate", "version": 1},
        ]
        with patch("norlan.chat.core.approve") as approve:
            for message in ("请批准 hero", "请批准男主", "这个通过"):
                with self.subTest(message=message), self.assertRaises(ValueError):
                    execute_intent(self.root, message, _blank("approve", kind="asset", id="villain"), seen)
            approve.assert_not_called()

    def test_caption_correction_preserves_text_case_and_uses_seen_version(self):
        seen = context(self.state)
        episode = self.state["episodes"][0]
        message = episode["id"] + ' 字幕 把“你好吗”改成“Norlan 您好”'
        intent = quick_command(message, seen)
        self.assertEqual(intent["action"], "captions")
        self.assertIn("Norlan", intent["feedback"])
        with patch("norlan.chat.core.request_caption_fix", return_value=self.state) as corrected:
            execute_intent(self.root, message, intent, seen)
            corrected.assert_called_once_with(self.root, episode["id"], episode["version"], intent["feedback"])
        with patch("norlan.chat.core.request_caption_fix") as corrected:
            for message in (episode["id"] + " 字幕需要修改吗", episode["id"] + " 字幕不要修改"):
                with self.assertRaises(ValueError):
                    execute_intent(self.root, message, intent, seen)
            corrected.assert_not_called()

    def test_subtitle_realignment_is_an_explicit_caption_only_action(self):
        seen = context(self.state)
        episode = self.state["episodes"][0]
        instructions = (
            " 字幕 请重新对齐，前五条出现太早",
            " 请重新对齐字幕，前五条出现太早，不要重做视频",
            " 重新识别字幕",
            " 帮我重新校对字幕",
            " 请重新对齐字幕，不改视频",
            " 请重新对齐字幕，不改文字",
        )
        for instruction in instructions:
            message = episode["id"] + instruction
            intent = quick_command(message, seen)
            with self.subTest(message=message):
                self.assertEqual(intent["action"], "captions")
                self.assertEqual(intent["kind"], "episode")
                self.assertIn("重新", intent["feedback"])
                with patch("norlan.chat.core.request_caption_fix", return_value=self.state) as corrected, patch("norlan.chat.core.reject") as reject:
                    execute_intent(self.root, message, intent, seen)
                    corrected.assert_called_once_with(self.root, episode["id"], episode["version"], intent["feedback"])
                    reject.assert_not_called()

    def test_subtitle_realignment_questions_negations_and_observations_do_not_modify(self):
        seen = context(self.state)
        episode = self.state["episodes"][0]
        intent = _blank("captions", kind="episode", id=episode["id"], feedback="重新对齐字幕")
        with patch("norlan.chat.core.request_caption_fix") as corrected:
            for instruction in (
                " 字幕需要重新对齐吗", " 请重新对齐字幕？", " 字幕不要重新对齐",
                " 字幕不用再校对", " 字幕不必再重新识别", " 字幕别重新转录",
                " 字幕暂缓重新对齐", " 字幕稍后重新对齐", " 字幕先不重新对齐",
                " 字幕不要把字幕重新对齐", " 如果重新对齐字幕会发生什么",
                " 字幕前五条出现时间太早", " 字幕“请重新对齐”这句话",
                " 字幕不要改视频", " 字幕不要修改文字",
            ):
                message = episode["id"] + instruction
                with self.subTest(message=message), self.assertRaises(ValueError):
                    execute_intent(self.root, message, intent, seen)
            corrected.assert_not_called()

    def test_loaded_project_can_create_new_project(self):
        fresh = self.root / "new-project"
        core.create_project(fresh, "new novel", "demo")
        finished = threading.Event()
        def run(_root):
            finished.set()
            return core.load_project(_root)
        inputs = iter(["/new /explicit/novel.txt", "/quit"])
        def read(_prompt):
            value = next(inputs)
            if value == "/quit":
                self.assertTrue(finished.wait(timeout=2))
            return value
        with patch("builtins.input", side_effect=read), patch("norlan.chat._new_from_message", return_value=fresh) as create, patch("norlan.chat.pipeline.run", side_effect=run), patch("norlan.chat.remember"), redirect_stdout(io.StringIO()):
            self.assertEqual(chat(self.root), 0)
        create.assert_called_once_with("/new /explicit/novel.txt")
        self.assertTrue((self.root / "project.json").is_file())
        self.assertTrue(core.load_project(self.root)["paused"])

    def test_new_project_question_is_not_an_instruction(self):
        with self.assertRaises(ValueError):
            _new_from_message("能用 /tmp/story.txt 创建漫剧吗")

    def test_captions_cli_calls_only_caption_fix_then_resume(self):
        with patch("norlan.cli.core.request_caption_fix", return_value=self.state) as correction, patch("norlan.cli.core.set_paused"), patch("norlan.pipeline.run", return_value=self.state) as run, patch("norlan.cli.remember"), redirect_stdout(io.StringIO()):
            self.assertEqual(main(["captions", str(self.root), "E001", "--version", "2", "--correction", "把你好改为您好"]), 0)
        correction.assert_called_once_with(self.root.resolve(), "E001", 2, "把你好改为您好")
        run.assert_called_once_with(self.root.resolve())

    def test_doctor_uses_actual_codex_resolver(self):
        selected = "/Applications/Codex.app/Contents/Resources/codex"
        with patch("norlan.brain.resolve_codex", return_value=selected), patch("norlan.cli.subprocess.run", return_value=SimpleNamespace(returncode=0)) as run, patch("norlan.cli.shutil.which", return_value="/old/path"), redirect_stdout(io.StringIO()) as output:
            self.assertEqual(doctor(), 0)
        self.assertEqual(run.call_args.args[0], [selected, "login", "status"])
        self.assertIn(selected, output.getvalue())


class InstallerGuardTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.prefix, self.bin = self.root / "install", self.root / "bin"
        self.script = Path(__file__).resolve().parents[1] / "scripts/install.sh"
        self.env = {**os.environ, "NORLAN_INSTALL_DIR": str(self.prefix), "NORLAN_BIN_DIR": str(self.bin), "NORLAN_PYTHON": sys.executable}

    def tearDown(self):
        self.temp.cleanup()

    def run_installer(self):
        return subprocess.run(["bash", str(self.script)], env=self.env, capture_output=True, text=True, timeout=10)

    def test_unrelated_prefix_is_not_overwritten(self):
        self.prefix.mkdir()
        existing = self.prefix / "keep.txt"
        existing.write_text("keep")
        result = self.run_installer()
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(existing.read_text(), "keep")
        self.assertFalse((self.prefix / "venv").exists())

    def test_existing_unrelated_command_is_not_replaced(self):
        self.bin.mkdir()
        command = self.bin / "norlan"
        command.write_text("another executable")
        result = self.run_installer()
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(command.read_text(), "another executable")

    def test_owned_prefix_does_not_follow_venv_symlink(self):
        self.prefix.mkdir()
        (self.prefix / ".norlan-install").write_text("norlan-install-v1\n")
        external = self.root / "other-app"
        external.mkdir()
        (self.prefix / "venv").symlink_to(external, target_is_directory=True)
        result = self.run_installer()
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(list(external.iterdir()), [])


if __name__ == "__main__":
    unittest.main()
