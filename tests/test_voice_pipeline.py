"""Synthetic project/media regression tests; never contact GPT or providers."""
import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from norlan import core, pipeline


LINE = "还要一百万违约金。厂章给我，我再去谈。"
SOURCE = "测试父亲：" + LINE
NOVEL = "合成测试剧本，仅用于离线回归。\n" + SOURCE + "\n测试路人：楼上的！先下来！"
SELECTION = {"audition_text": LINE, "audition_source": SOURCE,
             "audition_reason": "归属测试父亲的连续完整陈述；相对平稳，但仍是压力情境。"}


def write(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


class VoicePipelineTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        core.create_project(self.root, NOVEL, mode="live")
        plan = pipeline.demo_plan()
        plan["style"] = "旧剧情风格：暴雨中急切哭喊，疲惫、气息短。"
        plan["assets"][0]["name"] = "测试父亲"
        core.install_plan(self.root, plan)
        core.approve(self.root, "plan", "plan", 1)
        with core.transaction(self.root) as state:
            for asset in state["assets"]:
                path = write(self.root / "media" / f"{asset['id']}.png", b"SYNTHETIC IMAGE")
                asset.update(status="candidate", version=1,
                             file=str(path.relative_to(self.root)), sha256=core.digest(path))
        for asset in core.load_project(self.root)["assets"]:
            core.approve(self.root, "asset", asset["id"], 1)
        with core.transaction(self.root) as state:
            # A legacy voice intentionally lacks the new selection fields.
            state["assets"].append({"id": "voice", "kind": "voice", "name": "测试父亲声音",
                                    "character_id": "hero", "prompt": "55岁男声，略沙哑，疲惫、气息短、急切。",
                                    "status": "planned", "version": 0, "history": [], "feedback": ""})
        core.configure(self.root, {"caps": {"video_second": 1}, "max_poll_seconds": 1}, limit=100)
        self.selector = self.enterContext(patch.object(pipeline, "_brain", return_value=copy.deepcopy(SELECTION)))
        # Any unexpected provider path fails immediately instead of networking.
        self.provider = self.enterContext(patch.object(pipeline, "_ark", side_effect=AssertionError("No real provider calls in tests")))
        self.video_patch = patch.object(pipeline, "_video", side_effect=self.fake_video)
        self.video = self.video_patch.start()
        self.addCleanup(self.video_patch.stop)
        self.enterContext(patch.object(pipeline.media, "extract_evidence", side_effect=self.fake_evidence))

    @staticmethod
    def fake_video(root, kind, item_id, output, duration, prompt, references, **kwargs):
        write(output, b"SYNTHETIC VOICE VIDEO")

    @staticmethod
    def fake_evidence(video, directory):
        frame = write(Path(directory) / "frame.png", b"SYNTHETIC FRAME")
        audio = write(Path(directory) / "audio.wav", b"SYNTHETIC AUDIO")
        return {"frames": [str(frame)], "tail_frame": str(frame), "audio": str(audio),
                "sha256": core.digest(video), "sample_times": [0],
                "metadata": {"duration": 6, "has_video": True, "has_audio": True}}

    def voice(self):
        return core._find(core.load_project(self.root), "asset", "voice")

    def test_selected_source_is_saved_and_used_with_neutral_six_second_prompt(self):
        self.assertFalse(pipeline._assets(self.root))
        self.selector.assert_called_once()
        self.assertEqual(self.selector.call_args.args[1:3], ("select_voice_audition", NOVEL))
        args = self.video.call_args.args
        self.assertEqual(args[1:3], ("voice", "voice"))
        self.assertEqual(args[4], 6)
        prompt = args[5]
        for expected in (LINE, "中性语气", "正常语速", "正常音量", "干声", "中年年龄感", "轻微沙感"):
            self.assertIn(expected, prompt)
        for excluded in ("旧剧情风格", "暴雨", "疲惫", "气息短", "急切", "今天的天气不错"):
            self.assertNotIn(excluded, prompt)
        self.assertEqual(args[6], [(self.root / "media/hero.png").resolve()])
        saved = self.voice()
        self.assertEqual(saved["version"], 1)
        self.assertEqual(saved["status"], "candidate")
        self.assertEqual(saved["audition_text"], LINE)
        self.assertEqual(saved["audition_source"], SOURCE)
        self.assertEqual(saved["audition_profile"]["audition_source"], SOURCE)
        self.assertEqual(saved["audition_prompt"], prompt)
        # A candidate awaits a person; another run must not regenerate it.
        pipeline._assets(self.root)
        self.selector.assert_called_once()
        self.video.assert_called_once()
        self.provider.assert_not_called()

    def test_empty_selection_gates_without_generation_reservation_or_version_change(self):
        self.selector.return_value = {"audition_text": "", "audition_source": "",
                                      "audition_reason": "只有危急喊话，没有足够长的稳定台词，需补充原文。"}
        before = self.voice()
        with patch.object(core, "reserve", wraps=core.reserve) as reserve:
            with self.assertRaisesRegex(core.GateError, "原文试音段落"):
                pipeline._assets(self.root)
            first = pipeline.run(self.root)
            second = pipeline.run(self.root)
        self.assertEqual(first["status"], "blocked")
        self.assertEqual(second["status"], "blocked")
        self.assertIn("只有危急喊话", second["last_error"])
        self.assertEqual(self.voice()["version"], before["version"])
        self.assertEqual(self.voice()["status"], before["status"])
        self.assertFalse(second["jobs"])
        self.selector.assert_called_once()
        self.video.assert_not_called()
        self.provider.assert_not_called()
        reserve.assert_not_called()

    def test_unmatched_source_cannot_reach_paid_generation(self):
        self.selector.return_value = {**SELECTION, "audition_source": "测试父亲在另一部故事中说：" + LINE}
        with patch.object(core, "reserve", wraps=core.reserve) as reserve:
            with self.assertRaisesRegex(core.GateError, "不在项目原剧本"):
                pipeline._assets(self.root)
        self.assertEqual(self.voice()["version"], 0)
        self.assertEqual(self.voice()["status"], "planned")
        self.assertFalse(core.load_project(self.root)["jobs"])
        self.video.assert_not_called()
        self.provider.assert_not_called()
        reserve.assert_not_called()

    def test_invalid_selection_is_not_requested_again_for_identical_input(self):
        self.selector.return_value = {**SELECTION, "audition_source": "并不在原剧本中的测试父亲台词：" + LINE}
        first = pipeline.run(self.root)
        second = pipeline.run(self.root)
        self.assertEqual(first["status"], "blocked")
        self.assertEqual(second["status"], "blocked")
        self.assertEqual(self.voice()["version"], 0)
        self.video.assert_not_called()
        self.provider.assert_not_called()
        self.selector.assert_called_once()

    def test_preselected_original_passage_skips_selector(self):
        with core.transaction(self.root) as state:
            core._find(state, "asset", "voice").update(SELECTION)
        self.assertFalse(pipeline._assets(self.root))
        self.selector.assert_not_called()
        self.video.assert_called_once()
        self.assertEqual(self.video.call_args.args[4], 6)

    def test_approved_legacy_voice_is_untouched(self):
        preview = write(self.root / "media/voice-approved.mp4", b"SYNTHETIC APPROVED VIDEO")
        audio = write(self.root / "media/voice-approved.wav", b"SYNTHETIC APPROVED AUDIO")
        with core.transaction(self.root) as state:
            core._find(state, "asset", "voice").update(status="candidate", version=3,
                file=str(preview.relative_to(self.root)), sha256=core.digest(preview),
                audio_file=str(audio.relative_to(self.root)), audio_sha256=core.digest(audio))
        core.approve(self.root, "asset", "voice", 3)
        before = copy.deepcopy(self.voice())
        with patch.object(core, "reserve", wraps=core.reserve) as reserve:
            self.assertTrue(pipeline._assets(self.root))
        self.assertEqual(self.voice(), before)
        self.assertEqual(preview.read_bytes(), b"SYNTHETIC APPROVED VIDEO")
        self.assertEqual(audio.read_bytes(), b"SYNTHETIC APPROVED AUDIO")
        self.selector.assert_not_called()
        self.video.assert_not_called()
        self.provider.assert_not_called()
        reserve.assert_not_called()

    def test_submitted_legacy_voice_recovers_original_task_without_new_post(self):
        # Exercise the real recovery branch, with only the provider mocked.
        self.video_patch.stop()
        core.reserve(self.root, "existing-voice", "voice", "voice", 5)
        core.update_job(self.root, "existing-voice", state="submitted", task_id="synthetic-provider-task")
        with core.transaction(self.root) as state:
            core._find(state, "asset", "voice").update(status="generating", version=1, job_id="existing-voice")
        api = Mock()
        api.poll_video.return_value = {"status": "completed", "url": "https://example.test/synthetic-video.mp4"}
        api.download.side_effect = lambda url, output: write(output, b"SYNTHETIC RECOVERED VIDEO")
        self.provider.side_effect = None
        self.provider.return_value = api
        with patch.object(core, "reserve", wraps=core.reserve) as reserve:
            self.assertFalse(pipeline._assets(self.root))
        self.selector.assert_not_called()
        reserve.assert_not_called()
        api.submit_video.assert_not_called()
        api.preflight.assert_not_called()
        api.poll_video.assert_called_once_with("synthetic-provider-task")
        self.assertEqual(self.voice()["version"], 1)
        self.assertEqual(self.voice()["job_id"], "existing-voice")
        self.assertEqual(self.voice()["status"], "candidate")
        self.assertEqual(set(core.load_project(self.root)["jobs"]), {"existing-voice"})


if __name__ == "__main__":
    unittest.main()
