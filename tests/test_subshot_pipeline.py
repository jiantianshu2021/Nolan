"""Synthetic multi-shot segments: real ledger, mocked providers, no API calls."""
import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from norlan import core, pipeline
from test_pipeline import choose_gpt_assets, fake_media, write


SUBSHOTS = [{"start": 0, "end": 5, "prompt": "中景，人物在桌旁转身。"},
            {"start": 5, "end": 10, "prompt": "切近景，人物拿起信封。"},
            {"start": 10, "end": 15, "prompt": "切双人中景，人物将信封递出。"}]


class SubshotPipelineTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.enterContext(fake_media())
        core.create_project(self.root, "离线多镜头生成段测试。")
        plan = pipeline.demo_plan()
        plan["episodes"][0]["shots"][0].update(duration=15, subshots=copy.deepcopy(SUBSHOTS))
        core.install_plan(self.root, plan)
        core.approve(self.root, "plan", "plan", 1)
        choose_gpt_assets(self.root)
        state = pipeline.run(self.root)
        for asset in state["assets"]:
            core.approve(self.root, "asset", asset["id"], asset["version"])
        preview = write(self.root / "media/voice.mp4", b"SYNTHETIC VOICE PREVIEW")
        audio = write(self.root / "media/voice.wav", b"SYNTHETIC VOICE REFERENCE")
        with core.transaction(self.root) as state:
            state["assets"].append({"id": "voice", "kind": "voice", "name": "演示人物声音",
                "character_id": "hero", "prompt": "中性声线", "status": "candidate", "version": 1,
                "file": str(preview.relative_to(self.root)), "sha256": core.digest(preview),
                "audio_file": str(audio.relative_to(self.root)), "audio_sha256": core.digest(audio)})
            shot = state["episodes"][0]["shots"][0]
            shot["asset_ids"].append("voice")
            shot["dialogue"] = "请把这封信交给他。"
        core.approve(self.root, "asset", "voice", 1)
        core.configure(self.root, {"caps": {"video_second": 1}, "max_poll_seconds": 1}, limit=100)
        with core.transaction(self.root) as state:
            state["mode"] = "live"
        previous_video = write(self.root / "media/previous-segment-v1.mp4", b"SYNTHETIC PREVIOUS VIDEO")
        previous_tail = write(self.root / "evidence/previous-tail.png", b"SYNTHETIC ACTUAL TAIL")
        relative_tail = str(previous_tail.relative_to(self.root))
        self.official_tail = write(self.root / "media/official-tails/previous.jpg", b"\xff\xd8\xffSYNTHETIC OFFICIAL TAIL")
        stamp, previous_sha = core.now(), core.digest(previous_video)
        self.previous = {"id": "previous-segment", "status": "accepted", "scene_id": "station", "version": 1,
            "file": str(previous_video.relative_to(self.root)), "sha256": core.digest(previous_video),
            "job_id": "synthetic-previous-job", "approved_sha256": previous_sha, "approved_at": stamp,
            "human_decision": {"decision": "approved", "version": 1, "sha256": previous_sha, "at": stamp},
            "review": {"observed_end_state": "人物在桌旁。", "sha256": previous_sha, "version": 1},
            "evidence": {"tail_frame": relative_tail, "frame_hashes": {relative_tail: core.digest(previous_tail)}}}
        core.reserve(self.root, "synthetic-previous-job", "video", self.previous["id"], 1)
        core.update_job(self.root, "synthetic-previous-job", state="completed", accounting="spent",
            task_id="synthetic-previous-task", output=self.previous["file"], origin_video={
                "file": self.previous["file"], "sha256": previous_sha, "task_id": "synthetic-previous-task", "created_at": stamp},
            official_tail={"file": str(self.official_tail.relative_to(self.root)), "sha256": core.digest(self.official_tail),
                "task_id": "synthetic-previous-task", "video_sha256": previous_sha,
                "source": "ark_official_tail", "created_at": stamp})
        self.provider = Mock()
        self.provider.submit_video.return_value = {"id": "synthetic-task"}
        self.provider.poll_video.return_value = {"status": "completed", "url": "https://example.test/video",
            "last_frame_url": "https://example.test/official-tail.jpg", "created_at": stamp}
        self.provider.download.side_effect = lambda url, path: write(path,
            b"\xff\xd8\xffSYNTHETIC OFFICIAL TAIL" if url.endswith(".jpg") else b"SYNTHETIC VIDEO " + str(path).encode())
        self.ark = self.enterContext(patch.object(pipeline, "_ark", return_value=self.provider))
        self.review_contexts = []
        self.review_verdicts = ["pass"]
        self.brain = self.enterContext(patch.object(pipeline, "_brain", side_effect=self.fake_brain))
        self.evidence = self.enterContext(patch.object(pipeline.media, "extract_evidence", side_effect=self.fake_evidence))

    def fake_brain(self, root, method, *args):
        if method == "refine_shot":
            return {"prompt": "SYNTHETIC refined prompt", "dialogue": "不可替换原定对白", "continuity_notes": "synthetic"}
        self.assertEqual(method, "review")
        self.review_contexts.append(copy.deepcopy(args[0]))
        verdict = self.review_verdicts.pop(0)
        return {"verdict": verdict, "issues": ["内部镜头二的动作需要修改"] if verdict == "fail" else [],
                "repair_prompt": "修正内部镜头二的动作，整段保持顺序。", "observed_end_state": "合成测试段末状态"}

    def fake_evidence(self, video, directory, *, subshots=None):
        self.assertEqual(subshots, SUBSHOTS)
        times = [0, 2.5, 4.9, 5.1, 7.5, 9.9, 10.1, 12.5, 14.966667]
        frames = [str(write(Path(directory) / f"sample-{index}.png", f"SYNTHETIC FRAME {index}".encode()))
                  for index in range(len(times))]
        return {"frames": frames, "tail_frame": frames[-1], "audio": None,
                "sha256": core.digest(video), "sample_times": times,
                "metadata": {"duration": 15, "has_video": True, "has_audio": True},
                "sampling_note": "只是计划点采样，不代表已检测实际切镜。"}

    def test_three_subshots_are_one_task_with_locked_timeline_dialogue_and_references(self):
        self.assertFalse(pipeline._shot(self.root, "E001", "E001-S001", self.previous))
        self.provider.submit_video.assert_called_once()
        prompt, duration, images = self.provider.submit_video.call_args.args
        self.assertEqual(duration, 15)
        for item in SUBSHOTS:
            self.assertIn(item["prompt"], prompt)
        self.assertIn("0–5秒", prompt)
        self.assertIn("5–10秒", prompt)
        self.assertIn("10–15秒", prompt)
        self.assertIn("必须使用的原定对白：请把这封信交给他。", prompt)
        self.assertNotIn("不可替换原定对白", prompt)
        self.assertEqual(images[-1], self.official_tail)
        self.assertNotIn(self.root / self.previous["evidence"]["tail_frame"], images)
        self.assertEqual(len(images), 3)
        self.assertEqual(self.provider.submit_video.call_args.kwargs["audio_references"], [self.root / "media/voice.wav"])
        state = core.load_project(self.root)
        video_jobs = [job for job in state["jobs"].values() if job["kind"] == "video" and job["item_id"] == "E001-S001"]
        self.assertEqual(len(video_jobs), 1)
        self.assertEqual(video_jobs[0]["amount"], 15)
        self.assertEqual(state["episodes"][0]["shots"][0]["subshots"], SUBSHOTS)
        context = self.review_contexts[0]
        self.assertEqual([item["image_index"] for item in context["current_frame_samples"]], list(range(4, 13)))
        self.assertAlmostEqual(context["current_frame_samples"][-1]["time_seconds"], 14.966667)
        self.assertIn("未检测", context["review_scope"])
        self.assertIn("用户选择返工", context["repair_unit"])
        self.assertEqual(context["previous_actual"]["shot_id"], "previous-segment")
        self.assertEqual(context["previous_actual"]["reference_mode"], "official_tail")
        self.assertEqual(context["previous_actual"]["human_decision"], self.previous["human_decision"])
        self.assertEqual(context["actual"]["tail_frame"], context["actual"]["frames"][-1])

    def test_only_human_rejection_retries_entire_segment_and_preserves_previous_version(self):
        self.review_verdicts = ["fail", "pass"]
        self.assertFalse(pipeline._shot(self.root, "E001", "E001-S001", self.previous))
        self.assertFalse(pipeline._shot(self.root, "E001", "E001-S001", self.previous))
        self.provider.submit_video.assert_called_once()
        core.reject(self.root, "shot", "E001-S001", 1, "用户要求修正第二个内镜动作")
        self.assertFalse(pipeline._shot(self.root, "E001", "E001-S001", self.previous))
        self.assertEqual(self.provider.submit_video.call_count, 2)
        self.assertEqual([call.args[1] for call in self.provider.submit_video.call_args_list], [15, 15])
        shot = core.load_project(self.root)["episodes"][0]["shots"][0]
        self.assertEqual(shot["version"], 2)
        self.assertEqual(shot["subshots"], SUBSHOTS)
        self.assertEqual(shot["history"][0]["version"], 1)
        self.assertTrue((self.root / shot["history"][0]["file"]).is_file())
        self.assertNotEqual(shot["sha256"], shot["history"][0]["sha256"])

    def test_pending_segment_proposal_blocks_run_and_direct_new_video(self):
        with core.transaction(self.root) as state:
            state["episodes"][0]["segment_proposal"] = {"status": "candidate"}
        before = core.load_project(self.root)
        state = pipeline.run(self.root)
        self.assertEqual(state["status"], "awaiting_segment_plan")
        self.assertEqual(state["jobs"], before["jobs"])
        with self.assertRaisesRegex(core.GateError, "等待人工批准"):
            pipeline._video(self.root, "video", "E001-S001", self.root / "media/new.mp4", 15, "synthetic", [])
        self.provider.submit_video.assert_not_called()
        self.brain.assert_not_called()
        self.evidence.assert_not_called()

    def test_pending_proposal_allows_registered_task_get_and_download_only(self):
        output = self.root / "media/old-task.mp4"
        core.reserve(self.root, "old-task", "video", "E001-S001", 15)
        core.update_job(self.root, "old-task", state="submitted", task_id="old-provider-task")
        with core.transaction(self.root) as state:
            state["episodes"][0]["segment_proposal"] = {"status": "candidate"}
        with patch.object(core, "reserve", wraps=core.reserve) as reserve:
            pipeline._video(self.root, "video", "E001-S001", output, 15, "synthetic", [], old_job="old-task")
        self.provider.poll_video.assert_called_once_with("old-provider-task")
        self.assertEqual(self.provider.download.call_count, 2)
        self.provider.download.assert_any_call("https://example.test/video", output)
        self.assertEqual(core.load_project(self.root)["jobs"]["old-task"]["official_tail"]["source"], "ark_official_tail")
        self.provider.submit_video.assert_not_called()
        reserve.assert_not_called()
        self.assertTrue(output.is_file())


if __name__ == "__main__":
    unittest.main()
