from contextlib import contextmanager, ExitStack
import copy
import os
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import Mock, patch

from norlan import core, pipeline
from norlan.providers import ProviderError


def write(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


def choose_gpt_assets(root):
    """Explicit synthetic human choices; do not bypass the production gate."""
    from norlan.asset_inputs import input_state, pending_assets
    for asset in pending_assets(core.load_project(root)):
        core.set_asset_input(root, asset["id"], asset.get("version", 0),
                             input_state(asset)["revision"], "gpt", "")


def run_with_shot_approvals(root):
    """Synthetic human decisions for fixtures that need a rendered episode."""
    for _ in range(20):
        state = pipeline.run(root)
        if state["status"] != "review_required":
            return state
        pending = [shot for episode in state["episodes"] for shot in episode["shots"]
                   if shot["status"] == "review_required"]
        if not pending or not pending[0].get("file"):
            raise AssertionError("Synthetic fixture has no candidate for a human to approve")
        core.approve(root, "shot", pending[0]["id"], pending[0]["version"])
    raise AssertionError("Synthetic fixture exceeded its bounded approval loop")


@contextmanager
def fake_media():
    """Fast deterministic media for state-machine tests; real smoke test below."""
    def asset(path, label, *args, **kwargs):
        return write(path, str(label).encode())

    def clip(path, label, duration, **kwargs):
        return write(path, (str(path) + str(label)).encode())

    def evidence(video, directory):
        frames = [str(write(Path(directory) / f"{index}.png", f"frame {index}".encode())) for index in range(3)]
        audio = str(write(Path(directory) / "audio.wav", b"voice"))
        return {"frames": frames, "tail_frame": frames[-1], "audio": audio, "sha256": core.digest(video),
                "sample_times": [0, 2, 3.9], "metadata": {"duration": 4, "has_video": True, "has_audio": True}}

    def compose(clips, output, **kwargs):
        write(output, b"composed:" + b"".join(Path(p).read_bytes() for p in clips))
        return {"duration": len(clips) * 4}

    def voice_references(paths, directory):
        return list(paths), [{"source_sha256": core.digest(p), "sha256": core.digest(p),
                              "trim_start": 0, "trim_duration": 4, "derived": False} for p in paths]

    with ExitStack() as stack:
        for name, fn in {"synthetic_asset": asset, "synthetic_clip": clip, "extract_evidence": evidence,
                         "compose": compose, "prepare_voice_references": voice_references,
                         "probe": lambda p: {"duration": 4, "has_video": True, "has_audio": True}}.items():
            stack.enter_context(patch.object(pipeline.media, name, side_effect=fn))
        yield


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        core.create_project(self.root, "一段原创小说", mode="demo")

    def ready(self):
        state = pipeline.run(self.root)
        self.assertEqual(state["status"], "awaiting_plan")
        core.approve(self.root, "plan", "plan", state["plan"]["version"])
        state = pipeline.run(self.root)
        self.assertEqual(state["status"], "awaiting_asset_input")
        self.assertFalse(state["jobs"])
        choose_gpt_assets(self.root)
        state = pipeline.run(self.root)
        self.assertEqual(state["status"], "awaiting_assets")
        for a in state["assets"]:
            core.approve(self.root, "asset", a["id"], a["version"])

    def live(self):
        core.configure(self.root, {"caps": {"image": 1, "video_second": 1}, "max_poll_seconds": 1}, limit=100)
        with core.transaction(self.root) as state:
            state["mode"] = "live"

    def test_episode_gate_rework_and_cross_episode_actual_context(self):
        with fake_media():
            self.ready()
            state = run_with_shot_approvals(self.root)
            self.assertEqual(state["status"], "awaiting_episode_review")
            first = state["episodes"][0]
            self.assertEqual(state["episodes"][1]["status"], "planned")
            self.assertFalse(any(j["item_id"].startswith("E002") for j in state["jobs"].values()))
            before = len(state["jobs"])
            pipeline.run(self.root)
            self.assertEqual(len(core.load_project(self.root)["jobs"]), before)
            core.reject(self.root, "episode", "E001", first["version"], "最后一镜动作修正", ["E001-S002"])
            state = run_with_shot_approvals(self.root)
            self.assertEqual(state["episodes"][0]["shots"][0]["version"], 1)
            self.assertEqual(state["episodes"][0]["shots"][1]["version"], 2)
            with self.assertRaises(core.GateError):
                core.approve(self.root, "episode", "E001", first["version"])
            core.approve(self.root, "episode", "E001", state["episodes"][0]["version"])
            state = run_with_shot_approvals(self.root)
            self.assertEqual(state["episodes"][1]["status"], "awaiting_review")
            context = state["episodes"][1]["shots"][0]["reference_manifest"]["context"]
            self.assertEqual(context["previous_actual"]["shot_id"], "E001-S002")
            self.assertEqual(context["image_references"][-1]["role"], "previous_actual_tail")

    def test_rejected_asset_creates_new_task_and_version(self):
        with fake_media():
            self.ready()
            state = core.load_project(self.root)
            old = copy.deepcopy(state["assets"][0])
            core.reject(self.root, "asset", "hero", old["version"], "重绘服饰")
            state = pipeline.run(self.root)
            new = state["assets"][0]
            self.assertEqual(new["status"], "candidate")
            self.assertEqual(new["version"], old["version"] + 1)
            self.assertNotEqual(new["job_id"], old["job_id"])
            self.assertTrue((self.root / old["file"]).is_file())

    def test_completed_image_wrong_output_is_not_reused(self):
        with fake_media():
            self.ready()
            with core.transaction(self.root) as state:
                a = state["assets"][0]
                a.update(status="generating", version=2)
            with patch.object(pipeline, "_ark") as ark:
                state = pipeline.run(self.root)
            self.assertEqual(state["status"], "blocked")
            self.assertIn("输出路径", state["last_error"])
            ark.assert_not_called()

    def test_unknown_submission_is_never_repeated(self):
        with fake_media():
            self.ready()
            self.live()
            api = Mock()
            api.submit_video.side_effect = ProviderError("timeout", uncertain=True)
            shot = core.load_project(self.root)["episodes"][0]["shots"][0]
            output = self.root / "media" / "first.mp4"
            with patch.object(pipeline, "_ark", return_value=api):
                with self.assertRaises(ProviderError):
                    pipeline._video(self.root, "video", shot["id"], output, 4, "prompt", [])
                state = pipeline.run(self.root)
                pipeline.run(self.root)
            self.assertEqual(api.submit_video.call_count, 1)
            self.assertEqual(state["budget"]["unknown"], 4)
            self.assertEqual(state["status"], "blocked")

    def test_known_task_download_failure_recovers_same_id(self):
        with fake_media():
            self.ready()
            self.live()
            api = Mock()
            api.submit_video.return_value = {"id": "provider-task"}
            api.poll_video.return_value = {"status": "completed", "url": "https://example.test/video.mp4"}
            api.download.side_effect = ProviderError("network")
            output = self.root / "media" / "first.mp4"
            with patch.object(pipeline, "_ark", return_value=api):
                with self.assertRaises(ProviderError):
                    pipeline._video(self.root, "video", "E001-S001", output, 4, "prompt", [])
                state = core.load_project(self.root)
                job_id = state["episodes"][0]["shots"][0]["job_id"]
                self.assertEqual(state["jobs"][job_id]["state"], "download_pending")
                api.download.side_effect = lambda url, path: write(path, b"downloaded")
                pipeline._video(self.root, "video", "E001-S001", output, 4, "prompt", [], old_job=job_id)
            self.assertEqual(api.submit_video.call_count, 1)
            self.assertEqual([c.args[0] for c in api.poll_video.call_args_list], ["provider-task", "provider-task"])
            self.assertEqual(core.load_project(self.root)["budget"]["spent"], 4)

    def test_missing_credentials_or_model_reserves_no_media_budget(self):
        with fake_media():
            self.ready()
            self.live()
            before = len(core.load_project(self.root)["jobs"])
            output = self.root / "media" / "first.mp4"
            with patch.dict(os.environ, {}, clear=True):
                with self.assertRaises(ProviderError):
                    pipeline._video(self.root, "video", "E001-S001", output, 4, "prompt", [])
            with patch.dict(os.environ, {"ARK_API_KEY": "test-only"}):
                with self.assertRaises(ProviderError):
                    pipeline._video(self.root, "video", "E001-S001", output, 4, "prompt", [])
            state = core.load_project(self.root)
            self.assertEqual(len(state["jobs"]), before)
            self.assertEqual(state["budget"]["spent"], 0)

    def test_voice_audition_uses_its_bound_character_only(self):
        with fake_media():
            self.ready()
            with core.transaction(self.root) as state:
                character = copy.deepcopy(state["assets"][0])
                character.update(id="second_hero", name="第二角色")
                path = write(self.root / "media" / "second-hero.png", b"second character")
                character.update(file=str(path.relative_to(self.root)), sha256=core.digest(path))
                state["assets"].append(character)
                state["assets"].append({"id": "second_voice", "kind": "voice", "name": "第二角色声音", "prompt": "沉稳声音",
                                        "character_id": "second_hero", "status": "planned", "version": 0, "history": [], "feedback": ""})
            choose_gpt_assets(self.root)
            captured = []
            def video(root, kind, item_id, output, duration, prompt, references, **kwargs):
                captured.extend(references)
                write(output, b"voice preview")
            with patch.object(pipeline, "_video", side_effect=video):
                state = pipeline.run(self.root)
            self.assertEqual(captured, [path.resolve()])
            voice = state["assets"][-1]
            self.assertEqual(voice["status"], "candidate")
            core.approve(self.root, "asset", voice["id"], voice["version"])
            state = core.load_project(self.root)
            shot = copy.deepcopy(state["episodes"][0]["shots"][0])
            shot["asset_ids"].append("second_voice")
            with self.assertRaises(core.GateError):
                pipeline._shot_context(self.root, state["episodes"][0], shot, None)
            shot["asset_ids"].append("second_hero")
            context, images, voices = pipeline._shot_context(self.root, state["episodes"][0], shot, None)
            self.assertEqual(context["audio_references"][0]["character_id"], "second_hero")

    def test_changed_approved_episode_export_blocks_next_episode(self):
        with fake_media():
            self.ready()
            state = run_with_shot_approvals(self.root)
            first = state["episodes"][0]
            core.approve(self.root, "episode", "E001", first["version"])
            write(self.root / first["file"], b"changed approved episode")
            state = run_with_shot_approvals(self.root)
            self.assertEqual(state["status"], "blocked")
            self.assertFalse(any(j["item_id"].startswith("E002") for j in state["jobs"].values()))

    def test_pause_and_busy_run_do_not_submit(self):
        with fake_media():
            self.ready()
            core.set_paused(self.root, True)
            state = pipeline.run(self.root)
            before = len(state["jobs"])
            self.assertEqual(state["status"], "paused")
            core.set_paused(self.root, False)
            with core.run_lock(self.root):
                state = pipeline.run(self.root)
            self.assertEqual(len(state["jobs"]), before)
            state = pipeline.run(self.root)
            self.assertEqual(state["status"], "review_required")

    def test_demo_cannot_call_paid_services_even_when_configured(self):
        core.configure(self.root, {"music": {"mode": "volc", "call_cap": 1},
                                  "subtitles": {"mode": "openai", "call_cap": 1}}, limit=10)
        episode = pipeline.demo_plan()["episodes"][0]
        for shot in episode["shots"]:
            shot["sha256"] = "synthetic-test-hash"
        with patch("norlan.music.MusicProvider") as music, patch("norlan.transcription.Transcriber") as asr, patch("norlan.brain.Brain") as brain:
            with self.assertRaises(core.GateError):
                pipeline._soundtrack(self.root, episode)
            with self.assertRaises(core.GateError):
                pipeline._captions(self.root, episode)
            with self.assertRaises(core.GateError):
                pipeline._brain(self.root, "plan", "test")
            music.assert_not_called()
            asr.assert_not_called()
            brain.assert_not_called()
        self.assertFalse(core.load_project(self.root)["jobs"])

    def test_interrupted_submission_enters_unknown_gate(self):
        with fake_media():
            self.ready()
            core.configure(self.root, limit=10)
            core.reserve(self.root, "crash-window", "video", "E001-S001", 4)
            with patch.object(pipeline, "_ark") as ark:
                state = pipeline.run(self.root)
            self.assertEqual(state["jobs"]["crash-window"]["state"], "uncertain")
            self.assertEqual(state["budget"]["unknown"], 4)
            ark.assert_not_called()

    def test_explicit_human_revision_can_generate_once_beyond_old_attempt_cap(self):
        with fake_media():
            self.ready()
            core.configure(self.root, {"max_attempts": 1})
            state = pipeline.run(self.root)
            self.assertEqual(state["status"], "review_required")
            shot = state["episodes"][0]["shots"][0]
            core.reject(self.root, "shot", shot["id"], shot["version"], "用户明确要求改正动作")
            before = len(core.load_project(self.root)["jobs"])
            state = pipeline.run(self.root)
            self.assertEqual(state["status"], "review_required")
            self.assertEqual(state["episodes"][0]["shots"][0]["version"], 2)
            self.assertEqual(len(state["jobs"]), before + 1)
            self.assertEqual(len(pipeline.run(self.root)["jobs"]), before + 1)

    def test_approved_episode_invalidated_when_prior_episode_changes(self):
        with fake_media():
            self.ready()
            state = run_with_shot_approvals(self.root)
            core.approve(self.root, "episode", "E001", state["episodes"][0]["version"])
            state = run_with_shot_approvals(self.root)
            core.approve(self.root, "episode", "E002", state["episodes"][1]["version"])
            core.reject(self.root, "episode", "E001", state["episodes"][0]["version"], "改变结尾", ["E001-S002"])
            state = core.load_project(self.root)
            self.assertEqual(state["episodes"][1]["status"], "revision_requested")
            self.assertTrue(all(sh["status"] == "review_required" for sh in state["episodes"][1]["shots"]))
            self.assertNotIn("approved_sha256", state["episodes"][1])

    def test_provider_failure_waits_for_explicit_human_retry(self):
        with fake_media():
            self.ready()
            self.live()
            api = Mock()
            api.submit_video.side_effect = ProviderError("rejected", uncertain=False)
            refinement = {"prompt": "new", "dialogue": "", "continuity_notes": "none"}
            with patch.object(pipeline, "_ark", return_value=api), patch.object(pipeline, "_brain", return_value=refinement):
                state = pipeline.run(self.root)
                pipeline.run(self.root)
                state = pipeline.run(self.root)
                self.assertEqual(state["status"], "generation_failed")
                self.assertEqual(api.submit_video.call_count, 1)
                core.reject(self.root, "shot", "E001-S001", 1, "用户要求再试一次")
                state = pipeline.run(self.root)
                pipeline.run(self.root)
            self.assertEqual(api.submit_video.call_count, 2)
            self.assertEqual(state["status"], "generation_failed")
            self.assertEqual(state["budget"]["spent"], 8)

    @unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "requires FFmpeg")
    def test_real_synthetic_media_end_to_end(self):
        self.ready()
        state = run_with_shot_approvals(self.root)
        self.assertEqual(state["status"], "awaiting_episode_review", state.get("last_error"))
        first = state["episodes"][0]
        info = pipeline.media.probe(self.root / first["file"])
        self.assertTrue(info["has_video"])
        self.assertGreater(info["duration"], 7)
        self.assertFalse(any(j["item_id"].startswith("E002") for j in state["jobs"].values()))
        core.approve(self.root, "episode", "E001", first["version"])
        state = run_with_shot_approvals(self.root)
        self.assertEqual(state["episodes"][1]["status"], "awaiting_review", state.get("last_error"))


if __name__ == "__main__":
    unittest.main()
