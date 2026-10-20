"""Integration snapshots for human asset inputs and visible production phases.

All media and model providers are synthetic; no network or paid calls occur.
"""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from norlan import core, dashboard, pipeline
from norlan.billing import IMAGE_MODEL
from norlan.progress import episode_progress
from test_asset_inputs import data_url, png_data
from test_pipeline import choose_gpt_assets, fake_media, write, run_with_shot_approvals, run_confirming_prompts
from test_subshot_pipeline import confirming_shot


class PipelineIntakeTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        core.create_project(self.root, "离线合成测试剧本。")
        core.install_plan(self.root, pipeline.demo_plan())
        core.approve(self.root, "plan", "plan", 1)

    def make_live(self):
        core.configure(self.root, {"caps": {"image": 1, "video_second": 1},
                                   "max_poll_seconds": 1}, limit=100)
        with core.transaction(self.root) as state:
            state["mode"] = "live"

    def custom_asset(self):
        # Decoding is covered by test_asset_inputs. This test covers the exact
        # confirmed bytes/path reaching the generation adapter and ledger.
        with patch("norlan.asset_inputs.validate_image_file"):
            core.add_asset_reference(self.root, "hero", 0, 0, data_url())
        core.set_asset_input(self.root, "hero", 0, 1, "custom", "短银发、深蓝外套，保留参考图的衣领样式。")
        choose_gpt_assets(self.root)
        return core.load_project(self.root)["assets"][0]

    def test_confirmed_description_and_original_reference_reach_image_provider(self):
        asset = self.custom_asset()
        reference = asset["creation_input"]["references"][0]
        source = self.root / reference["file"]
        self.make_live()
        provider = Mock()

        def generate(prompt, output, *, references, on_usage):
            if "短银发" in prompt:
                self.assertEqual(references, [source])
                self.assertEqual(references[0].read_bytes(), png_data())
                self.assertEqual(core.digest(references[0]), reference["sha256"])
            write(output, b"SYNTHETIC GENERATED IMAGE")
            usage = {"provider_status": "succeeded", "model": IMAGE_MODEL,
                     "usage": {"generated_images": 1}}
            on_usage(usage)
            return usage

        provider.generate_image.side_effect = generate
        with patch.object(pipeline, "_ark", return_value=provider), \
             patch.object(pipeline, "_brain", side_effect=AssertionError("No GPT needed to preserve exact user input")):
            state = run_confirming_prompts(self.root)
        self.assertEqual(state["status"], "awaiting_assets", state.get("last_error"))
        self.assertEqual(provider.generate_image.call_count, 2)
        first = provider.generate_image.call_args_list[0]
        self.assertIn(asset["creation_input"]["description"], first.args[0])
        self.assertIn(asset["prompt"], first.args[0])
        self.assertIn(state["plan"]["style"], first.args[0])
        self.assertEqual(first.kwargs["references"], [source])
        saved = state["assets"][0]
        self.assertEqual(saved["status"], "candidate")
        self.assertNotIn("approved_sha256", saved)
        self.assertEqual(state["jobs"][saved["job_id"]]["input_references"], [reference])
        self.assertEqual(source.read_bytes(), png_data())

    def test_revision_keeps_global_style_and_sends_latest_human_feedback(self):
        choose_gpt_assets(self.root)
        self.assert_revision_keeps_style_and_feedback()

    def test_custom_revision_keeps_references_description_and_latest_feedback(self):
        asset = self.custom_asset()
        self.assert_revision_keeps_style_and_feedback(asset)

    def assert_revision_keeps_style_and_feedback(self, custom_asset=None):
        self.make_live()
        provider = Mock()

        def generate(prompt, output, **kwargs):
            write(output, b"SYNTHETIC IMAGE " + str(output).encode())
            return {"provider_status": "succeeded"}

        provider.generate_image.side_effect = generate
        feedback = "将人物的外套改为绿色，保留已经确定的发型。"
        with patch.object(pipeline, "_ark", return_value=provider):
            state = run_confirming_prompts(self.root)
            asset = state["assets"][0]
            core.reject(self.root, "asset", asset["id"], asset["version"], feedback)
            state = run_confirming_prompts(self.root)
        self.assertEqual(state["status"], "awaiting_assets", state.get("last_error"))
        revision_prompt = provider.generate_image.call_args.args[0]
        with self.subTest("global style"):
            self.assertIn(state["plan"]["style"], revision_prompt)
        with self.subTest("human feedback"):
            self.assertIn(feedback, revision_prompt)
        if custom_asset:
            creation = custom_asset["creation_input"]
            self.assertIn(creation["description"], revision_prompt)
            self.assertEqual(provider.generate_image.call_args.kwargs["references"],
                             [self.root / ref["file"] for ref in creation["references"]])
        self.assertEqual(state["assets"][0]["version"], 2)

    def test_tampered_confirmed_reference_stops_before_provider_or_reservation(self):
        asset = self.custom_asset()
        reference = asset["creation_input"]["references"][0]
        source = self.root / reference["file"]
        source.write_bytes(png_data(b"\x99\x88\x77"))
        self.make_live()
        with patch.object(pipeline, "_ark") as provider, \
             patch.object(pipeline, "_brain") as brain, \
             patch.object(core, "reserve", wraps=core.reserve) as reserve:
            state = run_confirming_prompts(self.root)
        self.assertEqual(state["status"], "blocked")
        self.assertIn("参考图片缺失或内容已改变", state["last_error"])
        self.assertEqual(state["jobs"], {})
        self.assertEqual(state["assets"][0]["version"], 0)
        provider.assert_not_called()
        brain.assert_not_called()
        reserve.assert_not_called()

    def test_human_retry_moves_failed_clip_to_history_before_next_refinement(self):
        core.set_video_review_enabled(self.root, True)
        with fake_media():
            choose_gpt_assets(self.root)
            state = run_confirming_prompts(self.root)
            for asset in state["assets"]:
                core.approve(self.root, "asset", asset["id"], asset["version"])
            self.make_live()
            provider = Mock()
            provider.submit_video.return_value = {"id": "synthetic-failed-clip"}
            provider.poll_video.return_value = {"status": "completed", "url": "https://example.test/clip",
                "last_frame_url": "https://example.test/official-tail.jpg", "created_at": core.now()}
            provider.download.side_effect = lambda url, output: write(output,
                b"\xff\xd8\xffSYNTHETIC OFFICIAL TAIL" if url.endswith(".jpg") else b"SYNTHETIC FAILED CLIP")
            refinements = []
            initial = core.load_project(self.root)["episodes"][0]["shots"][0]

            def brain(root, method, *args, **kwargs):
                if method == "plan_revision":
                    return {"needs_assets": False, "additions": [], "shot_updates": [],
                            "summary": "合成普通位置修正，复用既有资产。", "blockers": []}
                if method == "review":
                    return {"verdict": "fail", "issues": ["合成测试：人物位置不对"],
                            "observed_end_state": "synthetic", "repair_prompt": "修正人物位置"}
                self.assertEqual(method, "refine_shot")
                refinements.append(args[0].get("version", 0))
                if len(refinements) == 2:
                    current = core.load_project(root)
                    shot = current["episodes"][0]["shots"][0]
                    for field in ("file", "sha256", "evidence", "review", "refined_prompt", "job_id"):
                        self.assertNotIn(field, shot)
                    self.assertEqual(shot.get("revision", 0), initial.get("revision", 0) + 1)
                    self.assertEqual(shot["feedback"], "修正人物位置")
                    history = shot["history"][0]
                    self.assertEqual(history["version"], 1)
                    self.assertIn("evidence", history)
                    self.assertEqual(history["review"]["verdict"], "fail")
                    self.assertEqual(core.digest(self.root / history["file"]), history["sha256"])
                    public = dashboard.public_state(current)["episodes"][0]["shots"][0]
                    self.assertNotIn("file", public)
                    self.assertFalse(public["progress"]["completed"])
                    self.assertEqual(public["progress"]["stage"], "shot_preparation")
                    self.assertEqual(public["progress"]["target_version"], 2)
                    self.assertEqual(public["history"][0]["file"], history["file"])
                    self.assertEqual(public["history"][0]["version"], 1)
                    raise core.GateError("Synthetic stop before second submission")
                return {"prompt": "SYNTHETIC VIDEO", "dialogue": "", "continuity_notes": "synthetic"}

            with patch.object(pipeline, "_ark", return_value=provider), \
                 patch.object(pipeline, "_brain", side_effect=brain):
                self.assertFalse(confirming_shot(self.root, "E001", "E001-S001", None))
                self.assertEqual(refinements, [0])
                core.reject(self.root, "shot", "E001-S001", 1, "修正人物位置")
                with self.assertRaisesRegex(core.GateError, "Synthetic stop before second submission"):
                    confirming_shot(self.root, "E001", "E001-S001", None)
            self.assertEqual(refinements, [0, 1])
            provider.submit_video.assert_called_once()

    def test_live_episode_writes_observable_generation_and_postproduction_phases(self):
        with fake_media():
            choose_gpt_assets(self.root)
            state = run_confirming_prompts(self.root)
            for asset in state["assets"]:
                core.approve(self.root, "asset", asset["id"], asset["version"])
            self.make_live()
            provider = Mock()
            submitted = []
            observed = []
            real_progress = pipeline._progress

            def stage():
                current = core.load_project(self.root)
                return current["episodes"][0]["progress_stage"]["stage"]

            def record(*args, **kwargs):
                real_progress(*args, **kwargs)
                current = core.load_project(self.root)
                result = episode_progress(current, current["episodes"][0], now=core.now())
                observed.append((stage(), result["percent"]))

            def submit(*args, **kwargs):
                self.assertEqual(stage(), "shot_submission")
                task_id = "synthetic-task-" + str(len(submitted) + 1)
                submitted.append(task_id)
                return {"id": task_id}

            def poll(task_id):
                self.assertEqual(stage(), "shot_generation")
                return {"status": "completed", "url": "https://example.test/" + task_id,
                        "last_frame_url": "https://example.test/" + task_id + ".jpg", "created_at": core.now()}

            def download(url, output):
                self.assertEqual(stage(), "shot_download")
                write(output, (b"\xff\xd8\xffSYNTHETIC OFFICIAL TAIL " if url.endswith(".jpg") else b"SYNTHETIC VIDEO ") + url.encode())

            def brain(root, method, *args, **kwargs):
                if method == "refine_shot":
                    self.assertEqual(stage(), "shot_preparation")
                    return {"prompt": "SYNTHETIC VIDEO", "dialogue": "", "continuity_notes": "synthetic"}
                self.assertEqual(method, "review")
                self.assertIn(stage(), {"shot_review", "batch_review"})
                return {"verdict": "pass", "issues": [], "observed_end_state": "synthetic", "repair_prompt": ""}

            provider.submit_video.side_effect = submit
            provider.poll_video.side_effect = poll
            provider.download.side_effect = download
            with patch.object(pipeline, "_ark", return_value=provider), \
                 patch.object(pipeline, "_brain", side_effect=brain), \
                 patch.object(pipeline, "_progress", side_effect=record):
                state = run_with_shot_approvals(self.root)

        self.assertEqual(state["status"], "awaiting_episode_review", state.get("last_error"))
        stages = [stage for stage, _ in observed]
        self.assertIn("shot_checks", stages)
        self.assertEqual(stages[-3:], ["music", "subtitles", "rendering"])
        self.assertEqual(observed[-3:], [("music", 90.0), ("subtitles", 93.0), ("rendering", 96.0)])
        final = episode_progress(state, state["episodes"][0])
        self.assertEqual(final["percent"], 100)
        self.assertTrue(final["waiting_for_human"])
        self.assertEqual(episode_progress(state, state["episodes"][1])["stage"], "awaiting_previous_episode")
        self.assertEqual(len(submitted), 2)


if __name__ == "__main__":
    unittest.main()
