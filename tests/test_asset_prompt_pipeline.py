"""Synthetic live-mode asset consent integration; no real providers or approvals.

All explicit approvals here concern temporary synthetic fixtures. Provider calls
are mocks, and every media file is visibly synthetic bytes in a temp directory.
"""
import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from norlan import asset_inputs, core, pipeline


IMAGE_CHOICE = "doubao-seedream-4-0-250828"
VOICE_CHOICE = "doubao-seedance-2-0-260128"
LINE = "设备已经完成初始化，请按照操作说明继续查看当前任务。"
SOURCE = "演示人物卡：" + LINE


def write(path, content):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return path


class AssetPromptPipelineTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        core.create_project(self.root, "合成离线测试剧本。\n" + SOURCE, mode="live")
        core.install_plan(self.root, pipeline.demo_plan())
        core.approve(self.root, "plan", "plan", 1)
        core.set_remaining_asset_inputs_gpt(self.root, 1)
        core.configure(self.root, {"caps": {"image": 1, "video_second": 1},
                                   "max_poll_seconds": 1}, limit=200)
        self.images, self.videos, self.provider_configs = [], [], []
        self.active_config = {}
        self.provider = Mock()
        self.provider.generate_image.side_effect = self.generate_image
        self.provider.submit_video.side_effect = self.submit_video
        self.provider.poll_video.return_value = {"status": "completed",
            "url": "https://example.test/synthetic-voice.mp4"}
        self.provider.download.side_effect = lambda _url, target: write(target, b"SYNTHETIC DOWNLOADED VIDEO")
        self.ark = self.enterContext(patch.object(pipeline, "_ark", side_effect=self.provider_for))
        self.brain = self.enterContext(patch.object(pipeline, "_brain", side_effect=AssertionError(
            "This synthetic fixture must not call GPT or a real provider")))
        self.enterContext(patch.object(pipeline.media, "extract_evidence", side_effect=self.evidence))

    def provider_for(self, _root, config=None):
        self.active_config = copy.deepcopy(config or {})
        self.provider_configs.append(copy.deepcopy(self.active_config))
        return self.provider

    def generate_image(self, prompt, target, *, references, **_kwargs):
        self.images.append({"prompt": prompt, "references": list(references),
                            "config": copy.deepcopy(self.active_config)})
        write(target, b"SYNTHETIC IMAGE " + str(len(self.images)).encode())
        return {}

    def submit_video(self, prompt, duration, references, **kwargs):
        self.videos.append({"prompt": prompt, "duration": duration, "references": list(references),
                            "config": copy.deepcopy(self.active_config), "kwargs": kwargs})
        return {"id": "synthetic-voice-task-" + str(len(self.videos))}

    @staticmethod
    def evidence(video, directory):
        frame = write(Path(directory) / "frame.png", b"SYNTHETIC FRAME")
        audio = write(Path(directory) / "audio.wav", b"SYNTHETIC AUDIO")
        return {"frames": [str(frame)], "tail_frame": str(frame), "audio": str(audio),
                "sha256": core.digest(video), "sample_times": [0],
                "metadata": {"duration": 6, "has_video": True, "has_audio": True}}

    def asset(self, identifier="hero"):
        return core._find(core.load_project(self.root), "asset", identifier)

    def approve_prompt(self, identifier="hero", *, model=IMAGE_CHOICE, prompt=None):
        from norlan import asset_prompt_gate
        candidate = self.asset(identifier)["generation_prompt"]
        asset_prompt_gate.approve(self.root, identifier, candidate["id"],
                                  candidate["request_hash"], model=model, prompt=prompt)
        return candidate

    def materialize_approved_visuals(self):
        with core.transaction(self.root) as state:
            for asset in state["assets"]:
                image = write(self.root / "media" / (asset["id"] + ".png"), b"SYNTHETIC APPROVED IMAGE")
                asset.update(status="candidate", version=1, file=str(image.relative_to(self.root)),
                             sha256=core.digest(image))
        for asset in core.load_project(self.root)["assets"]:
            core.approve(self.root, "asset", asset["id"], asset["version"])

    def add_voice(self, *, legacy_submitted=False):
        voice = {"id": "synthetic-voice", "kind": "voice", "name": "合成测试声音",
            "character_id": "hero", "prompt": "中性语气、平稳清晰的成年声音", "status": "planned",
            "version": 0, "history": [], "feedback": "", "audition_text": LINE,
            "audition_source": SOURCE, "audition_reason": "合成原文中的完整中性陈述",
            "creation_input": {**asset_inputs.empty_input(), "mode": "gpt", "confirmed": True, "revision": 1}}
        if legacy_submitted:
            voice.update(status="generating", version=1, job_id="synthetic-legacy-voice")
        with core.transaction(self.root) as state:
            state["assets"].append(voice)
        if legacy_submitted:
            core.reserve(self.root, "synthetic-legacy-voice", "voice", voice["id"], 6)
            core.update_job(self.root, "synthetic-legacy-voice", state="submitted", task_id="synthetic-existing-provider-id")
        return voice["id"]

    def test_first_live_run_and_repeated_continue_prepare_without_credentials_media_jobs_or_provider(self):
        with patch.object(core, "reserve", wraps=core.reserve) as reserve:
            first = pipeline.run(self.root)
            second = pipeline.run(self.root)
        self.assertEqual(first["status"], "awaiting_asset_prompt_approval")
        self.assertEqual(second["status"], "awaiting_asset_prompt_approval")
        self.assertEqual(first["jobs"], {})
        self.assertEqual(second["jobs"], {})
        self.assertEqual(second["budget"]["reserved"], 0)
        for earlier, current in zip(first["assets"], second["assets"]):
            self.assertEqual(current["status"], "awaiting_asset_prompt_approval")
            self.assertEqual(current["version"], 0)
            self.assertEqual(current["generation_prompt"], earlier["generation_prompt"])
            self.assertTrue(current["generation_prompt"]["prompt"])
        self.ark.assert_not_called()
        self.brain.assert_not_called()
        reserve.assert_not_called()
        self.assertFalse(list((self.root / "media" / "assets").glob("*")))

    def test_only_confirmed_asset_uses_exact_prompt_selected_image_model_and_saved_references(self):
        # A synthetic local upload exercises reference binding without image APIs.
        from tests.test_asset_inputs import data_url
        core.set_asset_input(self.root, "hero", 0, 1, "custom", "合成人物参考图")
        revision = self.asset()["creation_input"]["revision"]
        core.add_asset_reference(self.root, "hero", 0, revision, data_url())
        revision = self.asset()["creation_input"]["revision"]
        core.set_asset_input(self.root, "hero", 0, revision, "custom", "合成人物参考图")
        initial_config = copy.deepcopy(core.load_project(self.root)["settings"]["ark"])
        prepared = pipeline.run(self.root)
        self.assertEqual(prepared["status"], "awaiting_asset_prompt_approval")
        exact = "  人工最终人物提示词。\n保留这份原文与参考图。\n\n"
        candidate = self.approve_prompt(prompt=exact)
        state = pipeline.run(self.root)
        self.assertEqual(len(self.images), 1)
        self.assertEqual(self.images[0]["prompt"], exact)
        self.assertEqual(self.images[0]["config"]["image_model"], IMAGE_CHOICE)
        expected_refs = [core.safe_path(self.root, ref["file"]) for ref in self.asset()["creation_input"]["references"]]
        self.assertEqual(self.images[0]["references"], expected_refs)
        self.assertEqual(self.asset()["status"], "candidate")
        self.assertEqual(self.asset()["version"], 1)
        self.assertEqual(self.asset("station")["status"], "awaiting_asset_prompt_approval")
        self.assertEqual(self.asset("station")["version"], 0)
        self.assertEqual(state["status"], "awaiting_asset_prompt_approval")
        self.assertEqual(len(state["jobs"]), 1)
        job = next(iter(state["jobs"].values()))
        self.assertEqual((job["kind"], job["item_id"]), ("image", "hero"))
        self.assertEqual(job["asset_generation"]["model"], IMAGE_CHOICE)
        self.assertEqual(job["asset_generation"]["prompt"], exact)
        self.assertEqual(job["billing"]["price"]["model"], IMAGE_CHOICE)
        self.assertNotEqual(candidate["prompt"], exact)
        self.assertEqual(state["settings"]["ark"], initial_config)
        self.provider.submit_video.assert_not_called()
        self.brain.assert_not_called()

    def test_confirmed_image_is_not_submitted_again_by_ordinary_continue(self):
        pipeline.run(self.root)
        self.approve_prompt()
        state = pipeline.run(self.root)
        before_jobs = copy.deepcopy(state["jobs"])
        before_asset = copy.deepcopy(self.asset())
        for _ in range(3):
            state = pipeline.run(self.root)
        self.assertEqual(self.provider.generate_image.call_count, 1)
        self.assertEqual(state["jobs"], before_jobs)
        self.assertEqual(self.asset(), before_asset)
        self.assertEqual(self.asset("station")["version"], 0)

    def test_rework_requires_fresh_prompt_gate_and_preserves_original_media_before_second_consent(self):
        pipeline.run(self.root)
        original_candidate = self.approve_prompt()
        pipeline.run(self.root)
        previous = self.asset()
        original_file = core.safe_path(self.root, previous["file"])
        original_bytes = original_file.read_bytes()
        original_jobs = set(core.load_project(self.root)["jobs"])
        core.reject(self.root, "asset", "hero", 1, "合成返修意见：改成深蓝色衬衫，保留人物五官。")
        state = pipeline.run(self.root)
        revised = self.asset()
        self.assertEqual(revised["status"], "awaiting_asset_prompt_approval")
        self.assertEqual(revised["version"], 1)
        self.assertNotEqual(revised["generation_prompt"]["id"], original_candidate["id"])
        self.assertEqual(revised["generation_prompt"]["target_version"], 2)
        self.assertEqual(set(state["jobs"]), original_jobs)
        self.assertEqual(self.provider.generate_image.call_count, 1)
        self.assertEqual(original_file.read_bytes(), original_bytes)
        exact = "  合成第二版人工定稿。\n深蓝衬衫。\n"
        self.approve_prompt(prompt=exact)
        state = pipeline.run(self.root)
        self.assertEqual(self.provider.generate_image.call_count, 2)
        self.assertEqual(self.images[-1]["prompt"], exact)
        self.assertEqual((self.asset()["status"], self.asset()["version"]), ("candidate", 2))
        self.assertEqual(original_file.read_bytes(), original_bytes)
        self.assertTrue(any(item.get("file") == previous["file"] and item.get("version") == 1 for item in self.asset()["history"]))
        self.assertEqual(len(state["jobs"]), 2)
        self.assertEqual(self.asset("station")["version"], 0)

    def test_voice_prepares_final_audition_before_consent_without_provider_or_media_job(self):
        self.materialize_approved_visuals()
        voice_id = self.add_voice()
        state = pipeline.run(self.root)
        voice = self.asset(voice_id)
        self.assertEqual(state["status"], "awaiting_asset_prompt_approval")
        self.assertEqual(voice["version"], 0)
        self.assertIn(LINE, voice["generation_prompt"]["prompt"])
        self.assertIn("中性语气", voice["generation_prompt"]["prompt"])
        self.assertGreaterEqual(voice["generation_prompt"]["duration"], 4)
        self.assertEqual(state["jobs"], {})
        self.ark.assert_not_called()
        self.brain.assert_not_called()

    def test_voice_confirmation_reaches_submit_with_exact_prompt_selected_model_and_approved_character(self):
        self.materialize_approved_visuals()
        original_visuals = copy.deepcopy(core.load_project(self.root)["assets"])
        voice_id = self.add_voice()
        original_settings = copy.deepcopy(core.load_project(self.root)["settings"])
        pipeline.run(self.root)
        exact = "  合成试听最终定稿。\n中性语气：" + LINE + "\n\n"
        candidate = self.approve_prompt(voice_id, model=VOICE_CHOICE, prompt=exact)
        state = pipeline.run(self.root)
        self.assertEqual(len(self.videos), 1)
        self.assertEqual(self.videos[0]["prompt"], exact)
        self.assertEqual(self.videos[0]["config"]["video_model"], VOICE_CHOICE)
        self.assertEqual(self.videos[0]["duration"], candidate["duration"])
        self.assertEqual(self.videos[0]["references"], [core.safe_path(self.root, original_visuals[0]["file"])])
        self.assertEqual((self.asset(voice_id)["status"], self.asset(voice_id)["version"]), ("candidate", 1))
        self.assertTrue(self.asset(voice_id)["audio_file"])
        self.assertEqual(state["assets"][:2], original_visuals)
        self.assertEqual(state["settings"], original_settings)
        self.assertEqual(len(state["jobs"]), 1)
        job = next(iter(state["jobs"].values()))
        self.assertEqual(job["video_request"]["video_model"], VOICE_CHOICE)
        self.assertEqual(job["asset_generation"]["prompt"], exact)
        self.assertEqual(job["billing"]["price"]["model"], VOICE_CHOICE)
        for _ in range(2):
            pipeline.run(self.root)
        self.provider.submit_video.assert_called_once()
        self.provider.generate_image.assert_not_called()
        self.brain.assert_not_called()

    def test_confirmed_image_definite_failure_stops_without_reusing_consent_to_retry(self):
        from norlan.providers import ProviderError
        pipeline.run(self.root)
        self.approve_prompt()
        self.provider.generate_image.side_effect = ProviderError("合成供应商明确拒绝", failure={
            "stage": "submission", "message": "合成供应商明确拒绝", "uncertain": False,
            "provider_code": "Synthetic.InvalidInput", "http_status": 400})
        failed = pipeline.run(self.root)
        self.assertEqual(failed["status"], "asset_generation_failed")
        self.assertEqual(self.asset()["status"], "generation_failed")
        failed_job = copy.deepcopy(next(iter(failed["jobs"].values())))
        self.assertEqual(failed_job["state"], "failed")
        for _ in range(3):
            state = pipeline.run(self.root)
            self.assertEqual(state["status"], "asset_generation_failed")
        self.provider.generate_image.assert_called_once()
        self.assertEqual(len(state["jobs"]), 1)
        self.assertEqual(next(iter(state["jobs"].values())), failed_job)
        self.assertEqual(self.asset("station")["version"], 0)
        self.provider.submit_video.assert_not_called()
        self.brain.assert_not_called()

    def test_legacy_submitted_voice_recovers_original_task_without_retroactive_prompt_gate_or_new_post(self):
        from norlan import asset_prompt_gate
        self.materialize_approved_visuals()
        voice_id = self.add_voice(legacy_submitted=True)
        with patch.object(asset_prompt_gate, "prepare", side_effect=AssertionError(
                "Existing submitted jobs must never receive a fresh approval gate")) as prepare, \
                patch.object(core, "reserve", wraps=core.reserve) as reserve:
            state = pipeline.run(self.root)
        self.assertEqual(set(state["jobs"]), {"synthetic-legacy-voice"})
        self.assertEqual(state["jobs"]["synthetic-legacy-voice"]["accounting"], "spent")
        self.assertEqual((self.asset(voice_id)["status"], self.asset(voice_id)["version"]), ("candidate", 1))
        self.assertNotIn("generation_prompt", self.asset(voice_id))
        self.assertEqual(self.asset(voice_id)["job_id"], "synthetic-legacy-voice")
        self.provider.poll_video.assert_called_once_with("synthetic-existing-provider-id")
        self.provider.download.assert_called_once()
        self.provider.submit_video.assert_not_called()
        self.provider.generate_image.assert_not_called()
        self.provider.preflight.assert_not_called()
        self.brain.assert_not_called()
        reserve.assert_not_called()
        prepare.assert_not_called()

    def test_budget_refusal_retains_same_consent_version_and_resumes_once_after_limit_increase(self):
        pipeline.run(self.root)
        exact = "  合成预算恢复测试的人工最终提示词。\n原文与目标版本保持。  "
        self.approve_prompt(prompt=exact)
        approved = copy.deepcopy(self.asset()["generation_prompt"])
        decision = copy.deepcopy(self.asset()["generation_prompt_approval"])
        core.configure(self.root, {}, limit=0.01)
        with patch.object(core, "reserve", wraps=core.reserve) as reserve:
            for _ in range(3):
                state = pipeline.run(self.root)
                asset = self.asset()
                self.assertEqual(state["status"], "blocked")
                self.assertIn("预算不足", state["last_error"])
                self.assertEqual((asset["status"], asset["version"]), ("asset_prompt_approved", 0))
                self.assertEqual(asset["generation_prompt"], approved)
                self.assertEqual(asset["generation_prompt_approval"], decision)
                self.assertNotIn("generation_request", asset)
                self.assertNotIn("job_id", asset)
                self.assertEqual(state["jobs"], {})
            reserve.assert_not_called()
        self.ark.assert_not_called()
        self.provider.generate_image.assert_not_called()
        self.provider.submit_video.assert_not_called()
        core.configure(self.root, {}, limit=200)
        # Raising the existing budget resumes the already authorized request;
        # there is no second human approval or hidden replacement candidate.
        for _ in range(3):
            state = pipeline.run(self.root)
        self.assertEqual((self.asset()["status"], self.asset()["version"]), ("candidate", 1))
        self.assertEqual(self.asset()["generation_request"], approved)
        self.assertEqual(self.asset()["generation_prompt_approval"], decision)
        self.assertEqual(self.images[0]["prompt"], exact)
        self.assertEqual(len(state["jobs"]), 1)
        self.provider.generate_image.assert_called_once()
        self.provider.submit_video.assert_not_called()
        self.brain.assert_not_called()

    def test_local_voice_transport_refusal_retains_consent_and_resumes_same_version_once(self):
        from norlan.providers import ProviderError
        self.materialize_approved_visuals()
        voice_id = self.add_voice()
        pipeline.run(self.root)
        exact = "  合成传输恢复测试。\n平稳朗读：" + LINE + "\n"
        self.approve_prompt(voice_id, model=VOICE_CHOICE, prompt=exact)
        approved = copy.deepcopy(self.asset(voice_id)["generation_prompt"])
        decision = copy.deepcopy(self.asset(voice_id)["generation_prompt_approval"])
        refusal = ProviderError("合成测试：本机参考素材传输校验未通过。",
                                uncertain=False, stage="local_validation")
        with patch.object(pipeline, "_prepare_video_transport", side_effect=refusal) as transport, \
                patch.object(core, "reserve", wraps=core.reserve) as reserve:
            for _ in range(3):
                state = pipeline.run(self.root)
                asset = self.asset(voice_id)
                self.assertEqual(state["status"], "blocked")
                self.assertIn("本机参考素材传输校验未通过", state["last_error"])
                self.assertEqual((asset["status"], asset["version"]), ("asset_prompt_approved", 0))
                self.assertEqual(asset["generation_prompt"], approved)
                self.assertEqual(asset["generation_prompt_approval"], decision)
                self.assertNotIn("generation_request", asset)
                self.assertNotIn("job_id", asset)
                self.assertEqual(state["jobs"], {})
            self.assertEqual(transport.call_count, 3)
            self.assertEqual(transport.call_args.kwargs["source_manifest_override"],
                             approved["reference_manifest"])
            reserve.assert_not_called()
        self.provider.submit_video.assert_not_called()
        self.provider.generate_image.assert_not_called()
        # Removing the synthetic local refusal reuses exact same consent and
        # actual reference manifest; no voice selection or prompt rewrite runs.
        for _ in range(3):
            state = pipeline.run(self.root)
        self.assertEqual((self.asset(voice_id)["status"], self.asset(voice_id)["version"]), ("candidate", 1))
        self.assertEqual(self.asset(voice_id)["generation_request"], approved)
        self.assertEqual(self.asset(voice_id)["generation_prompt_approval"], decision)
        self.assertEqual(self.videos[0]["prompt"], exact)
        self.assertEqual(self.videos[0]["duration"], approved["duration"])
        self.assertEqual(len(state["jobs"]), 1)
        self.provider.submit_video.assert_called_once()
        self.provider.generate_image.assert_not_called()
        self.brain.assert_not_called()


if __name__ == "__main__":
    unittest.main()
