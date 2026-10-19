import base64
import io
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch
from urllib.error import HTTPError

from norlan.providers import ArkProvider, ProviderError
from norlan.brain import Brain, BrainError, COMMAND_SCHEMA, _codex_failure, _codex_version, _validate, resolve_codex, validate_plan


class Response(io.BytesIO):
    def __init__(self, body, headers=None, url="https://example.test/output.mp4"):
        super().__init__(body if isinstance(body, bytes) else json.dumps(body).encode())
        self.headers = headers or {}
        self.url = url

    def geturl(self):
        return self.url


def plan():
    return {"title": "测试", "style": "二维动画", "ratio": "16:9",
            "assets": [{"id": "hero", "kind": "character", "name": "主角", "prompt": "蓝色外套", "character_id": "",
                        "audition_text": "", "audition_reason": "", "audition_source": "",
                        "voice_only": False, "audition_speaker": ""}],
            "episodes": [{"id": "E001", "title": "开端", "summary": "寻找信封", "music_prompt": "钢琴淡入",
                          "shots": [{"id": "S001", "duration": 5, "prompt": "向右走", "dialogue": "找到你了。",
                                     "asset_ids": ["hero"], "scene_id": "room", "continuity_notes": "右手拿信封",
                                     "subshots": [{"start": 0, "end": 5, "prompt": "中景向右跟拍，右手拿信封走到门边。"}]}]}]}


class ProviderTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name)
        self.image = self.path / "ref.png"
        self.image.write_bytes(b"image-data")
        self.audio = self.path / "voice.wav"
        self.audio.write_bytes(b"audio-data")
        self.env = patch.dict(os.environ, {"ARK_API_KEY": "super-secret"})
        self.env.start()
        self.addCleanup(self.env.stop)
        self.api = ArkProvider({"image_model": "seedream-test", "video_model": "seedance-test"})

    def test_video_uploads_references_with_correct_roles(self):
        with patch("norlan.providers._api_open", return_value=Response({"id": "task-1"})) as http, \
             patch.object(self.api, "_audio_duration", return_value=4):
            result = self.api.submit_video("继续动作", 5, [self.image], audio_references=[self.audio])
        self.assertEqual(result["id"], "task-1")
        payload = json.loads(http.call_args.args[0].data)
        self.assertEqual(payload["duration"], 5)
        self.assertEqual([x.get("role") for x in payload["content"]], [None, "reference_image", "reference_audio"])
        self.assertTrue(payload["content"][1]["image_url"]["url"].startswith("data:image/png;base64,"))
        self.assertNotIn("seed", payload)

    def test_submit_timeout_is_uncertain_and_never_retried(self):
        with patch("norlan.providers._api_open", side_effect=TimeoutError("super-secret")) as http:
            with self.assertRaises(ProviderError) as caught:
                self.api.submit_video("hello", 5, [])
        self.assertEqual(http.call_count, 1)
        self.assertTrue(caught.exception.uncertain)
        self.assertNotIn("super-secret", str(caught.exception))

    def test_http_500_uncertain_400_definite(self):
        for code, uncertain in [(500, True), (400, False), (408, True), (429, False)]:
            with self.subTest(code=code), patch("norlan.providers._api_open", side_effect=HTTPError("https://test", code, "secret", {}, None)):
                with self.assertRaises(ProviderError) as caught:
                    self.api.submit_video("hello", 5, [])
                self.assertEqual(caught.exception.uncertain, uncertain)
                self.assertNotIn("secret", str(caught.exception))

    def test_missing_task_id_is_uncertain(self):
        with patch("norlan.providers._api_open", return_value=Response({"unexpected": True})):
            with self.assertRaises(ProviderError) as caught:
                self.api.submit_video("hello", 5, [])
        self.assertTrue(caught.exception.uncertain)

    def test_input_constraints_block_before_network(self):
        with patch("norlan.providers._api_open") as http:
            cases = [("hello", 5.5, [], None, []), ("hello", 5, [self.image] * 10, None, []),
                     ("hello", 5, [], None, [self.audio]), ("hello", 5, [self.image], self.image, [])]
            for prompt, duration, refs, previous, audios in cases:
                with self.assertRaises(ProviderError):
                    self.api.submit_video(prompt, duration, refs, previous_video=previous, audio_references=audios)
        http.assert_not_called()

    def test_audio_duration_budget(self):
        with patch.object(self.api, "_audio_duration", return_value=8), patch("norlan.providers._api_open") as http:
            with self.assertRaises(ProviderError):
                self.api.submit_video("hello", 5, [self.image], audio_references=[self.audio] * 2)
        http.assert_not_called()

    def test_image_atomic_save_and_reference(self):
        output = self.path / "asset.png"
        with patch("norlan.providers._api_open", return_value=Response({"data": [{"b64_json": base64.b64encode(b"new-image").decode()}]})) as http:
            result = self.api.generate_image("character", output, [self.image])
        self.assertEqual(output.read_bytes(), b"new-image")
        self.assertEqual(result["output"], str(output))
        self.assertEqual(json.loads(http.call_args.args[0].data)["sequential_image_generation"], "disabled")

    def test_seedream_flash_and_pro_use_supported_single_image_request(self):
        for model in ("doubao-seedream-5-0-flash-260915", "doubao-seedream-5-0-pro-260628"):
            with self.subTest(model=model):
                api = ArkProvider({"image_model": model})
                output = self.path / "asset.png"
                response = {"data": [{"b64_json": base64.b64encode(b"generated-image").decode()}]}
                with patch("norlan.providers._api_open", return_value=Response(response)) as http:
                    api.generate_image("同一人物的正面肖像", output, [self.image])
                payload = json.loads(http.call_args.args[0].data)
                self.assertEqual(payload["model"], model)
                self.assertNotIn("sequential_image_generation", payload)
                self.assertEqual(payload["response_format"], "b64_json")
                self.assertEqual(payload["output_format"], "png")
                self.assertFalse(payload["stream"])
                self.assertEqual(payload["size"], "2K")
                self.assertEqual(payload["image"], ["data:image/png;base64," + base64.b64encode(b"image-data").decode()])
                self.assertEqual(output.read_bytes(), b"generated-image")
                self.assertEqual(http.call_count, 1)

    def test_older_seedream_models_keep_explicit_single_image_mode(self):
        for model in ("doubao-seedream-4-0-250828", "doubao-seedream-4-5-251128",
                      "doubao-seedream-5-0-260128", "ep-previously-configured"):
            with self.subTest(model=model):
                api = ArkProvider({"image_model": model})
                response = {"data": [{"b64_json": base64.b64encode(b"generated-image").decode()}]}
                with patch("norlan.providers._api_open", return_value=Response(response)) as http:
                    api.generate_image("单张人物肖像", self.path / "asset.png")
                payload = json.loads(http.call_args.args[0].data)
                self.assertEqual(payload["sequential_image_generation"], "disabled")
                self.assertNotIn("output_format", payload)

    def test_known_image_video_model_mismatch_is_rejected_before_network(self):
        for kind, model, expected in (("image", "doubao-seedance-2-0-fast-260128", "image_model"),
                                      ("video", "doubao-seedream-5-0-flash-260915", "video_model")):
            with self.subTest(kind=kind), patch("norlan.providers._api_open") as http:
                api = ArkProvider({f"{kind}_model": model})
                with self.assertRaisesRegex(ProviderError, expected):
                    api.preflight(kind)
                with self.assertRaisesRegex(ProviderError, expected):
                    if kind == "image":
                        api.generate_image("角色", self.path / "asset.png")
                    else:
                        api.submit_video("继续动作", 5, [])
                http.assert_not_called()

    def test_seedream_flash_timeout_is_uncertain_and_never_retried(self):
        api = ArkProvider({"image_model": "doubao-seedream-5-0-flash-260915"})
        with patch("norlan.providers._api_open", side_effect=TimeoutError("super-secret")) as http:
            with self.assertRaises(ProviderError) as caught:
                api.generate_image("单张人物肖像", self.path / "asset.png")
        self.assertEqual(http.call_count, 1)
        self.assertTrue(caught.exception.uncertain)
        self.assertNotIn("super-secret", str(caught.exception))
        self.assertFalse((self.path / "asset.png").exists())

    def test_poll_does_not_submit_again(self):
        with patch("norlan.providers._api_open", return_value=Response({"status": "succeeded", "content": {"video_url": "https://example.test/movie.mp4"}})) as http:
            result = self.api.poll_video("task-123")
        self.assertEqual(result["status"], "completed")
        self.assertEqual(http.call_args.args[0].method, "GET")

    def test_failed_download_keeps_original_and_no_auth(self):
        output = self.path / "movie.mp4"
        output.write_bytes(b"original")
        with patch("norlan.providers.urlopen", return_value=Response(b"short", {"Content-Length": "100"})) as http:
            with self.assertRaises(ProviderError):
                self.api.download("https://example.test/movie.mp4", output)
        self.assertEqual(output.read_bytes(), b"original")
        self.assertNotIn("Authorization", http.call_args.args[0].headers)
        self.assertEqual(list(self.path.glob(".norlan-download-*")), [])


class BrainTests(unittest.TestCase):
    def test_voice_plan_requires_real_character_and_same_shot_reference(self):
        valid = plan()
        valid["assets"].append({"id": "hero_voice", "kind": "voice", "name": "主角声音", "prompt": "清晰低音", "character_id": "hero",
                                "audition_text": "", "audition_source": "", "audition_reason": "原文未找到可稳定试音的完整台词"})
        valid["episodes"][0]["shots"][0]["asset_ids"].append("hero_voice")
        validate_plan(valid)
        valid["assets"][-1]["character_id"] = "missing"
        with self.assertRaises(BrainError):
            validate_plan(valid)
        valid["assets"][-1]["character_id"] = "hero"
        valid["episodes"][0]["shots"][0]["asset_ids"] = ["hero_voice"]
        with self.assertRaises(BrainError):
            validate_plan(valid)

    def test_plan_references_and_durations_validated(self):
        self.assertEqual(validate_plan(plan())["title"], "测试")
        invalid = plan()
        invalid["episodes"][0]["shots"][0]["asset_ids"] = ["unknown"]
        with self.assertRaises(BrainError):
            validate_plan(invalid)
        invalid = plan()
        invalid["episodes"][0]["shots"][0]["duration"] = True
        with self.assertRaises(BrainError):
            validate_plan(invalid)

    def test_plan_duplicate_ids_and_unknown_keys_rejected(self):
        invalid = plan()
        invalid["episodes"][0]["shots"][0]["id"] = "hero"
        with self.assertRaises(BrainError):
            validate_plan(invalid)
        invalid = plan()
        invalid["shell"] = "some command"
        with self.assertRaises(BrainError):
            validate_plan(invalid)

    def test_codex_isolated_structured_and_secret_env_removed(self):
        captured = {}

        def run(args, **kwargs):
            captured.update(args=args, **kwargs)
            out = Path(args[args.index("--output-last-message") + 1])
            out.write_text(json.dumps(plan()))
            return subprocess.CompletedProcess(args, 0, "", "")

        with patch("norlan.brain.resolve_codex", return_value="/bin/codex"), \
             patch("norlan.brain.subprocess.run", side_effect=run), \
             patch.dict(os.environ, {"ARK_API_KEY": "should-not-leak", "OPENAI_API_KEY": "also-private"}):
            result = Brain({"model": "chosen-model"}).plan("原创故事")
        self.assertEqual(result["title"], "测试")
        self.assertEqual(captured["args"][-1], "-")
        self.assertIn("read-only", captured["args"])
        self.assertIn("shell_tool", captured["args"])
        self.assertNotIn("ARK_API_KEY", captured["env"])
        self.assertNotIn("OPENAI_API_KEY", captured["env"])
        self.assertNotIn("原创故事", " ".join(captured["args"]))
        self.assertFalse(Path(captured["cwd"]).exists())

    def test_review_without_actual_frames_never_claims_pass(self):
        with patch.object(Brain, "_call") as call:
            result = Brain().review({"prompt": "perfect video"}, [])
        call.assert_not_called()
        self.assertEqual(result["verdict"], "uncertain")
        self.assertFalse(result["audio_assessed"])

    def test_review_scope_is_attached(self):
        response = {"verdict": "pass", "issues": [], "observed_end_state": "人物站左侧", "repair_prompt": ""}
        with patch.object(Brain, "_call", return_value=response):
            result = Brain().review({}, [Path("frame.png")])
        self.assertEqual(result["scope"], "sampled_frames_only")
        self.assertFalse(result["audio_assessed"])

    def test_command_limit_null_and_invalid_actions(self):
        command = {"action": "status", "kind": "none", "id": "", "feedback": "", "shot_ids": [], "limit": None, "reply": "查看进度"}
        _validate(command, COMMAND_SCHEMA)
        command["action"] = "shell"
        with self.assertRaises(BrainError):
            _validate(command, COMMAND_SCHEMA)
        command["action"] = "budget"
        command["limit"] = -1
        with self.assertRaises(BrainError):
            _validate(command, COMMAND_SCHEMA)

    def test_openai_uses_responses_format_and_reports_usage(self):
        result = {"status": "completed", "usage": {"total_tokens": 12}, "output": [
            {"type": "message", "content": [{"type": "output_text", "text": json.dumps(plan())}]}]}
        brain = Brain({"provider": "openai", "model": "chosen-model"})
        with patch.dict(os.environ, {"OPENAI_API_KEY": "private"}), \
             patch("norlan.brain._api_open", return_value=Response(result)) as http:
            brain.plan("a story")
        body = json.loads(http.call_args.args[0].data)
        self.assertEqual(body["text"]["format"]["type"], "json_schema")
        self.assertFalse(body["store"])
        self.assertNotIn("tools", body)
        self.assertEqual(brain.last_usage["total_tokens"], 12)

    def test_cli_resolver_chooses_newest_version_without_model_call(self):
        _codex_version.cache_clear()
        versions = {"/path/codex": "codex-cli 0.137.0", "/app/codex": "codex-cli 0.162.0-alpha.17.2"}
        def which(name):
            return {"codex": "/path/codex", "/app/codex": "/app/codex"}.get(name)
        def run(args, **kwargs):
            self.assertEqual(args[1:], ["--version"])
            return subprocess.CompletedProcess(args, 0, versions[args[0]], "")
        with patch.dict(os.environ, {}, clear=True), patch("norlan.brain._MAC_CODEX_CANDIDATES", ("/app/codex",)), \
             patch("norlan.brain.shutil.which", side_effect=which), patch("norlan.brain.subprocess.run", side_effect=run) as probe:
            self.assertEqual(resolve_codex(), "/app/codex")
            self.assertEqual(resolve_codex(), "/app/codex")
            self.assertEqual(probe.call_count, 2)
        _codex_version.cache_clear()

    def test_cli_explicit_config_and_environment_win(self):
        with patch("norlan.brain.shutil.which", side_effect=lambda value: value), \
             patch("norlan.brain._codex_version") as versions, patch.dict(os.environ, {"NORLAN_CODEX_PATH": "/custom/env-codex"}):
            self.assertEqual(resolve_codex("/explicit/codex"), "/explicit/codex")
            self.assertEqual(resolve_codex(), "/custom/env-codex")
        versions.assert_not_called()

    def test_cli_invalid_versions_are_rejected_and_cached(self):
        _codex_version.cache_clear()
        with patch("norlan.brain.subprocess.run", return_value=subprocess.CompletedProcess([], 0, "not a version", "")) as probe:
            self.assertIsNone(_codex_version("/invalid/codex"))
            self.assertIsNone(_codex_version("/invalid/codex"))
        self.assertEqual(probe.call_count, 1)
        _codex_version.cache_clear()

    def test_cli_failure_messages_are_specific_and_sanitized(self):
        cases = [("unknown variant `max` while parsing model cache secret-token", "版本"),
                 ("model is not supported by this account secret-token", "已选模型"),
                 ("Unauthorized token has expired secret-token", "登录"),
                 ("network connection reset secret-token", "网络")]
        for error, expected in cases:
            with self.subTest(error=error):
                message = str(_codex_failure(1, error))
                self.assertIn(expected, message)
                self.assertNotIn("secret-token", message)
        self.assertNotIn("登录", str(_codex_failure(1, cases[1][0])))

    def test_caption_correction_changes_text_but_preserves_timing_and_metadata(self):
        original = [{"start": 0.2, "end": 1.3, "text": "他回来啦", "source": "asr"}]
        corrected = {"segments": [{"start": 0.2, "end": 1.3, "text": "她回来啦"}]}
        with patch.object(Brain, "_call", return_value=corrected):
            result = Brain().correct_subtitles(original, "把他改为她")
        self.assertEqual(result[0], {**original[0], "text": "她回来啦"})
        self.assertEqual(original[0]["text"], "他回来啦")

    def test_caption_time_or_count_changes_are_rejected(self):
        original = [{"start": 0, "end": 1, "text": "字幕"}]
        for corrected in ([{"start": 0, "end": 2, "text": "修正"}], []):
            with patch.object(Brain, "_call", return_value={"segments": corrected}):
                with self.assertRaises(BrainError):
                    Brain().correct_subtitles(original, "修正文字")


if __name__ == "__main__":
    unittest.main()
