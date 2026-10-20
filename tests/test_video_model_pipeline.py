"""Synthetic full provider payload/model recovery integration; no real APIs."""
import base64
import copy
from pathlib import Path
import time
import unittest
from unittest.mock import patch

from norlan import core, pipeline
from norlan.billing import estimate_reservation
from norlan.providers import ArkProvider, ProviderError
import test_prompt_gate as prompt_fixture
from test_pipeline import write
from test_video_model_gate import FAST, MINI, STANDARD, V25


REAL_ARK_FACTORY = pipeline._ark
REAL_ARK_PROVIDER = ArkProvider


class VideoModelPipelineTests(unittest.TestCase):
    answer = prompt_fixture.PromptGateTests.answer
    amendment = prompt_fixture.PromptGateTests.amendment

    def setUp(self):
        prompt_fixture.PromptGateTests.setUp(self)
        core.configure(self.root, {"ark": {"video_model": FAST, "resolution": "720p", "generate_audio": True}})
        # Replace only the fixture's _ark mock. Construct the real adapter, with
        # its transport, downloads and key lookup all intercepted locally.
        self.enterContext(patch.object(pipeline, "_ark", wraps=REAL_ARK_FACTORY))
        self.enterContext(patch.object(REAL_ARK_PROVIDER, "_key", return_value="SYNTHETIC-NOT-A-REAL-KEY"))
        self.enterContext(patch("norlan.providers._api_open", side_effect=AssertionError("Real API forbidden")))
        self.enterContext(patch("norlan.providers.urlopen", side_effect=AssertionError("Real media IO forbidden")))
        self.enterContext(patch.object(REAL_ARK_PROVIDER, "_request", autospec=True, side_effect=self.request))
        self.enterContext(patch.object(REAL_ARK_PROVIDER, "download", autospec=True, side_effect=self.download))
        self.enterContext(patch("norlan.providers.ArkProvider", side_effect=self.construct))
        self.configurations = []
        self.requests = []
        self.tasks = {}
        self.interrupt_poll_once = False

    def construct(self, config):
        self.configurations.append(copy.deepcopy(config))
        return REAL_ARK_PROVIDER(config)

    def request(self, provider, method, path, payload=None):
        self.requests.append({"method": method, "path": path,
                              "config": copy.deepcopy(provider.config), "payload": copy.deepcopy(payload)})
        if method == "POST":
            self.assertEqual(path, "/contents/generations/tasks")
            task_id = "synthetic-model-task-" + str(len(self.tasks) + 1)
            self.tasks[task_id] = copy.deepcopy(payload)
            return {"id": task_id}, "synthetic-submit"
        self.assertEqual(method, "GET")
        if self.interrupt_poll_once:
            self.interrupt_poll_once = False
            raise ProviderError("合成测试：暂时停止查询原任务", stage="poll")
        task_id = path.rsplit("/", 1)[-1]
        submitted = self.tasks[task_id]
        return {"id": task_id, "status": "succeeded", "model": submitted["model"],
                "created_at": int(time.time()), "duration": submitted["duration"],
                "resolution": submitted["resolution"], "usage": {"completion_tokens": 100000},
                "content": {"video_url": "https://example.test/" + task_id + ".mp4?token=SYNTHETIC",
                            "last_frame_url": "https://example.test/" + task_id + ".jpg?token=SYNTHETIC"}}, "synthetic-poll"

    def download(self, provider, url, output):
        name = url.split("?", 1)[0].rsplit("/", 1)[-1]
        content = ((b"\xff\xd8\xffSYNTHETIC OFFICIAL TAIL " if name.endswith(".jpg")
                    else b"SYNTHETIC ORIGINAL VIDEO ") + name.encode())
        return write(output, content)

    def shot(self, index=0):
        return core.load_project(self.root)["episodes"][0]["shots"][index]

    def job(self, index=0):
        return core.load_project(self.root)["jobs"][self.shot(index)["job_id"]]

    def prepare(self, index=0):
        self.assertEqual(pipeline.run(self.root)["status"], "awaiting_prompt_approval")
        return copy.deepcopy(self.shot(index)["prompt_candidate"])

    def approve(self, candidate, model=STANDARD, index=0, text=None):
        shot_id = self.shot(index)["id"]
        if text is None:
            return core.approve_prompt(self.root, shot_id, candidate["id"], candidate["request_hash"], video_model=model)
        return core.edit_prompt_and_approve(self.root, shot_id, candidate["id"], candidate["request_hash"], text,
                                           video_model=model)

    def posts(self):
        return [request for request in self.requests if request["method"] == "POST"]

    def assert_job_model(self, model, index=0, *, rate, video_input=False):
        job = self.job(index)
        self.assertEqual(job["video_request"]["video_model"], model)
        self.assertEqual(set(job["video_request"]),
                         {"video_model", "resolution", "generate_audio", "video_watermark", "ratio"})
        self.assertEqual(job["billing"]["price"]["model"], model)
        self.assertEqual(job["billing"]["price"]["rate"], rate)
        self.assertEqual(job["billing"]["price"]["has_video_input"], video_input)
        self.assertEqual(job["billing"]["model"], model)
        self.assertEqual(job["billing"]["usage_cost"], rate / 10)
        tail = job["official_tail"]
        self.assertEqual(tail["source"], "ark_official_tail")
        self.assertEqual(tail["task_id"], job["task_id"])
        self.assertEqual(tail["video_sha256"], self.shot(index)["sha256"])
        self.assertEqual(tail["sha256"], core.digest(self.root / tail["file"]))
        self.assertNotIn("token=SYNTHETIC", (self.root / "project.json").read_text())
        self.assertNotIn("SYNTHETIC-NOT-A-REAL-KEY", (self.root / "project.json").read_text())

    def test_ordinary_approval_constructs_selected_provider_and_posts_exact_model(self):
        original = self.prepare()
        calls = self.brain.call_count
        self.approve(original, STANDARD)
        self.assertFalse(self.posts())
        self.assertEqual(pipeline.run(self.root)["status"], "review_required")
        self.assertEqual(len(self.posts()), 1)
        post = self.posts()[0]
        self.assertEqual(post["config"]["video_model"], STANDARD)
        payload = post["payload"]
        self.assertEqual(payload["model"], STANDARD)
        self.assertEqual(payload["content"][0], {"type": "text", "text": original["prompt"]})
        self.assertEqual(payload["duration"], self.shot()["duration"])
        self.assertTrue(payload["return_last_frame"])
        self.assertFalse(payload["watermark"])
        self.assertEqual(len([item for item in payload["content"] if item["type"] == "image_url"]),
                         len(original["reference_manifest"]["images"]))
        self.assertEqual(self.brain.call_count, calls)
        self.assert_job_model(STANDARD, rate=46)

    def test_manual_approval_posts_exact_human_text_on_selected_model_without_gpt(self):
        original = self.prepare()
        raw = "  完整人工定稿。\n不调用 GPT 再写一遍。  "
        self.brain.reset_mock()
        self.brain.side_effect = AssertionError("Model selection/manual adoption must not call GPT")
        self.approve(original, MINI, text=raw)
        self.assertFalse(self.posts())
        self.assertEqual(pipeline.run(self.root)["status"], "review_required")
        self.assertEqual(len(self.posts()), 1)
        payload = self.posts()[0]["payload"]
        self.assertEqual(payload["model"], MINI)
        self.assertEqual(payload["content"][0]["text"], raw)
        self.assert_job_model(MINI, rate=23)
        self.brain.assert_not_called()

    def test_resume_queries_original_model_with_locked_options_and_never_posts_again(self):
        original = self.prepare()
        self.approve(original, STANDARD)
        self.interrupt_poll_once = True
        self.assertEqual(pipeline.run(self.root)["status"], "blocked")
        original_job = copy.deepcopy(self.job())
        original_approval = copy.deepcopy(self.shot()["prompt_approval"])
        self.assertEqual(original_job["state"], "submitted")
        self.assertEqual(len(self.posts()), 1)
        core.configure(self.root, {"ark": {"video_model": MINI, "resolution": "480p",
                                           "generate_audio": False, "video_watermark": True}})
        before_requests = len(self.requests)
        self.assertEqual(pipeline.run(self.root)["status"], "review_required")
        self.assertEqual(len(self.posts()), 1)
        new_requests = self.requests[before_requests:]
        self.assertTrue(new_requests)
        self.assertTrue(all(request["method"] == "GET" for request in new_requests))
        for request in new_requests:
            self.assertEqual(request["config"]["video_model"], STANDARD)
            self.assertEqual(request["config"]["resolution"], "720p")
            self.assertIs(request["config"]["generate_audio"], True)
            self.assertIs(request["config"]["video_watermark"], False)
            self.assertTrue(request["path"].endswith(original_job["task_id"]))
        self.assertEqual(self.job()["id"], original_job["id"])
        self.assertEqual(self.job()["video_request"], original_job["video_request"])
        self.assertEqual(self.shot()["prompt_approval"], original_approval)
        self.assert_job_model(STANDARD, rate=46)

    def test_adjacent_segments_can_use_different_models_and_keep_official_tail_reference(self):
        self.approve(self.prepare(), STANDARD)
        self.assertEqual(pipeline.run(self.root)["status"], "review_required")
        first = self.shot()
        original_bytes = (self.root / first["file"]).read_bytes()
        tail = self.root / self.job()["official_tail"]["file"]
        core.approve(self.root, "shot", first["id"], first["version"])
        next_candidate = self.prepare(1)
        self.assertEqual(next_candidate["video_model"], FAST)
        self.approve(next_candidate, MINI, index=1)
        self.assertEqual(pipeline.run(self.root)["status"], "review_required")
        self.assertEqual([request["payload"]["model"] for request in self.posts()], [STANDARD, MINI])
        references = [item for item in self.posts()[1]["payload"]["content"] if item["type"] == "image_url"]
        last_bytes = base64.b64decode(references[-1]["image_url"]["url"].split(",", 1)[1])
        self.assertEqual(last_bytes, tail.read_bytes())
        self.assertNotEqual(last_bytes, (self.root / first["evidence"]["tail_frame"]).read_bytes())
        self.assertEqual((self.root / first["file"]).read_bytes(), original_bytes)
        self.assertEqual(self.shot()["sha256"], first["sha256"])
        self.assert_job_model(MINI, 1, rate=23)
        current = self.shot(1)
        current_bytes = (self.root / current["file"]).read_bytes()
        core.reject(self.root, "shot", current["id"], current["version"], "合成测试：修改人物转身动作")
        archived = self.shot(1)["history"][-1]
        self.assertEqual(archived["video_model"], MINI)
        self.assertEqual(archived["job_id"], current["job_id"])
        self.assertEqual((self.root / archived["file"]).read_bytes(), current_bytes)
        self.assertEqual(self.prepare(1)["video_model"], MINI)
        self.assertEqual(len(self.posts()), 2)

    def test_original_video_reference_uses_selected_price_and_preserves_source_model(self):
        self.approve(self.prepare(), STANDARD)
        self.assertEqual(pipeline.run(self.root)["status"], "review_required")
        first = self.shot()
        core.approve(self.root, "shot", first["id"], first["version"])
        with core.transaction(self.root) as state:
            second = state["episodes"][0]["shots"][1]
            second["continuity_reference"] = {"mode": "original_video", "revision": second.get("revision", 0),
                "source_shot_id": first["id"], "source_sha256": first["sha256"]}
        self.approve(self.prepare(1), MINI, index=1)
        self.assertEqual(pipeline.run(self.root)["status"], "review_required")
        payload = self.posts()[1]["payload"]
        self.assertEqual(payload["model"], MINI)
        videos = [item for item in payload["content"] if item["type"] == "video_url"]
        self.assertEqual(len(videos), 1)
        self.assertEqual(videos[0]["role"], "reference_video")
        self.assertIn(self.job()["task_id"], videos[0]["video_url"]["url"])
        self.assertEqual(self.job(1)["reference_video_duration"], 4)
        self.assertEqual(self.job(1)["amount"], 0)
        self.assertTrue(self.job(1)["cost_track"])
        self.assertEqual(self.job()["billing"]["price"]["model"], STANDARD)
        self.assert_job_model(MINI, 1, rate=14, video_input=True)

    def test_seedance_25_retains_twenty_second_duration_and_official_tail(self):
        with core.transaction(self.root) as state:
            state["episodes"][0]["shots"][0]["duration"] = 20
        self.approve(self.prepare(), V25)
        self.assertEqual(pipeline.run(self.root)["status"], "review_required")
        self.assertEqual(len(self.posts()), 1)
        self.assertEqual(self.posts()[0]["payload"]["duration"], 20)
        self.assertEqual(self.shot()["duration"], 20)
        self.assert_job_model(V25, rate=70)

    def test_implicit_provider_defaults_are_recorded_as_actual_request_options(self):
        with core.transaction(self.root) as state:
            for key in ("resolution", "generate_audio", "video_watermark"):
                state["settings"]["ark"].pop(key, None)
        self.approve(self.prepare(), STANDARD)
        self.assertEqual(pipeline.run(self.root)["status"], "review_required")
        payload = self.posts()[0]["payload"]
        for request_key, payload_key in (("resolution", "resolution"), ("generate_audio", "generate_audio"),
                                         ("video_watermark", "watermark")):
            self.assertEqual(self.job()["video_request"].get(request_key), payload[payload_key])

    def test_zero_unapproved_budget_keeps_selected_consent_and_submits_once_without_gpt(self):
        original = self.prepare()
        with core.transaction(self.root) as state:
            state["budget"].update(limit=0, approved=False)
            state["settings"]["caps"]["video_second"] = 0
        for _ in range(2):
            self.assertEqual(pipeline.run(self.root)["status"], "awaiting_prompt_approval")
        self.assertFalse(self.posts())
        self.approve(original, STANDARD)
        candidate = copy.deepcopy(self.shot()["prompt_candidate"])
        decision = copy.deepcopy(self.shot()["prompt_approval"])
        self.brain.reset_mock()
        self.brain.side_effect = AssertionError("Generation must not rewrite approved text")
        for _ in range(2):
            self.assertEqual(pipeline.run(self.root)["status"], "review_required")
            self.assertEqual(self.shot()["status"], "review_required")
            self.assertEqual(self.shot()["version"], candidate["version"])
            self.assertEqual(self.shot()["prompt_candidate"], candidate)
            self.assertEqual(self.shot()["prompt_approval"], decision)
        self.assertEqual(len(self.posts()), 1)
        self.assertEqual(self.posts()[0]["payload"]["content"][0]["text"], candidate["prompt"])
        self.assert_job_model(STANDARD, rate=46)
        self.brain.assert_not_called()

    def test_selected_25_model_records_own_quote_without_enforcing_legacy_low_budget(self):
        original = self.prepare()
        duration = self.shot()["duration"]
        config = {"video_model": FAST, "resolution": "720p", "ratio": "16:9"}
        fast_quote = estimate_reservation(config, duration)
        selected_quote = estimate_reservation({**config, "video_model": V25}, duration)
        self.assertGreater(selected_quote, fast_quote)
        self.approve(original, V25)
        candidate = copy.deepcopy(self.shot()["prompt_candidate"])
        decision = copy.deepcopy(self.shot()["prompt_approval"])
        core.configure(self.root, limit=0.01)
        self.brain.reset_mock()
        self.brain.side_effect = AssertionError("An old limit must not rewrite the selected prompt")
        for _ in range(2):
            self.assertEqual(pipeline.run(self.root)["status"], "review_required")
        self.assertEqual(len(self.posts()), 1)
        self.assertEqual(self.job()["amount"], selected_quote)
        self.assertTrue(self.job()["cost_track"])
        self.assertEqual(self.shot()["prompt_candidate"], candidate)
        self.assertEqual(self.shot()["prompt_approval"], decision)
        self.assertEqual(self.posts()[0]["payload"]["content"][0]["text"], candidate["prompt"])
        self.assert_job_model(V25, rate=70)
        self.brain.assert_not_called()


if __name__ == "__main__":
    unittest.main()
