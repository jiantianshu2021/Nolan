"""Synthetic metering regressions; all generation and download IO is mocked."""

from contextlib import ExitStack
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from norlan import core, pipeline
from norlan.billing import IMAGE_MODEL, VIDEO_MODEL, price_snapshot
from norlan.providers import ArkProvider, ProviderError
from tests import test_pipeline as fixtures


class PipelineBillingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(fixtures.fake_media())
        self.stack.enter_context(patch.dict(os.environ, {"ARK_API_KEY": "synthetic-test-key"}))
        self.stack.enter_context(patch("norlan.providers._api_open", side_effect=AssertionError("Real API IO is forbidden in synthetic tests")))
        self.stack.enter_context(patch("norlan.providers.urlopen", side_effect=AssertionError("Real media IO is forbidden in synthetic tests")))
        core.create_project(self.root, "Synthetic billing test novel.", mode="demo")
        state = pipeline.run(self.root)
        core.approve(self.root, "plan", "plan", state["plan"]["version"])
        fixtures.choose_gpt_assets(self.root)
        state = pipeline.run(self.root)
        for asset in state["assets"]:
            core.approve(self.root, "asset", asset["id"], asset["version"])
        self.config = {"image_model": IMAGE_MODEL, "video_model": VIDEO_MODEL, "resolution": "720p"}
        core.configure(self.root, {"ark": self.config, "caps": {"image": 1, "video_second": 1}}, limit=100)
        with core.transaction(self.root) as state:
            state["mode"] = "live"

    def video_response(self, **values):
        # Missing service_tier is intentional: absence preserves the locked
        # default request, while an explicit unsupported tier stays unpriced.
        return {"status": "succeeded", "model": VIDEO_MODEL, "resolution": "720p",
                "usage": {"completion_tokens": 108108},
                "content": {"video_url": "https://example.test/synthetic-video.mp4"}, **values}

    def live_jobs(self, state=None):
        state = state or core.load_project(self.root)
        return [job for job in state["jobs"].values() if job["amount"] > 0]

    def assert_video_cost(self, state):
        self.assertEqual(state["budget"]["spent"], 4)
        self.assertEqual(state["budget"]["usage_cost"], 3.999996)
        self.assertEqual(state["budget"]["usage_cost_jobs"], 1)
        self.assertEqual(state["budget"]["cost_jobs"], 1)
        self.assertIsNone(state["budget"]["actual_cost"])

    def test_image_callback_persists_usage_before_download_error_and_blocks_resubmit(self):
        asset = core.load_project(self.root)["assets"][0]
        core.reject(self.root, "asset", asset["id"], asset["version"], "Synthetic image revision.")
        raw = {"model": IMAGE_MODEL, "usage": {"generated_images": 1, "input_images": 0},
               "data": [{"url": "https://example.test/synthetic-image.png"}], "secret_extension": "SECRET-RAW"}

        def fail_download(url, path):
            # This runs inside generate_image before it returns to the pipeline.
            saved = self.live_jobs()[0]
            self.assertEqual(saved["billing"]["usage"], {"generated_images": 1, "input_images": 0})
            self.assertEqual(saved["billing"]["usage_cost"], 0.12)
            self.assertEqual(saved["accounting"], "reserved")
            raise ProviderError("Synthetic download failed")

        with patch.object(ArkProvider, "_request", return_value=(raw, "synthetic-image-request")) as request, \
             patch.object(ArkProvider, "download", side_effect=fail_download):
            with self.assertRaises(ProviderError):
                pipeline._assets(self.root)
            after = core.load_project(self.root)
            job = self.live_jobs(after)[0]
            self.assertEqual(job["accounting"], "unknown")
            self.assertEqual(job["state"], "uncertain")
            self.assertEqual(after["budget"]["unknown"], 1)
            self.assertEqual(after["budget"]["usage_cost"], 0.12)
            self.assertIsNone(after["budget"]["actual_cost"])
            with self.assertRaises(core.GateError):
                pipeline._assets(self.root)
            self.assertEqual(pipeline.run(self.root)["status"], "blocked")
            self.assertEqual(request.call_count, 1)
            self.assertEqual(request.call_args.args[:2], ("POST", "/images/generations"))
        self.assertEqual(len(self.live_jobs()), 1)
        self.assertNotIn("SECRET-RAW", (self.root / "project.json").read_text())

    def test_completed_video_without_url_saves_usage_and_recovers_by_get_only(self):
        output = self.root / "media" / "synthetic-video.mp4"
        responses = [({"id": "synthetic-video-task"}, "synthetic-submit"),
                     (self.video_response(content={}), "synthetic-poll-1"),
                     (self.video_response(), "synthetic-poll-2")]
        with patch.object(ArkProvider, "_request", side_effect=responses) as request, \
             patch.object(ArkProvider, "download", side_effect=lambda url, path: fixtures.write(path, b"synthetic-video")) as download:
            with self.assertRaises(core.GateError):
                pipeline._video(self.root, "video", "E001-S001", output, 4, "Synthetic video prompt.", [])
            state = core.load_project(self.root)
            job = self.live_jobs(state)[0]
            self.assertEqual(job["task_id"], "synthetic-video-task")
            self.assertEqual(job["state"], "download_pending")
            self.assertEqual(job["accounting"], "spent")
            self.assert_video_cost(state)
            download.assert_not_called()
            pipeline._video(self.root, "video", "E001-S001", output, 4, "Synthetic video prompt.", [], old_job=job["id"])
            self.assertEqual([call.args[0] for call in request.call_args_list], ["POST", "GET", "GET"])
            self.assertEqual([call.args[1] for call in request.call_args_list[1:]],
                             ["/contents/generations/tasks/synthetic-video-task"] * 2)
            # Once its immutable output exists, resuming the same job needs no
            # further request and cannot create another billable generation.
            pipeline._video(self.root, "video", "E001-S001", output, 4, "Synthetic video prompt.", [], old_job=job["id"])
            self.assertEqual(request.call_count, 3)
        self.assertEqual(len(self.live_jobs()), 1)
        self.assert_video_cost(core.load_project(self.root))

    def test_video_usage_is_durable_before_a_download_throws(self):
        output = self.root / "media" / "synthetic-download-error.mp4"
        responses = [({"id": "synthetic-download-task"}, "synthetic-submit"),
                     (self.video_response(), "synthetic-poll")]

        def fail_download(url, path):
            state = core.load_project(self.root)
            self.assert_video_cost(state)
            self.assertEqual(self.live_jobs(state)[0]["state"], "download_pending")
            raise ProviderError("Synthetic video download failed")

        with patch.object(ArkProvider, "_request", side_effect=responses) as request, \
             patch.object(ArkProvider, "download", side_effect=fail_download):
            with self.assertRaises(ProviderError):
                pipeline._video(self.root, "video", "E001-S001", output, 4, "Synthetic prompt.", [])
        self.assertEqual([call.args[0] for call in request.call_args_list], ["POST", "GET"])
        state = core.load_project(self.root)
        self.assert_video_cost(state)
        self.assertEqual(self.live_jobs(state)[0]["task_id"], "synthetic-download-task")
        self.assertFalse(output.exists())

    def test_duplicate_completed_queries_do_not_double_count_usage(self):
        output = self.root / "media" / "synthetic-requery.mp4"
        core.reserve(self.root, "synthetic-requery", "video", "E001-S001", 4,
                     pricing=price_snapshot(self.config, "video"))
        core.update_job(self.root, "synthetic-requery", state="submitted", task_id="synthetic-task")
        with patch.object(ArkProvider, "_request", return_value=(self.video_response(), "synthetic-request")) as request, \
             patch.object(ArkProvider, "download", side_effect=lambda url, path: fixtures.write(path, b"synthetic-video")):
            pipeline._poll_video(self.root, "synthetic-requery", output)
            pipeline._poll_video(self.root, "synthetic-requery", output)
        self.assertEqual([call.args[0] for call in request.call_args_list], ["GET", "GET"])
        state = core.load_project(self.root)
        self.assert_video_cost(state)
        self.assertEqual(len(self.live_jobs(state)), 1)
        events = [event for event in state["events"] if event["action"] == "provider_usage_recorded"]
        self.assertEqual(len(events), 1)

    def test_failed_task_reported_zero_usage_keeps_cap_and_is_not_a_verified_bill(self):
        output = self.root / "media" / "synthetic-failed.mp4"
        core.reserve(self.root, "synthetic-failed", "video", "E001-S001", 4,
                     pricing=price_snapshot(self.config, "video"))
        core.update_job(self.root, "synthetic-failed", state="submitted", task_id="synthetic-task")
        response = self.video_response(status="failed", usage={"completion_tokens": 0}, content={},
                                       error={"code": "SYNTHETIC_FAILURE", "message": "SECRET-RAW"})
        with patch.object(ArkProvider, "_request", return_value=(response, "synthetic-request")), \
             patch.object(ArkProvider, "download") as download:
            with self.assertRaises(core.GateError):
                pipeline._poll_video(self.root, "synthetic-failed", output)
            download.assert_not_called()
        state = core.load_project(self.root)
        self.assertEqual(state["budget"]["spent"], 4)
        self.assertEqual(state["budget"]["usage_cost"], 0)
        self.assertEqual(state["budget"]["usage_cost_jobs"], 1)
        self.assertIsNone(state["budget"]["actual_cost"])
        self.assertEqual(state["jobs"]["synthetic-failed"]["state"], "failed")
        self.assertNotIn("SECRET-RAW", (self.root / "project.json").read_text())


if __name__ == "__main__":
    unittest.main()
