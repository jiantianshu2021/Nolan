"""Synthetic transport/consent/ledger integration; all paid API IO is forbidden.

Media compression is stubbed with distinct derived bytes here. Real FFmpeg
quality, alpha and cache behavior belong to the transport media tests.
"""
import base64
import copy
import hashlib
import json
from pathlib import Path
import unittest
from unittest.mock import Mock, patch

from norlan import core, pipeline
from norlan.providers import ArkProvider, ProviderError
from test_pipeline import write
from test_video_model_gate import MINI, STANDARD
import test_video_model_pipeline as model_fixture


class ReferenceTransportPipelineTests(unittest.TestCase):
    answer = model_fixture.VideoModelPipelineTests.answer
    amendment = model_fixture.VideoModelPipelineTests.amendment
    construct = model_fixture.VideoModelPipelineTests.construct
    download = model_fixture.VideoModelPipelineTests.download
    shot = model_fixture.VideoModelPipelineTests.shot
    job = model_fixture.VideoModelPipelineTests.job
    prepare = model_fixture.VideoModelPipelineTests.prepare
    approve = model_fixture.VideoModelPipelineTests.approve
    posts = model_fixture.VideoModelPipelineTests.posts

    def setUp(self):
        model_fixture.VideoModelPipelineTests.setUp(self)
        self.post_error = None
        self.prepared_invocations = []
        self.transport = self.enterContext(patch.object(
            pipeline.media, "prepare_video_references", side_effect=self.prepare_transport))
        self.enterContext(patch.object(ArkProvider, "_audio_duration", return_value=4.0))

    def request(self, provider, method, path, payload=None):
        if method == "POST" and self.post_error is not None:
            self.requests.append({"method": method, "path": path,
                "config": copy.deepcopy(provider.config), "payload": copy.deepcopy(payload)})
            raise self.post_error
        return model_fixture.VideoModelPipelineTests.request(self, provider, method, path, payload)

    def prepare_transport(self, images, audio, directory):
        prepared = {"images": [], "audio": []}
        manifest = {"version": 1, "policy": "synthetic-reference-transport",
                    "synthetic": True, "images": [], "audio": []}
        for kind, paths in (("images", images), ("audio", audio)):
            for index, original in enumerate(paths):
                source = Path(original)
                # Keep JPEG tails and audio byte-identical, as the production
                # helper does for already small references.
                derived = kind == "images" and source.suffix.lower() == ".png"
                target = (write(Path(directory) / (core.digest(source) + ".jpg"),
                                b"\xff\xd8\xffSYNTHETIC TRANSPORT " + source.name.encode())
                          if derived else source)
                prepared[kind].append(target)
                manifest[kind].append({"source_path": str(source), "path": str(target),
                    "source_sha256": core.digest(source), "sha256": core.digest(target),
                    "source_bytes": source.stat().st_size, "bytes": target.stat().st_size,
                    "derived": derived, "synthetic": True})
        self.prepared_invocations.append({"images": list(images), "audio": list(audio),
            "prepared_images": prepared["images"], "prepared_audio": prepared["audio"]})
        return prepared["images"], prepared["audio"], manifest

    def test_dynamic_mock_method_is_not_mistaken_for_transport_implementation(self):
        provider = Mock()
        self.assertTrue(callable(provider.preflight_video_request))
        refs = [self.root / "not-needed.png"]
        audio = []
        actual = pipeline._prepare_video_transport(
            self.root, provider, "video", self.shot()["id"], "合成测试", 4, refs, audio)
        self.assertEqual(actual, (refs, audio, None))
        self.transport.assert_not_called()
        provider.preflight_video_request.assert_not_called()

    def test_oversized_whole_payload_blocks_before_reserve_and_keeps_manual_consent(self):
        candidate = self.prepare()
        raw = "  合成完整人工定稿。\n保留本次人工文本与所选模型。  "
        self.approve(candidate, MINI, text=raw)
        approved = copy.deepcopy(self.shot()["prompt_candidate"])
        baseline = core.load_project(self.root)
        self.brain.reset_mock()
        self.brain.side_effect = AssertionError("Local preflight cannot rewrite approved text")
        with patch("norlan.providers.REQUEST_LIMIT_BYTES", 128):
            for _ in range(2):
                self.assertEqual(pipeline.run(self.root)["status"], "blocked")
                state = core.load_project(self.root)
                self.assertEqual(state["jobs"], baseline["jobs"])
                self.assertEqual(state["budget"], baseline["budget"])
                shot = self.shot()
                self.assertEqual(shot["status"], "prompt_approved")
                self.assertEqual(shot["prompt_candidate"], approved)
                self.assertEqual(shot["prompt_candidate"]["prompt"], raw)
                self.assertEqual(shot["prompt_candidate"]["video_model"], MINI)
                self.assertNotIn("job_id", shot)
                self.assertFalse(shot.get("attempts"))
        self.assertFalse(self.posts())
        self.brain.assert_not_called()

    def test_post_uses_each_derived_image_in_original_order_and_records_hashes(self):
        candidate = self.prepare()
        original_refs = candidate["reference_manifest"]["images"]
        before = {path: (self.root / path).read_bytes() for path in original_refs}
        self.approve(candidate, STANDARD)
        self.assertEqual(pipeline.run(self.root)["status"], "review_required")
        self.assertEqual(len(self.posts()), 1)
        payload = self.posts()[0]["payload"]
        self.assertEqual(payload["model"], STANDARD)
        self.assertEqual(payload["content"][0]["text"], candidate["prompt"])
        wire_images = [item for item in payload["content"] if item["type"] == "image_url"]
        transport = self.job()["reference_transport"]
        self.assertEqual(len(wire_images), len(original_refs))
        self.assertEqual(len(transport["images"]), len(original_refs))
        for original, row, item in zip(original_refs, transport["images"], wire_images):
            self.assertEqual(item["role"], "reference_image")
            self.assertEqual(row["source_path"], original)
            self.assertEqual(row["source_sha256"], core.digest(self.root / original))
            self.assertTrue(row["derived"])
            self.assertNotEqual(row["path"], original)
            transmitted = base64.b64decode(item["image_url"]["url"].split(",", 1)[1])
            self.assertEqual(transmitted, (self.root / row["path"]).read_bytes())
            self.assertEqual(hashlib.sha256(transmitted).hexdigest(), row["sha256"])
            self.assertEqual((self.root / original).read_bytes(), before[original])
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.assertEqual(transport["request_bytes"], len(body))
        self.assertEqual(transport["payload_sha256"], hashlib.sha256(body).hexdigest())
        self.assertEqual(transport["request_limit_bytes"], 64_000_000)
        saved = (self.root / "project.json").read_text()
        self.assertNotIn("data:image/", saved)
        self.assertNotIn("SYNTHETIC-NOT-A-REAL-KEY", saved)

    def test_next_segment_keeps_original_official_tail_bytes_and_reference_order(self):
        self.approve(self.prepare(), STANDARD)
        self.assertEqual(pipeline.run(self.root)["status"], "review_required")
        first = self.shot()
        official_tail = self.root / self.job()["official_tail"]["file"]
        original_tail = official_tail.read_bytes()
        core.approve(self.root, "shot", first["id"], first["version"])
        candidate = self.prepare(1)
        self.assertEqual(candidate["reference_manifest"]["images"][-1],
                         str(official_tail.relative_to(self.root)))
        self.approve(candidate, MINI, index=1)
        self.assertEqual(pipeline.run(self.root)["status"], "review_required")
        payload = self.posts()[1]["payload"]
        wire_images = [item for item in payload["content"] if item["type"] == "image_url"]
        rows = self.job(1)["reference_transport"]["images"]
        self.assertEqual([row["source_path"] for row in rows],
                         candidate["reference_manifest"]["images"])
        self.assertFalse(rows[-1]["derived"])
        self.assertEqual(rows[-1]["source_path"], rows[-1]["path"])
        self.assertEqual(rows[-1]["source_sha256"], rows[-1]["sha256"])
        self.assertEqual(base64.b64decode(wire_images[-1]["image_url"]["url"].split(",", 1)[1]),
                         original_tail)
        self.assertEqual(official_tail.read_bytes(), original_tail)

    def test_transport_preserves_audio_order_and_bytes_without_trimming(self):
        images = [write(self.root / "synthetic/source.png", b"SYNTHETIC ORIGINAL IMAGE")]
        audio = [write(self.root / f"synthetic/voice-{i}.wav", f"SYNTHETIC VOICE {i}".encode())
                 for i in range(2)]
        provider = ArkProvider({"video_model": STANDARD})
        prepared_images, prepared_audio, manifest = pipeline._prepare_video_transport(
            self.root, provider, "video", self.shot()["id"], "合成测试语音引用", 4, images, audio)
        self.assertEqual(prepared_audio, audio)
        self.assertEqual(len(prepared_images), len(images))
        self.assertEqual([row["source_path"] for row in manifest["audio"]],
                         [str(path.relative_to(self.root)) for path in audio])
        self.assertTrue(all(not row["derived"] for row in manifest["audio"]))
        payload = provider._video_payload("合成测试语音引用", 4, prepared_images,
                                          audio_references=prepared_audio)
        wire = [item for item in payload["content"] if item["type"] == "audio_url"]
        self.assertEqual(len(wire), 2)
        for item, original in zip(wire, audio):
            self.assertEqual(item["role"], "reference_audio")
            self.assertEqual(base64.b64decode(item["audio_url"]["url"].split(",", 1)[1]),
                             original.read_bytes())

    def test_changed_derived_bytes_after_preflight_block_post_and_release_reservation(self):
        candidate = self.prepare()
        self.approve(candidate, STANDARD)
        before = core.load_project(self.root)["budget"]
        real_preflight = ArkProvider.preflight_video_request

        def change_after_check(provider, *args, **kwargs):
            checked = real_preflight(provider, *args, **kwargs)
            prepared = self.prepared_invocations[-1]["prepared_images"][0]
            prepared.write_bytes(prepared.read_bytes() + b"SYNTHETIC CHANGE AFTER CHECK")
            return checked

        with patch.object(ArkProvider, "preflight_video_request", autospec=True,
                          side_effect=change_after_check):
            self.assertEqual(pipeline.run(self.root)["status"], "generation_failed")
        job = self.job()
        self.assertEqual(job["state"], "failed")
        self.assertEqual(job["accounting"], "released")
        self.assertEqual(job["submission_status"], "not_submitted")
        self.assertEqual(job["failure"]["stage"], "local_validation")
        self.assertFalse(job["failure"]["uncertain"])
        self.assertIsNone(job["task_id"])
        self.assertFalse(self.posts())
        after = core.load_project(self.root)["budget"]
        for key in ("spent", "reserved", "unknown", "available"):
            self.assertEqual(after[key], before[key])
        pipeline.run(self.root)
        self.assertFalse(self.posts())

    def test_remote_rejection_retains_conservative_ledger_and_does_not_auto_retry(self):
        self.approve(self.prepare(), STANDARD)
        self.post_error = ProviderError("合成测试：供应商拒绝", stage="submission",
            failure={"http_status": 400, "provider_code": "InvalidParameter", "request_id": "synthetic-request"})
        self.assertEqual(pipeline.run(self.root)["status"], "generation_failed")
        self.assertEqual(self.job()["accounting"], "spent")
        self.assertEqual(self.job()["submission_status"], "rejected")
        self.assertEqual(self.job()["failure"]["http_status"], 400)
        self.assertEqual(len(self.posts()), 1)
        pipeline.run(self.root)
        self.assertEqual(len(self.posts()), 1)

    def test_uncertain_submission_keeps_unknown_budget_and_forbids_another_post(self):
        self.approve(self.prepare(), STANDARD)
        self.post_error = ProviderError("合成测试：提交连接中断", stage="submission", uncertain=True)
        self.assertEqual(pipeline.run(self.root)["status"], "blocked")
        self.assertEqual(self.job()["accounting"], "unknown")
        self.assertEqual(self.job()["submission_status"], "unknown")
        self.assertGreater(core.load_project(self.root)["budget"]["unknown"], 0)
        self.assertEqual(len(self.posts()), 1)
        pipeline.run(self.root)
        self.assertEqual(len(self.posts()), 1)

    def test_poll_recovery_reuses_saved_job_without_transport_or_post(self):
        self.approve(self.prepare(), STANDARD)
        self.interrupt_poll_once = True
        self.assertEqual(pipeline.run(self.root)["status"], "blocked")
        original = copy.deepcopy(self.job())
        self.assertEqual(original["state"], "submitted")
        self.assertEqual(len(self.posts()), 1)
        self.transport.reset_mock()
        self.transport.side_effect = AssertionError("Resuming must not recompress or reconstruct a POST")
        self.brain.reset_mock()
        self.brain.side_effect = AssertionError("Resuming must not rewrite an already submitted video")
        core.configure(self.root, {"ark": {"video_model": MINI}})
        self.assertEqual(pipeline.run(self.root)["status"], "review_required")
        self.assertEqual(len(self.posts()), 1)
        self.assertEqual(self.job()["id"], original["id"])
        self.assertEqual(self.job()["task_id"], original["task_id"])
        self.assertEqual(self.job()["reference_transport"], original["reference_transport"])
        self.assertEqual(self.job()["video_request"]["video_model"], STANDARD)
        self.transport.assert_not_called()
        self.brain.assert_not_called()


if __name__ == "__main__":
    unittest.main()
