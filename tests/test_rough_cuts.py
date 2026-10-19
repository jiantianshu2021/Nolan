"""Synthetic cumulative-preview integration tests; no online providers."""
import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from norlan import core, pipeline
from test_pipeline import choose_gpt_assets, fake_media, write


class RoughCutTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.enterContext(fake_media())
        core.create_project(self.root, "离线累计粗剪测试。")
        plan = pipeline.demo_plan()
        template = plan["episodes"][0]["shots"][0]
        plan["episodes"] = [plan["episodes"][0]]
        plan["episodes"][0]["shots"] = [
            {**copy.deepcopy(template), "id": f"E001-S{index:03}"} for index in range(1, 5)
        ]
        core.install_plan(self.root, plan)
        core.approve(self.root, "plan", "plan", 1)
        choose_gpt_assets(self.root)
        state = pipeline.run(self.root)
        for asset in state["assets"]:
            core.approve(self.root, "asset", asset["id"], asset["version"])
        pipeline.media.compose.reset_mock()

    def episode(self):
        return core.load_project(self.root)["episodes"][0]

    def request_recompose(self):
        # Synthetic stand-in for a requested local subtitle/music recomposition;
        # clip versions and source hashes remain unchanged.
        with core.transaction(self.root) as state:
            state["status"] = "ready"
            state["episodes"][0]["status"] = "revision_requested"
            state["episodes"][0].pop("progress_stage", None)

    def test_each_finished_shot_publishes_cut_before_batch_review(self):
        evidence = pipeline.media.extract_evidence.side_effect
        publication_checks = []

        def observe(video, directory):
            if "-rough-" in str(video):
                relative = str(Path(video).relative_to(self.root))
                cuts = self.episode()["rough_cuts"]
                cut = next(item for item in cuts if item["file"] == relative)
                publication_checks.append(cut["shot_count"])
                self.assertEqual(core.digest(video), cut["sha256"])
                # The cut is already public before extraction and GPT review.
                self.assertFalse(any(item["key"] == cut["key"] for item in self.episode()["reviews"]))
            return evidence(video, directory)

        with patch.object(pipeline.media, "extract_evidence", side_effect=observe), \
             patch.object(pipeline, "_brain") as brain:
            state = pipeline.run(self.root)
        self.assertEqual(state["status"], "awaiting_episode_review", state.get("last_error"))
        episode = state["episodes"][0]
        self.assertEqual([cut["shot_count"] for cut in episode["rough_cuts"]], [2, 3, 4])
        self.assertEqual(publication_checks, [3, 4])
        self.assertEqual(len(episode["reviews"]), 2)
        self.assertEqual(pipeline.media.compose.call_count, 4)  # 3 previews + final
        brain.assert_not_called()
        for cut in episode["rough_cuts"]:
            self.assertTrue(cut["created_at"])
            self.assertEqual(cut["shots"], [
                {"id": shot["id"], "version": shot["version"], "sha256": shot["sha256"]}
                for shot in episode["shots"][:cut["shot_count"]]
            ])

    def test_recomposition_reuses_existing_cuts_and_reviews(self):
        pipeline.run(self.root)
        before = copy.deepcopy(self.episode())
        self.request_recompose()
        pipeline.media.compose.reset_mock()
        pipeline.run(self.root)
        after = self.episode()
        self.assertEqual(after["rough_cuts"], before["rough_cuts"])
        self.assertEqual(after["reviews"], before["reviews"])
        self.assertEqual(pipeline.media.compose.call_count, 1)
        self.assertNotIn("-rough-", str(pipeline.media.compose.call_args.args[1]))

    def test_revision_preserves_old_cuts_and_adds_only_changed_prefixes(self):
        state = pipeline.run(self.root)
        old_cuts = copy.deepcopy(state["episodes"][0]["rough_cuts"])
        old_bytes = {cut["file"]: (self.root / cut["file"]).read_bytes() for cut in old_cuts}
        core.reject(self.root, "episode", "E001", state["episodes"][0]["version"],
                    "合成测试：修改第三镜动作。", ["E001-S003"])
        state = pipeline.run(self.root)
        self.assertEqual(state["status"], "awaiting_episode_review", state.get("last_error"))
        cuts = state["episodes"][0]["rough_cuts"]
        self.assertEqual(cuts[:3], old_cuts)
        self.assertEqual([cut["shot_count"] for cut in cuts], [2, 3, 4, 3, 4])
        self.assertEqual(len({cut["key"] for cut in cuts}), len(cuts))
        for cut in old_cuts:
            self.assertEqual((self.root / cut["file"]).read_bytes(), old_bytes[cut["file"]])
        self.assertEqual(cuts[-1]["shots"][2]["version"], 2)
        self.assertNotEqual(cuts[-1]["shots"][2]["sha256"], old_cuts[-1]["shots"][2]["sha256"])

    def test_legacy_batch_review_file_is_indexed_without_reencoding(self):
        with core.transaction(self.root) as state:
            state["episodes"][0]["shots"] = state["episodes"][0]["shots"][:2]
        pipeline.run(self.root)
        legacy = copy.deepcopy(self.episode()["reviews"][0])
        with core.transaction(self.root) as state:
            state["episodes"][0].pop("rough_cuts")
        self.request_recompose()
        pipeline.media.compose.reset_mock()
        pipeline.run(self.root)
        indexed = self.episode()["rough_cuts"]
        self.assertEqual(len(indexed), 1)
        for field in ("key", "file", "sha256"):
            self.assertEqual(indexed[0][field], legacy[field])
        self.assertEqual(pipeline.media.compose.call_count, 1)  # final only

    def test_tampered_registered_cut_is_not_overwritten_or_reused(self):
        state = pipeline.run(self.root)
        cut = state["episodes"][0]["rough_cuts"][0]
        path = self.root / cut["file"]
        path.write_bytes(b"SYNTHETIC MODIFIED FILE")
        self.request_recompose()
        pipeline.media.compose.reset_mock()
        before_jobs = len(state["jobs"])
        state = pipeline.run(self.root)
        self.assertEqual(state["status"], "blocked")
        self.assertIn("内容已改变", state["last_error"])
        self.assertEqual(path.read_bytes(), b"SYNTHETIC MODIFIED FILE")
        self.assertEqual(len(state["jobs"]), before_jobs)
        pipeline.media.compose.assert_not_called()

    def test_preview_survives_failed_review_and_resume_keeps_batch_frequency(self):
        with core.transaction(self.root) as state:
            state["mode"] = "live"
            for shot in state["episodes"][0]["shots"]:
                path = write(self.root / "media" / (shot["id"] + ".mp4"), b"SYNTHETIC " + shot["id"].encode())
                shot.update(status="accepted", version=1, file=str(path.relative_to(self.root)), sha256=core.digest(path))

        def interrupted(root, method, context, images):
            self.assertEqual(method, "review")
            self.assertEqual([cut["shot_count"] for cut in self.episode()["rough_cuts"]], [2, 3])
            raise RuntimeError("Synthetic review interruption")

        with patch.object(pipeline, "_brain", side_effect=interrupted) as brain:
            state = pipeline.run(self.root)
        self.assertEqual(state["status"], "blocked")
        self.assertEqual(brain.call_count, 1)
        before = copy.deepcopy(self.episode()["rough_cuts"])
        pipeline.media.compose.reset_mock()
        review = {"verdict": "pass", "issues": [], "observed_end_state": "synthetic", "repair_prompt": ""}
        with patch.object(pipeline, "_brain", side_effect=lambda *args: copy.deepcopy(review)) as brain:
            state = pipeline.run(self.root)
        self.assertEqual(state["status"], "awaiting_episode_review", state.get("last_error"))
        self.assertEqual(self.episode()["rough_cuts"][:2], before)
        self.assertEqual(brain.call_count, 2)  # only the 3-shot and 4-shot boundaries
        self.assertEqual(pipeline.media.compose.call_count, 2)  # cut 4 + final


if __name__ == "__main__":
    unittest.main()
