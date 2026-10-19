"""Synthetic local folder through real FFmpeg; no network or paid providers."""
import json
from pathlib import Path
import shutil
import tempfile
import unittest

from norlan import core, episode_post
import test_episode_post as post_fixtures


class FolderPostTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        post_fixtures.EpisodePostTests.setUpClass()

    @classmethod
    def tearDownClass(cls):
        post_fixtures.EpisodePostTests.tearDownClass()

    def setUp(self):
        self.case = post_fixtures.EpisodePostTests()
        self.case.setUp()
        self.addCleanup(self.case.doCleanups)
        self.root = self.case.root
        self.folder_temp = tempfile.TemporaryDirectory(prefix="nolan-synthetic-score-folder-")
        self.addCleanup(self.folder_temp.cleanup)
        self.folder = Path(self.folder_temp.name).resolve()
        shutil.copyfile(self.case.fixture / "synthetic-music.wav", self.folder / "synthetic.wav")
        (self.folder / "index.json").write_text(json.dumps({"tracks": [{"file": "synthetic.wav", "title": "SYNTHETIC TEST MUSIC",
            "author": "Synthetic Author", "license": {"name": "SYNTHETIC TEST LICENSE", "url": "https://example.invalid/license"},
            "source_url": "https://example.invalid/source", "attribution": "Synthetic test attribution",
            "editorial": {"moods": ["工业", "科技"], "suggested_use": "合成测试，不能用作真实作品"}}]}))
        core.configure_music_folder(self.root, self.folder)

    def test_folder_scan_and_music_approval_preserve_video_then_export_credits(self):
        before = self.case.shots()
        report = core.scan_music_folder(self.root)
        self.assertEqual(len(report["imported"]), 1)
        self.assertFalse(self.case.finish(), "first local music always waits for human audition")
        state = core.load_project(self.root)
        track = state["music_library"]["tracks"][0]
        self.assertEqual(track["status"], "awaiting_approval")
        self.assertFalse(state["episodes"][0]["postproduction"]["music"]["can_generate"])
        episode_post.decide_music(self.root, track["id"], track["version"], track["sha256"])
        self.assertTrue(self.case.finish())
        after = self.case.episode()
        credits = core.safe_path(self.root, after["postproduction"]["music"]["credits_file"])
        self.assertIn("Synthetic test attribution", credits.read_text())
        self.assertIn("Synthetic Author", credits.read_text())
        self.assertEqual(self.case.shots(), before)
        self.case.soundtrack.assert_not_called()
        self.assertEqual(core.scan_music_folder(self.root)["imported"], [])
        core.approve(self.root, "episode", after["id"], after["version"])

    def test_unsubmitted_old_generation_flag_is_retired_without_any_call(self):
        with core.transaction(self.root) as state:
            state["episodes"][0].setdefault("postproduction", {})["music_generation_requested"] = True
        self.assertFalse(self.case.finish())
        self.assertFalse(self.case.episode()["postproduction"]["music_generation_requested"])
        self.case.soundtrack.assert_not_called()
        self.assertEqual(core.load_project(self.root)["jobs"], {})

    def test_missing_or_changed_credit_cannot_approve_a_local_music_export(self):
        core.scan_music_folder(self.root)
        track = core.load_project(self.root)["music_library"]["tracks"][0]
        episode_post.decide_music(self.root, track["id"], track["version"], track["sha256"])
        self.assertTrue(self.case.finish())
        ep = self.case.episode()
        with core.transaction(self.root) as state:
            state["episodes"][0]["postproduction"]["music"].pop("credits_file")
        with self.assertRaisesRegex(core.GateError, "署名"):
            core.approve(self.root, "episode", ep["id"], ep["version"])


if __name__ == "__main__":
    unittest.main()
