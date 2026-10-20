"""Synthetic source import checks; no model or media providers."""
import copy
from pathlib import Path
import tempfile
import unittest

from norlan import core, season_sources


SCRIPT = "## 第{number:02}集 合成第{number}集\n\n预计时长：约{duration}秒\n\n人物甲：这是合成测试文本。\n"


class SeasonSourceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.root = self.base / "project"
        self.folder = self.base / "scripts"
        self.folder.mkdir()
        self.text = "\n".join(SCRIPT.format(number=n, duration=120+n*5) for n in (1, 2, 3))
        (self.folder / "synthetic.md").write_text(self.text, encoding="utf-8")
        core.create_project(self.root, "合成旧剧本，请勿覆盖", mode="demo")
        with core.transaction(self.root) as state:
            state.update(status="complete", episodes=[{"id": "ep01", "title": "合成第1集", "status": "approved",
                "version": 5, "file": "exports/unchanged.mp4", "sha256": "a"*64,
                "approved_sha256": "a"*64, "shots": [{"id": "old-shot", "duration": 10}]}])
            state["plan"].update(status="approved", version=1)
            state["assets"] = [{"id": "old-asset", "status": "approved", "version": 3}]

    def test_import_preserves_all_existing_production_and_budget(self):
        old = core.load_project(self.root)
        state = season_sources.import_season(self.root, self.folder)
        self.assertEqual(state["episodes"][0], old["episodes"][0])
        for key in ("assets", "jobs", "budget", "plan", "settings"):
            self.assertEqual(state[key], old[key])
        self.assertEqual((self.root / "source/novel.txt").read_text(), "合成旧剧本，请勿覆盖")
        self.assertEqual([ep["id"] for ep in state["episodes"]], ["ep01", "ep02", "ep03"])
        self.assertEqual(state["episodes"][1]["source_duration"], 130)
        self.assertEqual(state["episodes"][1]["shots"], [])
        self.assertEqual(state["episodes"][2]["status"], "awaiting_episode_plan")
        self.assertEqual(state["status"], "ready")

    def test_import_is_idempotent_and_does_not_reset_progress(self):
        season_sources.import_season(self.root, self.folder)
        before = (self.root / "project.json").read_bytes()
        season_sources.import_season(self.root, self.folder)
        self.assertEqual(before, (self.root / "project.json").read_bytes())

    def test_changed_import_and_changed_existing_title_are_rejected(self):
        season_sources.import_season(self.root, self.folder)
        before = (self.root / "project.json").read_bytes()
        source = self.folder / "synthetic.md"
        source.write_text(self.text.replace("约130秒", "约140秒"), encoding="utf-8")
        with self.assertRaises(core.GateError):
            season_sources.import_season(self.root, self.folder)
        self.assertEqual(before, (self.root / "project.json").read_bytes())

    def test_source_accessor_reads_current_episode_only_and_binds_hash(self):
        state = season_sources.import_season(self.root, self.folder)
        episode = state["episodes"][1]
        text, sha = season_sources.source_text(self.root, episode=episode)
        self.assertIn("第02集", text)
        self.assertNotIn("第03集", text)
        self.assertEqual(sha, episode["source_sha256"])
        (self.root / episode["source_file"]).write_text("changed", encoding="utf-8")
        with self.assertRaises(core.GateError):
            season_sources.source_text(self.root, episode=episode)

    def test_parse_rejects_duplicate_missing_and_invalid_duration(self):
        for text in (self.text + SCRIPT.format(number=2,duration=130),
                     SCRIPT.format(number=2,duration=130), self.text.replace("约130秒", "约190秒")):
            with self.subTest(text=text[:45]), self.assertRaises(core.GateError):
                season_sources.parse_scripts([("synthetic", text)])

    def test_inflight_job_prevents_import_without_state_change(self):
        with core.transaction(self.root) as state:
            state["jobs"]["synthetic-job"] = {"id": "synthetic-job", "state": "submitted", "accounting": "reserved", "amount": 0}
        before = (self.root / "project.json").read_bytes()
        with self.assertRaises(core.GateError):
            season_sources.import_season(self.root, self.folder)
        self.assertEqual(before, (self.root / "project.json").read_bytes())


if __name__ == "__main__":
    unittest.main()
