"""Offline local-folder ingestion. Audio fixtures are synthetic sine waves only."""
import array
import copy
import hashlib
import io
import json
import math
from pathlib import Path
import shutil
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from unittest.mock import patch
import wave

from norlan import cli, core, music_folder


@unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "local FFmpeg required; no providers")
class MusicFolderTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="nolan-synthetic-folder-test-")
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name).resolve()
        self.root = self.base / "project"
        self.folder = self.base / "music-source"
        self.folder.mkdir()
        core.create_project(self.root, "合成测试，不是真实剧本", mode="demo")
        with core.transaction(self.root) as state:
            state["status"] = "paused"
            state["paused"] = True
            state["episodes"] = [{"id": "E001", "version": 3, "status": "approved", "shots": [
                {"id": "S001", "version": 2, "status": "accepted", "sha256": "0" * 64}],
                "approved_sha256": "1" * 64, "postproduction": {"final_ready": True, "music_key": "keep"}}]
            state["assets"] = [{"id": "asset-keep", "status": "approved"}]
        self.source = self.tone(self.folder / "合成测试 A.wav")

    def tone(self, path, frequency=400):
        samples = array.array("h", [round(3000 * math.sin(2 * math.pi * frequency * i / 8000)) for i in range(4000)])
        with wave.open(str(path), "wb") as handle:
            handle.setnchannels(1)
            handle.setsampwidth(2)
            handle.setframerate(8000)
            handle.writeframes(samples.tobytes())
        return path

    def write_index(self, tracks):
        (self.folder / "index.json").write_text(json.dumps({"schema_version": 1, "library_name": "合成测试库",
            "approval_required": True, "tracks": tracks}, ensure_ascii=False), encoding="utf-8")

    def test_configure_and_scan_keep_approved_picture_progress_and_costs(self):
        before = core.load_project(self.root)
        original = self.source.read_bytes()
        configured = core.configure_music_folder(self.root, self.folder)
        self.assertNotIn("music_library", configured)
        self.assertEqual(configured["settings"]["music"]["mode"], "library")
        report = core.scan_music_folder(self.root)
        after = core.load_project(self.root)
        self.assertEqual(len(report["imported"]), 1)
        self.assertEqual(report["duplicates"], 0)
        self.assertEqual(report["skipped"], [])
        track = after["music_library"]["tracks"][0]
        self.assertEqual(track["status"], "awaiting_approval")
        self.assertNotIn("approved_sha256", track)
        self.assertEqual(track["provenance"]["license_status"], "unknown")
        self.assertTrue(core.safe_path(self.root, track["file"]).is_relative_to(self.root / "music/library"))
        self.assertEqual(core.safe_path(self.root, track["file"]).read_bytes(), original)
        self.assertEqual(self.source.read_bytes(), original)
        for key in ("status", "paused", "episodes", "assets", "jobs", "budget"):
            self.assertEqual(after[key], before[key], key)
        self.assertEqual(after["settings"]["music"]["folder_scan"], report)

    def test_hash_deduplication_covers_rejected_and_approved_tracks_without_new_event(self):
        shutil.copyfile(self.source, self.folder / "same bytes.wav")
        first = core.scan_music_folder(self.root, self.folder)
        self.assertEqual((len(first["imported"]), first["duplicates"]), (1, 1))
        for decision in ("rejected", "approved"):
            with core.transaction(self.root) as state:
                state["music_library"]["tracks"][0]["status"] = decision
            before = core.load_project(self.root)
            report = core.scan_music_folder(self.root)
            after = core.load_project(self.root)
            self.assertEqual((report["imported"], report["duplicates"]), ([], 2))
            self.assertEqual(after["music_library"], before["music_library"])
            self.assertEqual(after["events"], before["events"])
            self.assertEqual(after["budget"], before["budget"])

    def test_starter_metadata_has_bounded_source_license_and_editorial_not_fake_approval(self):
        digest = hashlib.sha256(self.source.read_bytes()).hexdigest()
        self.write_index([{"file": self.source.name, "id": "catalogue-origin-id", "title": "仅测试的音乐标题",
            "author": "合成测试作者", "sha256": digest,
            "source_url": "https://example.invalid/synthetic/music", "download_url": "https://example.invalid/synthetic.wav",
            "license": {"name": "合成许可证字段，不是真实授权", "url": "https://example.invalid/license",
                        "commercial_use": True, "adaptation": True, "attribution_required": True},
            "attribution": "合成测试署名", "source_metadata": {"instruments": ["合成测试乐器"], "feel": "平静",
                "tempo_bpm": 90, "isrc": "SYNTHETIC-TEST", "catalogue_length": "0:01"},
            "editorial": {"moods": ["测试情绪"], "suggested_use": "测试用途" * 500,
                          "avoid": "不是配乐质量验证", "basis": "素材页面标签，未实际听审"},
            "verification": {"retrieved_at": "2026-10-20", "source_evidence_file": "evidence/test.json"},
            "api_key": "must-not-enter-state", "arbitrary_large_object": {"history": "x" * 5000}}])
        core.scan_music_folder(self.root, self.folder)
        state = core.load_project(self.root)
        track = state["music_library"]["tracks"][0]
        metadata = track["metadata"]
        self.assertEqual(metadata["author"], "合成测试作者")
        self.assertTrue(metadata["license"]["commercial_use"])
        self.assertEqual(metadata["source_metadata"]["feel"], ["平静"])
        self.assertEqual(metadata["source_metadata"]["catalogue_length"], "0:01")
        self.assertEqual(metadata["editorial"]["basis"], "素材页面标签，未实际听审")
        self.assertEqual(track["provenance"]["license"], metadata["license"])
        self.assertEqual(track["provenance"]["editorial"], metadata["editorial"])
        self.assertEqual(track["provenance"]["license_status"], "declared")
        self.assertLessEqual(len(track["prompt"]), 450)
        self.assertNotIn("must-not-enter-state", json.dumps(state))
        self.assertEqual(track["status"], "awaiting_approval")

    def test_new_file_is_discovered_but_unchanged_track_never_mutates(self):
        core.scan_music_folder(self.root, self.folder)
        original = copy.deepcopy(core.load_project(self.root)["music_library"]["tracks"][0])
        self.tone(self.folder / "合成测试 B.wav", frequency=800)
        report = core.scan_music_folder(self.root)
        self.assertEqual((len(report["imported"]), report["duplicates"]), (1, 1))
        self.assertEqual(core.load_project(self.root)["music_library"]["tracks"][0], original)

    def test_source_bpm_numeric_strings_are_normalized_without_nonfinite_values(self):
        for value, expected in (("83", 83.0), ("110", 110.0), (" 83.5 ", 83.5),
                                ("NaN", None), ("Inf", None), ("-Inf", None), ("0", None),
                                ("401", None), ("83 BPM", None), (True, None), (None, None)):
            with self.subTest(value=value):
                metadata = music_folder._metadata({"source_metadata": {"tempo_bpm": value}})
                source = metadata.get("source_metadata", {})
                if expected is None:
                    self.assertNotIn("tempo_bpm", source)
                else:
                    self.assertEqual(source["tempo_bpm"], expected)
                    self.assertIn(str(expected), music_folder._prompt(metadata))

    def test_invalid_and_symlink_audio_skipped_and_subfolders_not_walked(self):
        (self.folder / "bad.mp3").write_text("not valid audio", encoding="utf-8")
        (self.folder / "outside.wav").symlink_to(self.source)
        nested = self.folder / "nested"
        nested.mkdir()
        self.tone(nested / "should-not-import.wav", frequency=600)
        (self.folder / "readme.txt").write_text("ignored")
        report = core.scan_music_folder(self.root, self.folder)
        self.assertEqual(len(report["imported"]), 1)
        self.assertEqual(len(report["skipped"]), 2)
        self.assertEqual({item["name"] for item in report["skipped"]}, {"bad.mp3", "outside.wav"})
        self.assertNotIn(str(self.base), report["skipped"][0]["reason"])

    def test_unsupported_and_empty_folders_do_not_invent_music(self):
        self.source.unlink()
        report = core.scan_music_folder(self.root, self.folder)
        self.assertEqual(report["imported"], [])
        self.assertNotIn("music_library", core.load_project(self.root))

    def test_index_checksum_mismatch_skips_candidate_without_replacing_original(self):
        original = self.source.read_bytes()
        self.write_index([{"file": self.source.name, "sha256": "0" * 64}])
        report = core.scan_music_folder(self.root, self.folder)
        self.assertEqual(len(report["skipped"]), 1)
        self.assertIn("摘要", report["skipped"][0]["reason"])
        self.assertEqual(report["imported"], [])
        self.assertEqual(self.source.read_bytes(), original)

    def test_paths_index_links_and_oversized_libraries_fail_before_any_import(self):
        with self.assertRaisesRegex(music_folder.MusicFolderError, "不存在"):
            core.scan_music_folder(self.root, self.base / "missing")
        with self.assertRaisesRegex(music_folder.MusicFolderError, "单个"):
            core.configure_music_folder(self.root, self.source)
        linked = self.base / "linked"
        linked.symlink_to(self.folder, target_is_directory=True)
        with self.assertRaisesRegex(music_folder.MusicFolderError, "符号链接"):
            core.configure_music_folder(self.root, linked)
        self.write_index([{"file": "../outside.wav"}])
        with self.assertRaisesRegex(music_folder.MusicFolderError, "外部路径"):
            core.scan_music_folder(self.root, self.folder)
        (self.folder / "index.json").unlink()
        (self.folder / "index.json").symlink_to(self.source)
        with self.assertRaisesRegex(music_folder.MusicFolderError, "符号链接"):
            core.scan_music_folder(self.root, self.folder)
        (self.folder / "index.json").unlink()
        self.tone(self.folder / "another.wav", frequency=800)
        with patch.object(music_folder, "MAX_FILES", 1):
            with self.assertRaisesRegex(music_folder.MusicFolderError, "最多 100"):
                core.scan_music_folder(self.root, self.folder)
        with patch.object(music_folder, "MAX_ENTRIES", 1):
            with self.assertRaisesRegex(music_folder.MusicFolderError, "1000"):
                core.scan_music_folder(self.root, self.folder)
        with patch.object(music_folder, "MAX_TOTAL_BYTES", 1):
            with self.assertRaisesRegex(music_folder.MusicFolderError, "500 MB"):
                core.scan_music_folder(self.root, self.folder)
        self.assertNotIn("music_library", core.load_project(self.root))

    def test_configure_does_not_bypass_active_production_lock(self):
        with core.run_lock(self.root):
            with self.assertRaisesRegex(core.GateError, "已有制作任务"):
                core.scan_music_folder(self.root, self.folder)
            with self.assertRaisesRegex(core.GateError, "已有制作任务"):
                core.configure_music_folder(self.root, self.folder)

    def test_cli_register_and_scan_never_launch_production_or_provider(self):
        out = io.StringIO()
        with patch.object(cli, "remember"), patch("norlan.pipeline.run") as production, redirect_stdout(out):
            self.assertEqual(cli.main(["music-folder", str(self.root), str(self.folder)]), 0)
            self.assertIn("已登记", out.getvalue())
            self.assertEqual(cli.main(["music-folder", str(self.root), "--scan"]), 0)
            production.assert_not_called()
        self.assertIn("新增候选 1 首", out.getvalue())
        self.assertIn("未调用付费", out.getvalue())
        out = io.StringIO()
        with patch.object(cli, "remember"), redirect_stdout(out):
            self.assertEqual(cli.main(["music-folder", str(self.root), "--scan", "--json"]), 0)
        self.assertEqual(json.loads(out.getvalue())["duplicates"], 1)
        with redirect_stderr(io.StringIO()):
            self.assertEqual(cli.main(["music-folder", str(self.root)]), 2)


if __name__ == "__main__":
    unittest.main()
