"""Synthetic metadata only; test local selection and attribution publication."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from norlan import music_library as music
from norlan.mcp_server import _compact_music_status


class CatalogCreditTests(unittest.TestCase):
    def test_conversational_status_omits_large_music_metadata_and_keeps_counts(self):
        source = {"mode": "library", "folder": "/synthetic/music", "tracks": [
            {"id": f"test-{i}", "title": "SYNTHETIC", "version": 1, "status": "awaiting_approval",
             "prompt": "unnecessary-full-prompt" * 1000, "metadata": {"attribution": "unnecessary-credits" * 1000}}
            for i in range(100)]}
        result = _compact_music_status(source)
        self.assertEqual(result["track_count"], 100)
        self.assertEqual(result["awaiting_approval_count"], 100)
        self.assertEqual(len(result["tracks"]), 12)
        self.assertNotIn("unnecessary", str(result))

    def test_small_catalog_prefers_relevant_and_continuing_approved_themes(self):
        catalog = [{"id": f"music-{i}", "title": "SYNTHETIC", "prompt": "普通音乐", "tags": []} for i in range(60)]
        catalog[50]["tags"] = ["工业", "科技", "系统"]
        selected = music.select_catalog(catalog, "工业科技系统", previous_ids=["music-40"], preferred_ids=["music-30"])
        self.assertEqual(len(selected), 12)
        self.assertEqual([t["id"] for t in selected[:3]], ["music-30", "music-50", "music-40"])
        self.assertEqual(catalog[0]["id"], "music-0")

    def test_catalog_limit_cannot_expand_model_context(self):
        with self.assertRaises(music.MusicLibraryError):
            music.select_catalog([], "", limit=100)

    def test_used_tracks_only_get_exact_credits_and_edit_notice(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            library = {"tracks": [
                {"id": "used", "title": "SYNTHETIC USED TRACK", "provenance": {
                    "author": "Synthetic Author", "source_url": "https://example.invalid/test-only",
                    "license": {"name": "SYNTHETIC LICENSE", "url": "https://example.invalid/license"},
                    "attribution": "Synthetic Author — Synthetic Track"}},
                {"id": "unused", "title": "UNUSED TRACK", "provenance": {}}]}
            output = root / "exports" / "episode-v1.credits.txt"
            with patch.object(music, "_verify") as verify:
                result = music.write_music_credits({"cues": [{"track_id": "used"}, {"track_id": "used"}]}, library, output, project=root)
            text = output.read_text()
            self.assertIn("Synthetic Author", text)
            self.assertIn("https://example.invalid/license", text)
            self.assertIn("剪辑片段", text)
            self.assertNotIn("UNUSED TRACK", text)
            self.assertEqual(text.count("1. SYNTHETIC USED TRACK"), 1)
            self.assertEqual(result["sha256"], music._digest(output))
            verify.assert_called_once()

    def test_unknown_license_is_not_presented_as_a_commercial_permission(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            with patch.object(music, "_verify"):
                result = music.write_music_credits({"cues": [{"track_id": "test"}]},
                    {"tracks": [{"id": "test", "title": "SYNTHETIC", "provenance": {}}]},
                    root / "exports" / "test.credits.txt", project=root)
            self.assertIn("许可信息尚未登记", Path(result["file"]).read_text())

    def test_existing_credit_file_cannot_be_overwritten_and_external_paths_are_blocked(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            target = root / "exports" / "test.credits.txt"
            target.parent.mkdir()
            target.write_text("older immutable credit")
            with self.assertRaises(music.MusicLibraryError):
                music.write_music_credits({"cues": []}, {"tracks": []}, target, project=root)
            with self.assertRaises(music.MusicLibraryError):
                music.write_music_credits({"cues": []}, {"tracks": []}, root / "outside.txt", project=root)


if __name__ == "__main__":
    unittest.main()
