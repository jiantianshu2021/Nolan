"""Synthetic postproduction UI/HTTP contracts. No model or paid provider calls."""
import copy
import http.client
import json
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import threading
import unittest
from unittest.mock import patch

from norlan import dashboard, progress, subtitle_sync


def synthetic_state():
    return {"id": "synthetic-post", "title": "合成后期测试", "mode": "demo", "status": "ready", "paused": False,
            "plan": {"status": "approved", "version": 1}, "assets": [], "jobs": {},
            "budget": {"currency": "CNY"}, "settings": {"postproduction": {"enabled": True}, "secret": "PRIVATE-CONFIG"},
            "music_library": {"tracks": [{"id": "theme-1", "title": "合成测试音乐", "version": 1,
                "file": "music/library/theme-v1.mp3", "sha256": "a" * 64, "duration": 60,
                "status": "awaiting_approval", "provider_raw": "PRIVATE-PROVIDER", "provenance": {"key": "PRIVATE-KEY"}}]},
            "episodes": [{"id": "E1", "title": "合成测试第一集", "status": "generating", "version": 0,
                "shots": [{"id": "S1", "status": "accepted", "version": 1, "duration": 5,
                    "file": "media/E1/S1-v1.mp4", "sha256": "b" * 64,
                    "approved_sha256": "b" * 64, "approved_version": 1, "approved_at": "2026-01-01T00:00:00Z",
                    "review": {"human_override": True}}],
                "postproduction": {"status": "awaiting_music_approval", "revision": 1, "final_ready": False,
                    "subtitles": {"status": "corrected", "file": "exports/E1.srt", "engine": "本地识别 + GPT 订阅校对",
                        "segments": [{"start": 0.5, "end": 2, "text": "合成对白", "provider_raw": "PRIVATE-ASR"}]},
                    "music": {"status": "awaiting_approval", "can_generate": False,
                        "summary": "合成音乐安排", "cues": [{"track_id": "theme-1", "start": 0, "end": 5,
                            "source_start": 0, "gain_db": -18, "fade_in": 0.2, "fade_out": 0.5,
                            "reason": "合成气氛", "provider_raw": "PRIVATE-CUE"}]},
                    "provider_raw": "PRIVATE-POST"}}]}


class PostproductionProjectionTests(unittest.TestCase):
    def expose_verified(self, state):
        # Synthetic acoustic observations verify the real report/projection
        # contract; they do not test recognition or claim acoustic accuracy.
        post = state["episodes"][0]["postproduction"]
        subtitles = post["subtitles"]
        post["source_key"], subtitles["audio_sha256"] = "e" * 64, "d" * 64
        checker = {"name": "synthetic-independent-ctc", "version": 1}
        observations = {"independent": True, "method": "independent_ctc", "full_audio_checked": True,
            "audio_sha256": subtitles["audio_sha256"], "checker_identity": checker,
            "contradictions": [], "cues": [{"cue_index": index, "start": cue["start"], "end": cue["end"],
                "text": cue["text"], "coverage": 1.0, "consistent": True}
                for index, cue in enumerate(subtitles["segments"])]}
        subtitles["sync_check"] = subtitle_sync.evaluate_alignment(subtitles["segments"], observations, 5.0,
            audio_sha256=subtitles["audio_sha256"], source_key=post["source_key"], checker_identity=checker)
        self.assertEqual(subtitles["sync_check"]["status"], "passed")
        return dashboard.public_state(state)

    def test_projection_keeps_media_and_editorial_plan_without_provider_internals(self):
        state = synthetic_state()
        before = copy.deepcopy(state)
        public = dashboard.public_state(state)
        post = public["episodes"][0]["postproduction"]
        self.assertTrue(post["enabled"])
        self.assertTrue(post["ready"])
        self.assertFalse(post["final_ready"])
        self.assertEqual(post["subtitles"]["segments"], [{"start": 0.5, "end": 2, "text": "合成对白"}])
        self.assertEqual(post["music"]["cues"][0]["track_id"], "theme-1")
        self.assertEqual(public["music_library"]["tracks"][0]["status"], "awaiting_approval")
        self.assertNotIn("PRIVATE-", json.dumps(public))
        self.assertEqual(state, before)

    def test_old_final_file_does_not_enable_new_final_approval(self):
        state = synthetic_state()
        episode = state["episodes"][0]
        episode.update(status="awaiting_review", version=1, file="exports/old.mp4", sha256="c" * 64)
        self.assertFalse(dashboard.public_state(state)["episodes"][0]["postproduction"]["final_ready"])
        self.assertNotEqual(progress.episode_progress(state, episode)["percent"], 100)
        episode["postproduction"].update(final_ready=True, status="awaiting_final_approval",
            final_file=episode["file"], final_version=episode["version"], final_sha256=episode["sha256"])
        post = self.expose_verified(state)["episodes"][0]["postproduction"]
        self.assertTrue(post["final_ready"])
        self.assertTrue(post["final_available"])
        self.assertEqual((post["final_file"], post["final_version"], post["final_sha256"]),
                         (episode["file"], episode["version"], episode["sha256"]))
        self.assertEqual(progress.episode_progress(state, episode)["percent"], 100)

    def test_music_reports_publish_bounded_safe_summary_without_audio_evidence(self):
        state = synthetic_state()
        post = state["episodes"][0]["postproduction"]
        post["music"]["analysis"] = {"status": "analyzed", "policy_version": "music-acoustic-v1",
            "summary": "已分析实际音频", "tracks_analyzed": 6, "cues_analyzed": 13,
            "windows": [{"file": "/private/audio.wav"}], "raw": "PRIVATE-MUSIC-ANALYSIS"}
        post["music_mix_check"] = {"status": "passed", "policy_version": "music-acoustic-v1",
            "summary": "数值检查通过 token=sk-abcdefghijklmno", "repair_attempts": 1,
            "normalization_target_lufs": -18, "duck_max_reduction_db": 5,
            "source_analysis_used": True, "checked_at": "2026-10-20T00:00:00Z",
            "file_sha256": "PRIVATE-MIX-HASH", "evidence": {"file": "/private/stem.wav"},
            "issues": [{"start": 1.5, "end": 3.5, "message": "合成测试疑点", "raw": "PRIVATE-ISSUE"},
                       {"start": float("nan"), "end": float("inf"), "message": "无效时间"},
                       {"start": True, "end": 3, "message": "布尔值不是时间"}]}
        before = copy.deepcopy(state)
        public = dashboard.public_state(state)
        safe = public["episodes"][0]["postproduction"]
        self.assertEqual(safe["music"]["analysis"], {"status": "analyzed", "policy_version": "music-acoustic-v1",
            "summary": "已分析实际音频", "tracks_analyzed": 6, "cues_analyzed": 13})
        self.assertEqual(safe["music_mix_check"]["repair_attempts"], 1)
        self.assertEqual(safe["music_mix_check"]["normalization_target_lufs"], -18)
        self.assertEqual(safe["music_mix_check"]["duck_max_reduction_db"], 5)
        self.assertTrue(safe["music_mix_check"]["source_analysis_used"])
        self.assertEqual(safe["music_mix_check"]["issues"][0], {"start": 1.5, "end": 3.5, "message": "合成测试疑点"})
        self.assertTrue(all("start" not in issue for issue in safe["music_mix_check"]["issues"][1:]))
        self.assertNotIn("PRIVATE-", json.dumps(public))
        self.assertNotIn("sk-abcdefghijklmno", json.dumps(public))
        self.assertEqual(state, before)

    def test_music_report_invalid_and_oversized_values_are_not_passed_to_browser(self):
        state = synthetic_state()
        post = state["episodes"][0]["postproduction"]
        post["music"]["analysis"] = {"summary": "长" * 3000, "tracks_analyzed": True, "cues_analyzed": -1}
        post["music_mix_check"] = {"status": "private-state", "summary": "长" * 3000,
            "repair_attempts": True, "normalization_target_lufs": float("nan"),
            "duck_max_reduction_db": float("inf"), "source_analysis_used": "true",
            "issues": [{"message": "长" * 3000}] * 300}
        safe = dashboard.public_state(state)["episodes"][0]["postproduction"]
        self.assertEqual(safe["music_mix_check"]["status"], "needs_attention")
        self.assertLessEqual(len(safe["music_mix_check"]["summary"]), 401)
        self.assertEqual(len(safe["music_mix_check"]["issues"]), 100)
        self.assertTrue({"repair_attempts", "normalization_target_lufs", "duck_max_reduction_db", "source_analysis_used"}.isdisjoint(safe["music_mix_check"]))
        self.assertTrue({"tracks_analyzed", "cues_analyzed"}.isdisjoint(safe["music"]["analysis"]))

    def test_old_music_state_without_reports_stays_compatible(self):
        safe = dashboard.public_state(synthetic_state())["episodes"][0]["postproduction"]
        self.assertNotIn("music_mix_check", safe)
        self.assertNotIn("analysis", safe["music"])

    def test_new_mix_report_must_match_current_export_to_open_final_human_gate(self):
        from norlan.core import _json_digest
        state = synthetic_state()
        episode = state["episodes"][0]
        episode.update(status="awaiting_review", version=1, file="exports/E1-v1.mp4", sha256="c" * 64)
        post = episode["postproduction"]
        post.update(status="awaiting_final_approval", final_ready=True,
            final_file=episode["file"], final_version=1, final_sha256=episode["sha256"])
        for status in ("passed", "manual"):
            with self.subTest(status=status):
                report = {"status": status, "summary": "合成绑定报告", "final_sha256": episode["sha256"]}
                post["music_mix_check"] = report
                episode["render_manifest"] = {"postproduction": {"music_mix_hash": _json_digest(report)}}
                public = self.expose_verified(state)["episodes"][0]["postproduction"]
                self.assertTrue(public["final_ready"])
                report["summary"] = "导出后的改动"
                public = dashboard.public_state(state)["episodes"][0]["postproduction"]
                self.assertTrue(public["final_available"])
                self.assertFalse(public["final_ready"])
                self.assertEqual(public["music_mix_check"]["status"], "needs_attention")
                self.assertIn("不一致", public["music_mix_check"]["summary"])
                report["final_sha256"] = "d" * 64
                episode["render_manifest"]["postproduction"]["music_mix_hash"] = _json_digest(report)
                self.assertFalse(dashboard.public_state(state)["episodes"][0]["postproduction"]["final_ready"])

    def test_failed_or_missing_new_mix_check_retains_current_preview_without_approval(self):
        from norlan.core import _json_digest
        state = synthetic_state()
        episode = state["episodes"][0]
        episode.update(status="awaiting_review", version=1, file="exports/E1-v1.mp4", sha256="c" * 64)
        post = episode["postproduction"]
        report = {"status": "needs_attention", "summary": "合成音乐过轻", "final_sha256": episode["sha256"]}
        post.update(status="music_mix_needs_attention", final_ready=True,
            final_file=episode["file"], final_version=1, final_sha256=episode["sha256"], music_mix_check=report)
        episode["render_manifest"] = {"postproduction": {"music_mix_hash": _json_digest(report)}}
        public = self.expose_verified(state)["episodes"][0]["postproduction"]
        self.assertTrue(public["final_available"])
        self.assertFalse(public["final_ready"])
        self.assertIn("过轻", public["music_mix_check"]["summary"])
        del post["music_mix_check"]
        public = dashboard.public_state(state)["episodes"][0]["postproduction"]
        self.assertTrue(public["final_available"])
        self.assertFalse(public["final_ready"])
        self.assertIn("缺失", public["music_mix_check"]["summary"])

    def test_active_mix_progress_is_not_checked_against_retired_export_manifest(self):
        from norlan.core import _json_digest
        state = synthetic_state()
        episode = state["episodes"][0]
        old_report = {"status": "passed", "final_sha256": "c" * 64}
        episode.update(status="postproduction", version=1, file="exports/E1-v1.mp4", sha256="c" * 64,
            render_manifest={"postproduction": {"music_mix_hash": _json_digest(old_report)}})
        post = episode["postproduction"]
        post.update(final_ready=False, final_file=episode["file"], final_version=1,
            final_sha256=episode["sha256"])
        for status in ("checking", "repairing"):
            with self.subTest(status=status):
                post.update(status="checking_music_mix", music_mix_check={"status": status,
                    "policy_version": 1, "summary": "新混音正在处理", "binding": {"music_key": "new-source-key"}})
                safe = dashboard.public_state(state)["episodes"][0]["postproduction"]
                self.assertFalse(safe["final_available"])
                self.assertFalse(safe["final_ready"])
                self.assertEqual(safe["music_mix_check"]["status"], status)
                self.assertEqual(safe["music_mix_check"]["summary"], "新混音正在处理")
                self.assertEqual(safe["music_mix_check"]["policy_version"], "1")
                self.assertNotIn("binding", safe["music_mix_check"])

    def test_completed_legacy_postproduction_exposes_existing_final_without_mutation(self):
        state = synthetic_state()
        episode = state["episodes"][0]
        episode.update(status="awaiting_review", version=1, file="exports/E1-v1.mp4", sha256="c" * 64,
                       subtitles_burned=True,
                       render_manifest={"subtitles": [{"start": 0.5, "end": 2, "text": "合成对白"}]})
        episode["postproduction"].update(status="awaiting_final_approval", final_ready=True,
            final_file=episode["file"], final_version=1, final_sha256=episode["sha256"])
        before = copy.deepcopy(state)
        exposed = dashboard.public_state(state)["episodes"][0]
        self.assertTrue(exposed["postproduction"]["final_available"])
        self.assertFalse(exposed["postproduction"]["final_ready"], "an old export is viewable but needs automatic checking before a new approval")
        self.assertEqual(exposed["postproduction"]["final_file"], "exports/E1-v1.mp4")
        self.assertTrue(exposed["has_subtitles"])
        self.assertTrue(exposed["subtitles_burned"])
        self.assertEqual(state, before)
        episode["status"] = "approved"
        approved = dashboard.public_state(state)["episodes"][0]
        self.assertTrue(approved["postproduction"]["final_available"])
        self.assertEqual(approved["status"], "approved", "upgrading does not revoke a human decision")

    def test_sync_diagnostics_are_bounded_sanitized_and_not_verification_internals(self):
        state = synthetic_state()
        report = {"status": "needs_attention", "summary": "检查完成，仍有疑点", "attempts": 2,
                  "max_attempts": 2, "checked_cues": 7, "total_cues": 8,
                  "source_audio_sha256": "PRIVATE-AUDIO", "source_key": "PRIVATE-SOURCE",
                  "caption_hash": "PRIVATE-CAPTIONS", "evidence": {"path": "/private/model.json"},
                  "issues": [{"start": 6.5, "end": 8.9, "reason": "字幕开始早于实际人声 token=sk-abcdefghijklmno",
                              "code": "early_start", "cue_index": 0, "raw": "PRIVATE-ASR"},
                             {"start": float("nan"), "end": float("inf"), "reason": "时间无效"},
                             {"start": True, "end": 4, "reason": "布尔值不能当作时间"}]}
        state["episodes"][0]["postproduction"]["subtitles"]["sync_check"] = report
        public = dashboard.public_state(state)
        safe = public["episodes"][0]["postproduction"]["subtitles"]["sync_check"]
        self.assertEqual(safe["attempts"], 2)
        self.assertEqual((safe["checked_cues"], safe["total_cues"]), (7, 8))
        self.assertEqual(safe["issues"][0]["start"], 6.5)
        self.assertEqual(safe["issues"][0]["cue_index"], 0)
        self.assertNotIn("sk-abcdefghijklmno", safe["issues"][0]["reason"])
        self.assertTrue(all("start" not in item for item in safe["issues"][1:]))
        self.assertNotIn("PRIVATE-", json.dumps(public))
        self.assertNotIn("evidence", safe)
        self.assertNotIn("source_audio_sha256", safe)
        report.update(summary="过长" * 1000, issues=report["issues"] * 300, attempts=True, checked_cues=-1)
        safe = dashboard.public_state(state)["episodes"][0]["postproduction"]["subtitles"]["sync_check"]
        self.assertLessEqual(len(safe["summary"]), 401)
        self.assertLessEqual(len(safe["issues"]), 200)
        self.assertNotIn("attempts", safe)
        self.assertNotIn("checked_cues", safe)

    def test_passed_but_stale_sync_report_never_opens_final_approval(self):
        state = synthetic_state()
        episode = state["episodes"][0]
        episode.update(status="awaiting_review", version=1, file="exports/E1-v1.mp4", sha256="c" * 64)
        episode["postproduction"].update(status="awaiting_final_approval", final_ready=True,
            final_file=episode["file"], final_version=1, final_sha256=episode["sha256"])
        episode["postproduction"]["subtitles"]["sync_check"] = {"status": "passed", "summary": "旧检查"}
        with patch("norlan.subtitle_sync.report_matches", return_value=False):
            post = dashboard.public_state(state)["episodes"][0]["postproduction"]
        self.assertTrue(post["final_available"])
        self.assertFalse(post["final_ready"])
        self.assertEqual(post["subtitles"]["sync_check"]["status"], "needs_attention")
        self.assertIn("不一致", post["subtitles"]["sync_check"]["summary"])

    def test_invalid_report_status_is_not_presented_as_passed(self):
        state = synthetic_state()
        state["episodes"][0]["postproduction"]["subtitles"]["sync_check"] = {
            "status": "private-internal-state", "summary": "未知检查状态"}
        post = dashboard.public_state(state)["episodes"][0]["postproduction"]
        self.assertEqual(post["subtitles"]["sync_check"]["status"], "needs_attention")

    def test_stale_or_unbound_final_preview_is_not_exposed(self):
        state = synthetic_state()
        episode = state["episodes"][0]
        episode.update(status="awaiting_review", version=2, file="exports/E1-v2.mp4", sha256="c" * 64)
        episode["postproduction"].update(status="awaiting_final_approval", final_ready=True,
            final_file=episode["file"], final_version=2, final_sha256=episode["sha256"])
        for change in ({"final_file": "exports/E1-v1.mp4"}, {"final_sha256": "d" * 64},
                       {"final_version": 1}, {"final_version": True}, {"final_ready": False},
                       {"status": "rendering"}, {"status": "error"}, {"final_file": None}):
            with self.subTest(change=change):
                candidate = copy.deepcopy(state)
                candidate["episodes"][0]["postproduction"].update(change)
                post = dashboard.public_state(candidate)["episodes"][0]["postproduction"]
                self.assertFalse(post["final_ready"])
                self.assertTrue({"final_file", "final_version", "final_sha256"}.isdisjoint(post))
        for change in ({"status": "postproduction"}, {"version": 3}, {"sha256": "d" * 64}):
            with self.subTest(episode_change=change):
                candidate = copy.deepcopy(state)
                candidate["episodes"][0].update(change)
                self.assertFalse(dashboard.public_state(candidate)["episodes"][0]["postproduction"]["final_ready"])
        episode["shots"][0]["status"] = "review_required"
        self.assertFalse(dashboard.public_state(state)["episodes"][0]["postproduction"]["final_ready"])

    def test_final_preview_rejects_invalid_even_matching_media_identity(self):
        for file, version, sha in (("/private/final.mp4", 1, "c" * 64),
                                  ("exports/../final.mp4", 1, "c" * 64),
                                  ("exports/final.txt", 1, "c" * 64),
                                  ("exports/final.mp4", True, "c" * 64),
                                  ("exports/final.mp4", 1, "invalid")):
            with self.subTest(file=file, version=version, sha=sha):
                state = synthetic_state()
                episode = state["episodes"][0]
                episode.update(status="awaiting_review", file=file, version=version, sha256=sha)
                episode["postproduction"].update(status="awaiting_final_approval", final_ready=True,
                    final_file=file, final_version=version, final_sha256=sha)
                post = dashboard.public_state(state)["episodes"][0]["postproduction"]
                self.assertFalse(post["final_ready"])
                self.assertNotIn("final_file", post)

    def test_unapproved_segment_locks_postproduction_and_config_is_explained(self):
        state = synthetic_state()
        state["episodes"][0]["shots"][0]["status"] = "review_required"
        state["music_library"]["tracks"] = []
        post = dashboard.public_state(state)["episodes"][0]["postproduction"]
        self.assertFalse(post["ready"])
        self.assertIn("首次导入音乐", " ".join(post["missing_configuration"]))

    def test_postproduction_progress_names_actual_human_gate_and_failures(self):
        state = synthetic_state()
        episode = state["episodes"][0]
        for status, expected in (("awaiting_music_approval", "试听"), ("configuration_required", "补充配置"),
                                 ("failed", "后期已停止"), ("transcribing", "本地识别"),
                                 ("correcting_subtitles", "校对"), ("arranging_music", "编排")):
            episode["postproduction"]["status"] = status
            actual = progress.episode_progress(state, episode)
            self.assertIn(expected, actual["label"])
            self.assertLess(actual["percent"], 100)


class PostproductionHttpTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name).resolve()
        self.state = synthetic_state()
        self.core = self.enterContext(patch.object(dashboard, "_core")).return_value
        self.core.load_project.return_value = self.state
        self.run = self.enterContext(patch.object(dashboard, "_run"))
        self.server = dashboard.make_server(self.root, port=0)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.addCleanup(self.shutdown)

    def shutdown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)

    def request(self, payload=None, path="/api/action", headers=None, method="POST"):
        connection = http.client.HTTPConnection("127.0.0.1", self.server.server_address[1], timeout=3)
        body = json.dumps(payload) if payload is not None else None
        request_headers = {"Content-Type": "application/json", "X-Norlan-Token": self.server.context.token}
        request_headers.update(headers or {})
        connection.request(method, path, body, request_headers)
        response = connection.getresponse()
        status, result = response.status, response.read()
        connection.close()
        return status, result

    def test_music_human_decision_is_bound_to_track_version_and_hash(self):
        payload = {"action": "music_approve", "id": "theme-1", "version": 1, "sha256": "a" * 64}
        self.assertEqual(self.request(payload)[0], 200)
        self.core.approve_music.assert_called_once_with(self.root, "theme-1", 1, "a" * 64)
        self.assertEqual(self.request({**payload, "sha256": "bad"})[0], 409)
        self.assertEqual(self.request({**payload, "version": True})[0], 409)
        self.assertEqual(self.request({**payload, "action": "music_reject", "feedback": " "})[0], 409)
        self.assertEqual(self.request({**payload, "action": "music_reject", "feedback": "合成试听意见"})[0], 200)
        self.core.reject_music.assert_called_once_with(self.root, "theme-1", 1, "a" * 64, "合成试听意见")
        self.run.assert_not_called()

    def test_postproduction_edits_never_use_video_rework_action(self):
        for target in ("continue", "all", "music", "subtitles"):
            payload = {"action": "postproduction", "id": "E1", "version": 0, "target": target, "feedback": "合成意见"}
            self.assertEqual(self.request(payload)[0], 200)
        self.assertEqual(self.core.request_postproduction.call_count, 4)
        self.core.reject.assert_not_called()
        self.run.assert_not_called()
        self.assertEqual(self.request({**payload, "target": "video"})[0], 409)

    def test_upload_has_separate_limit_and_no_user_supplied_filesystem_path(self):
        payload = {"id": "E1", "version": 0, "title": "合成测试", "data_url": "data:audio/wav;base64,UklGRg=="}
        self.assertEqual(self.request(payload, "/api/music-upload")[0], 200)
        self.core.import_music.assert_called_once_with(self.root, "E1", 0, payload["data_url"], "合成测试")
        self.assertEqual(self.request(payload, "/api/music-upload", {"Content-Length": str(dashboard.MAX_MUSIC_BODY + 1)})[0], 413)
        self.assertEqual(self.request({"id": "E1", "version": 0, "title": "合成测试", "path": "/private/audio.wav"}, "/api/music-upload")[0], 409)
        self.assertEqual(self.request(payload, "/api/music-upload", {"X-Norlan-Token": "bad"})[0], 403)
        self.assertEqual(self.request(payload, "/api/music-upload", {"Origin": "https://invalid.example"})[0], 403)

    def test_explicit_generation_and_upload_block_during_active_production(self):
        self.assertEqual(self.request({"action": "music_generate", "id": "E1", "version": 0})[0], 200)
        self.core.request_music_generation.assert_called_once_with(self.root, "E1", 0)
        self.server.context.running = True
        self.assertEqual(self.request({"action": "postproduction", "id": "E1", "version": 0, "target": "continue"})[0], 409)
        self.core.request_postproduction.assert_not_called()

    def test_audition_media_serve_only_explicit_project_audio(self):
        path = self.root / "music/library/theme-v1.mp3"
        path.parent.mkdir(parents=True)
        path.write_bytes(b"synthetic-not-real-audio")
        self.assertEqual(self.request(None, "/media/music/library/theme-v1.mp3", method="GET")[0], 200)
        (path.parent / "unreferenced.mp3").write_bytes(b"private")
        self.assertEqual(self.request(None, "/media/music/library/unreferenced.mp3", method="GET")[0], 404)

    def test_sync_approval_gate_errors_are_explicit_and_do_not_start_production(self):
        for message in ("字幕同步自动检查尚未通过，暂不能最终批准；请查看本集字幕检查报告。",
                        "字幕同步检查记录与当前音轨或字幕不一致，请重新检查后再批准。"):
            with self.subTest(message=message):
                self.core.approve.side_effect = ValueError(message)
                status, body = self.request({"action": "approve", "kind": "episode", "id": "E1", "version": 1})
                self.assertEqual(status, 409)
                self.assertEqual(json.loads(body)["error"], message)
        self.run.assert_not_called()


@unittest.skipUnless(shutil.which("node"), "Node.js required for synthetic DOM test")
class PostproductionRenderingTests(unittest.TestCase):
    def test_panel_is_after_rough_cut_and_human_final_gate_requires_completed_post(self):
        html = Path(dashboard.__file__).with_name("dashboard.html").read_text()
        sources = []
        for name in ("el", "mediaURL", "appendPrompt", "episodeCard", "postproductionFeedback", "musicTrack", "postproductionEditor", "subtitleSyncReport", "musicMixReport", "postproductionPanel"):
            match = re.search(r"(?:async )?function " + name + r"\(", html)
            next_match = re.search(r"\n(?:async )?function ", html[match.start() + 1:])
            end = match.start() + 1 + next_match.start() if next_match else len(html)
            sources.append(html[match.start():end])
        program = r'''
const assert=require('node:assert/strict'),vm=require('node:vm');
const sources=JSON.parse(require('node:fs').readFileSync(0,'utf8'));
class Element{constructor(tag){this.tag=tag;this.children=[];this.dataset={};this.listeners={};this.textContent='';}append(...nodes){this.children.push(...nodes);}setAttribute(k,v){this[k]=v;}addEventListener(k,v){this.listeners[k]=v;}}
const actions=[],state={paused:false,dashboard:{running:false},episodes:[{id:'E1'}],music_library:{tracks:[{id:'theme-1',title:'合成音乐',version:2,sha256:'a'.repeat(64),file:'music/library/theme.mp3',status:'awaiting_approval'}]}};
const context={state,assert,actions,document:{createElement:tag=>new Element(tag),createTextNode:text=>({tag:'text',textContent:text})},badge:()=>new Element('badge'),reviewDetails:()=>null,appendEpisodeProduction:card=>card.append(Object.assign(new Element('section'),{className:'synthetic-rough-cut'})),perform:(...args)=>actions.push(args),uploadMusic:()=>{},showRevision:()=>{}};
vm.runInNewContext(sources.join('\n')+`
const walk=node=>[node,...(node.children||[]).flatMap(walk)];
const episode={id:'E1',title:'第一集',status:'awaiting_review',version:1,shots:[],postproduction:{enabled:true,ready:true,final_ready:false,status:'awaiting_music_approval',subtitles:{segments:[{start:0,end:1,text:'合成字幕'}]},music:{can_generate:false}}};
let card=episodeCard(episode,0),nodes=walk(card);assert.equal(nodes.some(n=>n.tag==='button'&&n.textContent==='通过本集，完成制作'),false);
assert.ok(card.children.findIndex(n=>n.className==='postproduction')>card.children.findIndex(n=>n.className==='synthetic-rough-cut'));
const audio=nodes.find(n=>n.tag==='audio');assert.equal(audio.src,'/media/music/library/theme.mp3');
nodes.find(n=>n.tag==='button'&&n.textContent==='确认这首音乐并继续').listeners.click();assert.equal(actions.at(-1)[0].action,'music_approve');assert.equal(actions.at(-1)[0].version,2);assert.equal(actions.at(-1)[0].sha256,'a'.repeat(64));
assert.equal(nodes.find(n=>n.tag==='button'&&n.textContent==='继续本集后期').disabled,true);
assert.equal(nodes.find(n=>n.tag==='button'&&n.textContent==='生成一首新音乐').disabled,true);
const feedback=nodes.find(n=>n.tag==='textarea'&&n.dataset.feedbackFor==='post:E1:subtitles');feedback.value='合成字幕修改';
const subtitleSection=nodes.find(n=>n.className==='postproduction-step');walk(subtitleSection).find(n=>n.tag==='button'&&n.textContent==='提交意见并重新合成').listeners.click();assert.equal(actions.at(-1)[0].action,'postproduction');assert.equal(actions.at(-1)[0].target,'subtitles');
episode.file='exports/E1-v1.mp4';Object.assign(episode.postproduction,{final_available:true,final_ready:true,final_file:episode.file,final_version:1,subtitles:{sync_check:{status:"passed"}}});card=episodeCard(episode,0);assert.equal(walk(card).some(n=>n.tag==='button'&&n.textContent==='通过本集，完成制作'),true);
episode.postproduction.music.analysis={summary:'合成分析摘要',tracks_analyzed:6,cues_analyzed:13};episode.postproduction.music_mix_check={status:'passed',summary:'合成混音数值通过',normalization_target_lufs:-18,duck_max_reduction_db:5,source_analysis_used:true,repair_attempts:1,issues:[]};card=episodeCard(episode,0);const mix=walk(card).find(n=>n.className==='music-mix-report');assert.ok(mix);assert.equal(mix.tag,'details');assert.equal(mix.open,undefined);assert.equal(mix.children[0].textContent,'配乐检查报告 · 数值检查通过，待你试听');assert.ok(walk(mix).some(n=>n.textContent.includes('数字检查用于发现')));assert.ok(walk(mix).some(n=>n.textContent.includes('已分析实际音频：6 首，选段 13 处')));assert.ok(walk(mix).some(n=>n.textContent.includes('压低上限：5 dB')));assert.equal(actions.some(row=>row[0].action==='approve'),false);
episode.postproduction.music_mix_check={status:'manual',summary:'合成手动采用',normalization_target_lufs:-18,repair_attempts:0};const manualMix=musicMixReport(episode);assert.ok(walk(manualMix).some(n=>n.textContent.includes('未再自动分析选段、复检或改音量')));assert.equal(walk(manualMix).some(n=>n.textContent.includes('音源响度基准')),false);
episode.postproduction.music_mix_check={status:'needs_attention',issues:[{start:1,end:3,message:'合成过轻时段'}]};card=episodeCard(episode,0);assert.ok(walk(card).some(n=>n.textContent.includes('配乐数字检查仍有疑点')));assert.ok(walk(card).some(n=>n.textContent==='合成过轻时段'));delete episode.postproduction.music.analysis;delete episode.postproduction.music_mix_check;
episode.postproduction.ready=false;const locked=postproductionPanel(episode);assert.equal(walk(locked).some(n=>n.tag==='button'),false);assert.equal(walk(locked).some(n=>n.textContent.includes('等待本集所有片段通过审批')),true);
`,context);
'''
        result = subprocess.run([shutil.which("node"), "-e", program], input=json.dumps(sources), text=True,
                                capture_output=True, timeout=15)
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
