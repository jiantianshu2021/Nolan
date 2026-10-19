"""Synthetic snapshots only: no provider requests and no media generation."""
import copy
import unittest

from norlan.progress import episode_progress


def fixture(count=25):
    episode = {"id": "E001", "status": "generating", "shots": [
        {"id": f"S{index:03}", "status": "planned", "duration": 5}
        for index in range(1, count + 1)
    ]}
    state = {"status": "ready", "plan": {"status": "approved"},
             "assets": [{"status": "approved"}], "episodes": [episode], "jobs": {}}
    return state, episode


def accept(shot):
    shot.update(status="accepted", file=f"media/{shot['id']}.mp4", sha256="synthetic-hash")


class ProgressTests(unittest.TestCase):
    def test_first_accepted_shot_of_twenty_five(self):
        state, ep = fixture()
        accept(ep["shots"][0])
        result = episode_progress(state, ep)
        self.assertEqual(result["percent"], 3.6)
        self.assertEqual(result["completed_shots"], 1)
        self.assertEqual(result["current_shot"], "S002")
        self.assertEqual(result["total_shots"], 25)
        self.assertIn("计划时长", result["basis"])

    def test_weights_use_duration_instead_of_shot_count(self):
        state, ep = fixture(2)
        ep["shots"][0]["duration"] = 4
        ep["shots"][1]["duration"] = 12
        accept(ep["shots"][0])
        self.assertEqual(episode_progress(state, ep)["percent"], 22.5)

    def test_submitted_task_does_not_gain_percent_over_time(self):
        state, ep = fixture()
        ep["shots"][0].update(status="generating", job_id="job-1")
        state["jobs"]["job-1"] = {"state": "submitted", "created_at": "2026-10-20T01:00:00Z"}
        first = episode_progress(state, ep, now="2026-10-20T01:01:00Z")
        later = episode_progress(state, ep, now="2026-10-20T02:00:00Z")
        self.assertEqual(first["percent"], later["percent"])
        self.assertEqual(first["elapsed_seconds"], 60)
        self.assertEqual(later["elapsed_seconds"], 3600)
        self.assertTrue(later["has_active_task"])
        self.assertEqual(later["stage"], "shot_generation")
        self.assertIn("视频引擎", later["label"])

    def test_pause_retains_completed_work_and_mentions_inflight_task(self):
        state, ep = fixture(2)
        accept(ep["shots"][0])
        ep["shots"][1]["job_id"] = "pending"
        state["jobs"]["pending"] = {"state": "submitted"}
        state["paused"] = True
        result = episode_progress(state, ep)
        self.assertEqual(result["percent"], 45)
        self.assertEqual(result["stage"], "paused")
        self.assertIn("可能仍在生成", result["label"])
        self.assertFalse(result["is_working"])

    def test_rework_reduces_progress_and_ignores_stale_render_hint(self):
        state, ep = fixture(2)
        for shot in ep["shots"]:
            accept(shot)
        ep["progress_stage"] = {"stage": "rendering"}
        self.assertEqual(episode_progress(state, ep)["percent"], 96)
        ep["shots"][0].update(status="planned")
        ep["status"] = "revision_requested"
        result = episode_progress(state, ep)
        self.assertEqual(result["percent"], 45)
        self.assertNotEqual(result["stage"], "rendering")

    def test_accepted_without_durable_artifact_is_not_complete(self):
        state, ep = fixture(1)
        ep["shots"][0]["status"] = "accepted"
        self.assertEqual(episode_progress(state, ep)["completed_shots"], 0)

    def test_postproduction_milestones_and_final_human_gate(self):
        state, ep = fixture(1)
        accept(ep["shots"][0])
        for stage, percent in (("music", 90), ("subtitles", 93), ("rendering", 96)):
            ep["progress_stage"] = {"stage": stage}
            result = episode_progress(state, ep)
            self.assertEqual(result["percent"], percent)
            self.assertEqual(result["stage"], stage)
        ep.update(file="exports/E001.mp4", sha256="synthetic-final", status="awaiting_review")
        result = episode_progress(state, ep)
        self.assertEqual(result["percent"], 100)
        self.assertTrue(result["waiting_for_human"])
        self.assertEqual(result["stage"], "awaiting_review")
        ep["status"] = "approved"
        result = episode_progress(state, ep)
        self.assertEqual(result["percent"], 100)
        self.assertFalse(result["waiting_for_human"])

    def test_old_render_file_during_caption_rework_never_looks_complete(self):
        state, ep = fixture(1)
        accept(ep["shots"][0])
        ep.update(file="exports/E001-v1.mp4", sha256="old-render", status="revision_requested")
        self.assertEqual(episode_progress(state, ep)["percent"], 90)

    def test_final_state_without_hash_does_not_reach_one_hundred(self):
        state, ep = fixture(1)
        accept(ep["shots"][0])
        ep.update(status="awaiting_review", file="exports/E001.mp4")
        self.assertLess(episode_progress(state, ep)["percent"], 100)

    def test_next_episode_waits_for_previous_approval(self):
        state, first = fixture(1)
        second = copy.deepcopy(first)
        second["id"] = "E002"
        second["progress_stage"] = {"stage": "shot_preparation"}
        state["episodes"].append(second)
        result = episode_progress(state, second)
        self.assertEqual(result["stage"], "awaiting_previous_episode")
        self.assertFalse(result["is_working"])
        self.assertTrue(result["waiting_for_human"])

    def test_asset_and_plan_gates(self):
        state, ep = fixture()
        state["assets"][0]["status"] = "candidate"
        self.assertEqual(episode_progress(state, ep)["stage"], "awaiting_assets")
        state["status"] = "awaiting_asset_input"
        self.assertEqual(episode_progress(state, ep)["stage"], "awaiting_asset_input")
        state["plan"]["status"] = "candidate"
        self.assertEqual(episode_progress(state, ep)["stage"], "awaiting_plan")

    def test_unknown_and_failed_calls_are_explicit(self):
        state, ep = fixture(1)
        ep["shots"][0]["job_id"] = "current"
        state["jobs"]["current"] = {"state": "uncertain", "accounting": "unknown"}
        result = episode_progress(state, ep)
        self.assertEqual(result["stage"], "unknown")
        self.assertTrue(result["waiting_for_human"])
        self.assertFalse(result["has_active_task"])
        state["jobs"]["current"] = {"state": "failed", "accounting": "spent"}
        self.assertEqual(episode_progress(state, ep)["stage"], "failed")

    def test_blocked_error_does_not_expose_raw_provider_payload(self):
        state, ep = fixture()
        state.update(status="blocked", last_error="Authorization=secret")
        result = episode_progress(state, ep)
        self.assertEqual(result["stage"], "blocked")
        self.assertNotIn("secret", str(result))

    def test_human_shot_review_is_not_accepted_progress(self):
        state, ep = fixture(1)
        ep["shots"][0].update(status="review_required", file="shot.mp4", sha256="synthetic")
        result = episode_progress(state, ep)
        self.assertEqual(result["percent"], 0)
        self.assertEqual(result["stage"], "review_required")
        self.assertTrue(result["waiting_for_human"])

    def test_download_then_technical_check_then_review_are_distinct(self):
        state, ep = fixture(1)
        shot = ep["shots"][0]
        shot.update(status="generating", job_id="current")
        state["jobs"]["current"] = {"state": "download_pending"}
        self.assertEqual(episode_progress(state, ep)["stage"], "shot_download")
        state["jobs"]["current"]["state"] = "completed"
        self.assertEqual(episode_progress(state, ep)["stage"], "shot_checks")
        shot["status"] = "reviewing"
        self.assertEqual(episode_progress(state, ep)["stage"], "shot_review")

    def test_elapsed_uses_explicit_now_and_preserves_state(self):
        state, ep = fixture(1)
        ep["progress_started_at"] = "2026-10-20T01:00:00Z"
        ep["progress_stage"] = {"stage": "shot_preparation", "shot_id": "S001",
                                "started_at": "2026-10-20T01:05:00Z"}
        before = copy.deepcopy(state)
        self.assertIsNone(episode_progress(state, ep)["elapsed_seconds"])
        result = episode_progress(state, ep, now="2026-10-20T01:06:00Z")
        self.assertEqual(result["elapsed_seconds"], 60)
        self.assertEqual(result["episode_elapsed_seconds"], 360)
        self.assertEqual(state, before)

    def test_bad_or_empty_durations_do_not_produce_nan(self):
        state, ep = fixture(2)
        ep["shots"][0]["duration"] = float("nan")
        ep["shots"][1]["duration"] = -1
        accept(ep["shots"][0])
        self.assertEqual(episode_progress(state, ep)["percent"], 45)
        ep["shots"] = []
        self.assertEqual(episode_progress(state, ep)["percent"], 0)


if __name__ == "__main__":
    unittest.main()
