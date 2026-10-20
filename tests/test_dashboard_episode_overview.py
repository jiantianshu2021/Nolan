"""Synthetic episode navigation projections; no providers or real projects."""
import copy
import json
import unittest
from unittest.mock import patch

from norlan import core, dashboard


def synthetic_episode(index, status="planned", **values):
    return {"id": f"E{index}", "title": f"合成测试第 {index} 集", "status": status,
            "version": 0, "shots": [], **values}


def synthetic_state(*episodes, **values):
    return {"id": "synthetic-episode-overview", "title": "合成分集总览测试",
            "mode": "demo", "status": "ready", "paused": False,
            "plan": {"status": "approved", "version": 1}, "assets": [],
            "settings": {}, "budget": {"limit": 0, "currency": "CNY"},
            "jobs": {}, "episodes": list(episodes), **values}


class EpisodeOverviewProjectionTests(unittest.TestCase):
    def test_completed_single_episode_does_not_invent_second_episode(self):
        state = synthetic_state(synthetic_episode(1, "approved"), status="complete")
        public = dashboard.public_state(state)
        navigation = public["episode_navigation"]
        self.assertEqual(navigation, {
            "total": 1, "approved_count": 1, "active_episode_id": None,
            "next_episode_number": 2, "all_planned_complete": True,
            "reason": "全部已规划集已完成，尚未规划第 2 集。"})
        self.assertEqual([episode["id"] for episode in public["episodes"]], ["E1"])
        self.assertFalse(public["episodes"][0]["navigation"]["can_produce"])
        self.assertIn("可查看", public["episodes"][0]["navigation"]["locked_reason"])

    def test_empty_plan_is_not_completed_and_has_no_phantom_active_episode(self):
        public = dashboard.public_state(synthetic_state())
        navigation = public["episode_navigation"]
        self.assertEqual((navigation["total"], navigation["approved_count"]), (0, 0))
        self.assertFalse(navigation["all_planned_complete"])
        self.assertIsNone(navigation["active_episode_id"])
        self.assertIsNone(navigation["next_episode_number"])
        self.assertIn("尚未规划", navigation["reason"])
        self.assertEqual(public["episodes"], [])

    def test_first_unapproved_episode_is_the_only_production_route(self):
        public = dashboard.public_state(synthetic_state(
            synthetic_episode(1, "approved"), synthetic_episode(2, "awaiting_review"),
            synthetic_episode(3), synthetic_episode(4)))
        navigation = public["episode_navigation"]
        self.assertEqual((navigation["total"], navigation["approved_count"]), (4, 1))
        self.assertEqual(navigation["active_episode_id"], "E2")
        self.assertIsNone(navigation["next_episode_number"])
        self.assertFalse(navigation["all_planned_complete"])
        episodes = public["episodes"]
        self.assertEqual([episode["navigation"]["index"] for episode in episodes], [1, 2, 3, 4])
        self.assertEqual([episode["navigation"]["previous_episode_id"] for episode in episodes],
                         [None, "E1", "E2", "E3"])
        self.assertEqual([episode["navigation"]["can_produce"] for episode in episodes],
                         [False, True, False, False])
        self.assertEqual(episodes[1]["navigation"]["locked_reason"], "")
        self.assertIn("前面", episodes[2]["navigation"]["locked_reason"])
        self.assertIn("前面", episodes[3]["navigation"]["locked_reason"])

    def test_previous_final_approval_unlocks_next_route_without_starting_it(self):
        state = synthetic_state(synthetic_episode(1, "awaiting_review"), synthetic_episode(2))
        before = dashboard.public_state(state)
        self.assertEqual(before["episode_navigation"]["active_episode_id"], "E1")
        self.assertFalse(before["episodes"][1]["navigation"]["can_produce"])
        state["episodes"][0]["status"] = "approved"
        after = dashboard.public_state(state)
        self.assertEqual(after["episode_navigation"]["active_episode_id"], "E2")
        self.assertTrue(after["episodes"][1]["navigation"]["can_produce"])
        self.assertEqual(state["episodes"][1]["status"], "planned")
        self.assertEqual(state["jobs"], {})

    def test_later_approval_cannot_skip_unapproved_predecessor(self):
        public = dashboard.public_state(synthetic_state(
            synthetic_episode(1), synthetic_episode(2, "approved"), synthetic_episode(3)))
        self.assertEqual(public["episode_navigation"]["active_episode_id"], "E1")
        self.assertEqual([episode["navigation"]["can_produce"] for episode in public["episodes"]],
                         [True, False, False])

    def test_all_planned_episodes_approved_returns_overview_completion(self):
        public = dashboard.public_state(synthetic_state(
            synthetic_episode(1, "approved"), synthetic_episode(2, "approved")))
        navigation = public["episode_navigation"]
        self.assertTrue(navigation["all_planned_complete"])
        self.assertEqual(navigation["next_episode_number"], 3)
        self.assertIsNone(navigation["active_episode_id"])
        self.assertIn("尚未规划第 3 集", navigation["reason"])
        self.assertTrue(all(not episode["navigation"]["can_produce"] for episode in public["episodes"]))

    def test_declared_season_completion_does_not_suggest_another_episode(self):
        state = synthetic_state(*(synthetic_episode(index, "approved") for index in range(1, 31)),
            season_source={"episode_count": 30, "origin": "PRIVATE-SEASON-FOLDER"})
        public = dashboard.public_state(state)
        navigation = public["episode_navigation"]
        self.assertTrue(navigation["all_planned_complete"])
        self.assertTrue(navigation["season_complete"])
        self.assertIsNone(navigation["next_episode_number"])
        self.assertIsNone(navigation["active_episode_id"])
        self.assertEqual(navigation["reason"], "本季 30 集已全部完成并通过审批。")
        self.assertNotIn("PRIVATE-SEASON-FOLDER", json.dumps(public))

    def test_incomplete_or_mismatched_season_manifest_cannot_declare_completion(self):
        for manifest in ({"episode_count": 30}, {"episode_count": True}, {}, None):
            with self.subTest(manifest=manifest):
                public = dashboard.public_state(synthetic_state(synthetic_episode(1, "approved"),
                    season_source=manifest))
                navigation = public["episode_navigation"]
                self.assertFalse(navigation.get("season_complete", False))
                self.assertEqual(navigation["next_episode_number"], 2)
        public = dashboard.public_state(synthetic_state(synthetic_episode(1, "approved"),
            synthetic_episode(2), season_source={"episode_count": 2}))
        self.assertFalse(public["episode_navigation"].get("season_complete", False))
        self.assertEqual(public["episode_navigation"]["active_episode_id"], "E2")

    def test_duration_prefers_current_export_measurement_over_dry_and_plan(self):
        episode = synthetic_episode(1, "approved", file="exports/E1-v1.mp4", duration=130.566667,
            postproduction={"dry_file": "exports/E1-dry.mp4", "duration": 130.5},
            shots=[{"id": "S1", "status": "planned", "duration": 10}])
        public = dashboard.public_state(synthetic_state(episode))["episodes"][0]
        self.assertEqual(public["duration"], 130.566667)
        self.assertEqual(public["duration_source"], "measured")

    def test_duration_uses_saved_dry_measurement_when_export_is_absent(self):
        episode = synthetic_episode(1, duration=999,
            postproduction={"dry_file": "exports/E1-dry.mp4", "duration": 59.7},
            shots=[{"id": "S1", "status": "planned", "duration": 10}])
        public = dashboard.public_state(synthetic_state(episode))["episodes"][0]
        self.assertEqual(public["duration"], 59.7)
        self.assertEqual(public["duration_source"], "measured")

    def test_stale_or_invalid_durations_fall_back_to_explicit_plan(self):
        for invalid in (None, True, float("nan"), float("inf"), -1, 0, "60", 4000):
            with self.subTest(duration=invalid):
                episode = synthetic_episode(1, file="exports/E1-v1.mp4", duration=invalid,
                    postproduction={"dry_file": "exports/E1-dry.mp4", "duration": invalid},
                    shots=[{"id": "S1", "status": "planned", "duration": 10},
                           {"id": "S2", "status": "planned", "duration": 15}])
                public = dashboard.public_state(synthetic_state(episode))["episodes"][0]
                self.assertEqual(public["duration"], 25)
                self.assertEqual(public["duration_source"], "planned")

    def test_plan_duration_is_never_mislabelled_as_measured_without_saved_media(self):
        episode = synthetic_episode(1, duration=999, postproduction={"duration": 888},
            shots=[{"id": "S1", "status": "planned", "duration": 10}])
        public = dashboard.public_state(synthetic_state(episode))["episodes"][0]
        self.assertEqual(public["duration"], 10)
        self.assertEqual(public["duration_source"], "planned")

    def test_missing_duration_remains_unknown(self):
        public = dashboard.public_state(synthetic_state(synthetic_episode(1)))["episodes"][0]
        self.assertIsNone(public["duration"])
        self.assertEqual(public["duration_source"], "unknown")

    def test_imported_script_slot_uses_source_duration_as_a_plan_without_shots(self):
        state = synthetic_state(synthetic_episode(1, "approved"),
            synthetic_episode(2, "awaiting_episode_plan", source_duration=130,
                source_file="source/episodes/ep02.md", source_text="PRIVATE-FULL-SCRIPT"))
        public = dashboard.public_state(state)
        episode = public["episodes"][1]
        self.assertEqual((episode["duration"], episode["duration_source"]), (130, "planned"))
        self.assertEqual(episode["source_duration"], 130)
        self.assertEqual(episode["shots"], [])
        self.assertEqual(episode["progress"]["stage"], "awaiting_episode_plan")
        self.assertIn("原稿", episode["progress"]["label"])
        self.assertTrue(episode["progress"]["waiting_for_human"])
        self.assertFalse(episode["progress"]["is_working"])
        self.assertNotIn("source_file", episode)
        self.assertNotIn("PRIVATE-FULL-SCRIPT", json.dumps(public))

    def test_imported_script_duration_does_not_replace_existing_shot_plan(self):
        episode = synthetic_episode(1, source_duration=130,
            shots=[{"id": "S1", "status": "planned", "duration": 10}])
        public = dashboard.public_state(synthetic_state(episode))["episodes"][0]
        self.assertEqual((public["duration"], public["duration_source"]), (10, "planned"))
        self.assertEqual(public["source_duration"], 130)

    def test_planning_candidate_waits_for_human_episode_plan_approval(self):
        episode = synthetic_episode(1, "awaiting_episode_plan",
            planning={"status": "candidate", "revision": 1, "signature": "a" * 64,
                      "summary": "按照原稿时长策划，保留原台词。", "episode": {"id": "E1", "shots": []},
                      "new_assets": []})
        public = dashboard.public_state(synthetic_state(episode))["episodes"][0]
        self.assertEqual(public["planning"]["status"], "candidate")
        self.assertEqual(public["progress"]["stage"], "awaiting_episode_plan_approval")
        self.assertIn("审批", public["progress"]["label"])
        self.assertTrue(public["progress"]["waiting_for_human"])
        self.assertFalse(public["progress"]["is_working"])

    def test_planning_preparation_is_working_and_rejection_returns_to_planning(self):
        episode = synthetic_episode(1, "awaiting_episode_plan", planning={"status": "preparing"})
        state = synthetic_state(episode, status="preparing_episode_plan")
        preparing = dashboard.public_state(state)["episodes"][0]["progress"]
        self.assertEqual(preparing["stage"], "preparing_episode_plan")
        self.assertTrue(preparing["is_working"])
        self.assertFalse(preparing["waiting_for_human"])
        episode["planning"]["status"] = "rejected"
        rejected = dashboard.public_state(state)["episodes"][0]["progress"]
        self.assertEqual(rejected["stage"], "awaiting_episode_plan")
        self.assertIn("修改意见", rejected["label"])
        self.assertFalse(rejected["is_working"])

    def test_planning_failure_or_uncertain_result_does_not_look_like_generation(self):
        for status in ("failed", "uncertain"):
            with self.subTest(status=status):
                episode = synthetic_episode(1, "episode_plan_failed", planning={"status": status,
                    "error": "规划停止 token=PRIVATE-KEY https://private.invalid/plan"})
                public = dashboard.public_state(synthetic_state(episode, status="blocked"))["episodes"][0]
                self.assertEqual(public["progress"]["stage"], "episode_plan_failed")
                self.assertFalse(public["progress"]["is_working"])
                self.assertTrue(public["progress"]["waiting_for_human"])
                self.assertIn("规划停止", public["planning"]["error"])
                self.assertNotIn("PRIVATE-", public["planning"]["error"])
                self.assertNotIn("https://private.invalid", public["planning"]["error"])

    def test_planning_projection_never_overrides_prior_episode_or_pause_gate(self):
        candidate = synthetic_episode(2, "awaiting_episode_plan", planning={"status": "candidate"})
        state = synthetic_state(synthetic_episode(1), candidate)
        future = dashboard.public_state(state)["episodes"][1]
        self.assertEqual(future["progress"]["stage"], "awaiting_previous_episode")
        state["episodes"][0]["status"] = "approved"
        state["paused"] = True
        paused = dashboard.public_state(state)["episodes"][1]
        self.assertEqual(paused["progress"]["stage"], "paused")
        self.assertFalse(paused["progress"]["is_working"])

    def test_planning_proposal_exposes_only_review_fields_without_context_or_paths(self):
        episode = synthetic_episode(1, "awaiting_episode_plan", planning={
            "status": "candidate", "revision": 2, "signature": "a" * 64, "summary": "本集新分镜方案",
            "source_sha256": "b" * 64, "source_duration": 130,
            "baseline": {"full_script": "PRIVATE-BASELINE"}, "context": "PRIVATE-CONTEXT",
            "source_file": "PRIVATE-PATH", "provider_result": "PRIVATE-PROVIDER",
            "episode": {"id": "E1", "title": "合成集", "summary": "沿用原稿", "music_prompt": "温暖",
                "source_file": "PRIVATE-CANDIDATE-PATH", "full_script": "PRIVATE-FULL-SCRIPT",
                "shots": [{"id": "S1", "duration": 10, "prompt": "人物走进车间", "dialogue": "你好。",
                    "asset_ids": ["char-main"], "scene_id": "workshop", "continuity_notes": "服装一致",
                    "provider_request": "PRIVATE-SHOT", "subshots": [
                        {"start": 0, "end": 10, "prompt": "固定全景", "raw": "PRIVATE-SUBSHOT"}]}]},
            "new_assets": [{"id": "char-main", "kind": "character", "name": "主角", "character_id": "",
                "voice_only": False, "prompt": "成年中国男子", "api_key": "PRIVATE-ASSET-KEY"}]})
        state = synthetic_state(episode)
        original = copy.deepcopy(state)
        planning = dashboard.public_state(state)["episodes"][0]["planning"]
        self.assertEqual(planning["signature"], "a" * 64)
        self.assertEqual(planning["source_duration"], 130)
        self.assertEqual(planning["episode"]["music_prompt"], "温暖")
        self.assertEqual(planning["episode"]["shots"][0]["dialogue"], "你好。")
        self.assertEqual(planning["episode"]["shots"][0]["subshots"],
                         [{"start": 0, "end": 10, "prompt": "固定全景"}])
        self.assertEqual(planning["new_assets"][0]["prompt"], "成年中国男子")
        self.assertNotIn("PRIVATE-", json.dumps(planning))
        self.assertEqual(state, original)

    def test_preparing_plan_with_hash_bound_completed_cache_can_be_recovered_locally(self):
        result = {"episode": {"summary": "PRIVATE-CACHED-RESULT"}, "new_assets": []}
        episode = synthetic_episode(1, "awaiting_episode_plan", planning={"status": "preparing", "signature": "a" * 64})
        state = synthetic_state(episode, jobs={"PRIVATE-JOB-ID": {
            "state": "completed", "item_id": "plan_episode", "post_cache_key": "a" * 64,
            "post_result": result, "post_result_hash": core._json_digest(result)}})
        before = copy.deepcopy(state)
        public = dashboard.public_state(state)
        projected = public["episodes"][0]
        self.assertTrue(projected["planning"]["recoverable"])
        self.assertIn("恢复审批方案", projected["progress"]["label"])
        self.assertFalse(projected["progress"]["is_working"])
        self.assertTrue(projected["progress"]["waiting_for_human"])
        self.assertNotIn("PRIVATE-", json.dumps(public))
        self.assertEqual(state, before)

    def test_incomplete_or_mismatched_plan_cache_never_advertises_recovery(self):
        result = {"episode": {}, "new_assets": []}
        original = {"state": "completed", "item_id": "plan_episode", "post_cache_key": "a" * 64,
                    "post_result": result, "post_result_hash": core._json_digest(result)}
        for changes in ({"state": "uncertain"}, {"state": "submitted"}, {"item_id": "review"},
                        {"post_cache_key": "b" * 64}, {"post_result_hash": "c" * 64},
                        {"post_result": {"changed": True}}, {"post_result": float("nan")}):
            with self.subTest(changes=changes):
                job = {**original, **changes}
                episode = synthetic_episode(1, "awaiting_episode_plan",
                    planning={"status": "preparing", "signature": "a" * 64})
                public = dashboard.public_state(synthetic_state(episode, jobs={"cache": job}))
                self.assertFalse(public["episodes"][0]["planning"].get("recoverable", False))
        for missing in ("post_result", "post_result_hash"):
            with self.subTest(missing=missing):
                job = copy.deepcopy(original)
                job.pop(missing)
                episode = synthetic_episode(1, "awaiting_episode_plan",
                    planning={"status": "preparing", "signature": "a" * 64})
                public = dashboard.public_state(synthetic_state(episode, jobs={"cache": job}))
                self.assertFalse(public["episodes"][0]["planning"].get("recoverable", False))

    def test_only_preparing_plan_can_advertise_recovery_and_latest_cache_must_match(self):
        result = {"episode": {}, "new_assets": []}
        job = {"state": "completed", "item_id": "plan_episode", "post_cache_key": "a" * 64,
               "post_result": result, "post_result_hash": core._json_digest(result)}
        for status in ("candidate", "uncertain", "failed", "approved", "rejected"):
            with self.subTest(status=status):
                episode = synthetic_episode(1, "awaiting_episode_plan", planning={"status": status, "signature": "a" * 64})
                public = dashboard.public_state(synthetic_state(episode, jobs={"cache": job}))
                self.assertFalse(public["episodes"][0]["planning"].get("recoverable", False))
        episode = synthetic_episode(1, "awaiting_episode_plan", planning={"status": "preparing", "signature": "a" * 64})
        public = dashboard.public_state(synthetic_state(episode, jobs={
            "older-valid": job, "newer-corrupt": {**job, "post_result_hash": "f" * 64}}))
        self.assertFalse(public["episodes"][0]["planning"].get("recoverable", False))

    def test_navigation_is_public_only_without_mutation_io_or_private_context(self):
        state = synthetic_state(synthetic_episode(1, "approved"), synthetic_episode(2),
            settings={"api_key": "PRIVATE-KEY"}, novel="PRIVATE-NOVEL")
        original = copy.deepcopy(state)
        with patch("norlan.media.probe", side_effect=AssertionError("projection must not inspect files")), \
                patch.object(dashboard, "_run", side_effect=AssertionError("projection must not run production")):
            public = dashboard.public_state(state)
        self.assertEqual(state, original)
        self.assertNotIn("episode_navigation", state)
        self.assertTrue(all("navigation" not in episode for episode in state["episodes"]))
        self.assertNotIn("PRIVATE-", json.dumps(public))


if __name__ == "__main__":
    unittest.main()
