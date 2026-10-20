"""Synthetic postproduction state and context tests; no media/provider calls."""
import copy
import json
import unittest
from unittest.mock import patch

from norlan import core, episode_post, pipeline


class MusicPostRecoveryTests(unittest.TestCase):
    def test_continue_does_not_repeat_an_exhausted_mix(self):
        ep = {"id": "synthetic-episode", "status": "awaiting_review",
              "postproduction": {"final_ready": True,
                                 "music_mix_check": {"status": "needs_attention", "repair_attempts": 2}}}
        state = {"settings": {"postproduction": {"enabled": True}}, "episodes": [ep]}
        with patch.object(pipeline, "_gate", return_value=state), \
                patch.object(episode_post, "check_final", side_effect=core.GateError("synthetic exhausted check")), \
                patch.object(pipeline, "_set") as update, \
                patch.object(pipeline, "_save_episode") as save, \
                patch.object(pipeline, "_shot") as shot, \
                patch.object(episode_post, "finish") as finish:
            for _ in range(3):
                pipeline._episode("synthetic-no-project-access", ep["id"])
            self.assertEqual(update.call_count, 3)
            save.assert_not_called()
            shot.assert_not_called()
            finish.assert_not_called()

    def test_bounded_context_retains_full_feedback_and_all_preferred_tracks(self):
        context = {"catalog": [{"id": f"synthetic-{i}", "prompt": "合成候选描述" * 40,
                    "audio_analysis": {"regions": [{"start": 20 * j, "end": 20 * j + 15,
                        "energy": "medium", "mean_dbfs": -18} for j in range(4)]}} for i in range(12)],
                   "timeline": [{"id": f"s{i}", "action": "合成剧情动作" * 40} for i in range(32)],
                   "feedback": "用户完整修改意见" * 400}
        original = copy.deepcopy(context)
        required = ["synthetic-0", "synthetic-1"]
        bounded = episode_post._bound_music_context(context, required)
        self.assertLessEqual(len(json.dumps(bounded, ensure_ascii=False)), 20000)
        self.assertEqual(bounded["feedback"], context["feedback"])
        self.assertTrue(set(required) <= {t["id"] for t in bounded["catalog"]})
        self.assertEqual(context, original)
