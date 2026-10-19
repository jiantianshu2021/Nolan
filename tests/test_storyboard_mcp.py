"""Synthetic projection checks; no provider or production calls."""
import copy
import json
import unittest

from norlan.mcp_server import _compact_storyboard_status


class StoryboardMCPTests(unittest.TestCase):
    def test_storyboard_status_keeps_decision_counts_without_recursive_context(self):
        value = {"summary": "合成：原单镜改成反应与道具两个镜头。", "before_count": 1,
                 "before_duration": 5, "duration": 10,
                 "subshots": [{"start": 0, "end": 2, "prompt": "镜头一"},
                              {"start": 2, "end": 5, "prompt": "镜头二"}],
                 "before": {"history": "PRIVATE_ARCHIVE" * 10000},
                 "reference_manifest": {"context": "PRIVATE_CONTEXT" * 10000}}
        original = copy.deepcopy(value)
        result = _compact_storyboard_status(value)
        self.assertEqual(result, {"summary": value["summary"], "before_count": 1, "proposed_count": 2,
                                  "before_duration": 5, "duration": 10})
        self.assertNotIn("PRIVATE", json.dumps(result))
        self.assertNotIn("subshots", result)
        self.assertEqual(value, original)

    def test_repeated_local_history_does_not_enlarge_model_status(self):
        value = {"summary": "合成修改", "before_count": 1, "subshots": [{}, {}]}
        sizes = []
        for size in (10, 1000, 100000):
            value["history"] = ["ARCHIVE" * size]
            sizes.append(len(json.dumps(_compact_storyboard_status(value))))
        self.assertEqual(len(set(sizes)), 1)
        self.assertIsNone(_compact_storyboard_status(None))


if __name__ == "__main__":
    unittest.main()
