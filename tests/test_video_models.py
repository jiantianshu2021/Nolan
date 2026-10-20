"""Synthetic catalog and billing fixtures; no network or generation calls."""

import copy
import unittest

from norlan import video_models
from norlan.billing import estimate_reservation, price_snapshot, summarize_costs, usage_record


FAST = "doubao-seedance-2-0-fast-260128"
STANDARD = "doubao-seedance-2-0-260128"
MINI = "doubao-seedance-2-0-mini-260615"
NEW = "doubao-seedance-2-5-260628"


class VideoModelCatalogTests(unittest.TestCase):
    def state(self, **config):
        return {"settings": {"ark": {"video_model": FAST, **config}}, "plan": {"ratio": "16:9"}}

    def test_effective_config_does_not_change_defaults_or_source(self):
        state = self.state(image_model="synthetic-image", video_watermark=False,
                           nested={"only": "copy"}, ratio="1:1")
        before = copy.deepcopy(state)
        config = video_models.effective_config(state, {"video_model": MINI})
        self.assertEqual(config["video_model"], MINI)
        self.assertEqual(config["ratio"], "16:9")
        self.assertEqual(config["image_model"], "synthetic-image")
        config["nested"]["only"] = "changed"
        self.assertEqual(state, before)
        self.assertEqual(video_models.effective_config(state, {"video_model": ""})["video_model"], FAST)
        self.assertNotIn("video_model", video_models.effective_config({"settings": {"ark": {}}}))

    def test_catalog_has_current_official_choices_and_disabled_legacy(self):
        result = video_models.catalog(self.state())
        self.assertEqual(result["default_model"], FAST)
        self.assertEqual(len(result["models"]), 6)
        self.assertEqual({m["id"] for m in result["models"] if m["available"]}, {FAST, STANDARD, MINI, NEW})
        self.assertTrue(all(m["reason"] for m in result["models"] if not m["available"]))
        self.assertNotIn("doubao-seedance-1-5-pro-251215", str(result))
        self.assertNotIn("doubao-seedance-1-5-pro-251215", str(video_models.catalog(self.state(
            video_model="doubao-seedance-1-5-pro-251215"))["models"]))
        self.assertNotIn("SECRET", str(video_models.catalog(self.state(video_model="https://bad?key=SECRET"))))

    def test_requests_keep_current_references_and_never_shorten_duration(self):
        for model in (FAST, STANDARD, MINI, NEW):
            video_models.validate_request({"video_model": model}, 15, 9, 3, True)
        video_models.validate_request({"video_model": NEW}, 30, 9, 3, True)
        for model in (FAST, STANDARD, MINI):
            with self.subTest(model=model), self.assertRaisesRegex(ValueError, "不会自动修改"):
                video_models.validate_request({"video_model": model}, 16, 1, 1, False)
        for duration in (True, 3, 31, 4.5, float("inf"), "10", None):
            with self.subTest(duration=duration), self.assertRaises(ValueError):
                video_models.validate_request({"video_model": NEW}, duration, 1, 0, False)
        for counts in ((10, 0, False), (1, 4, False), (True, 0, False), (0, 1, False), (1, 1, 1)):
            with self.subTest(counts=counts), self.assertRaises(ValueError):
                video_models.validate_request({"video_model": NEW}, 10, *counts)

    def test_resolution_is_model_specific(self):
        video_models.validate_request({"video_model": STANDARD, "resolution": "4k"}, 10, 1, 1, False)
        video_models.validate_request({"video_model": NEW, "resolution": "1080p"}, 10, 1, 1, False)
        for model in (FAST, MINI):
            with self.subTest(model=model), self.assertRaisesRegex(ValueError, "1080p"):
                video_models.validate_request({"video_model": model, "resolution": "1080p"}, 10, 1, 1, False)

    def test_selection_only_accepts_verified_models_or_existing_endpoint(self):
        state, shot = self.state(), {"duration": 10}
        manifest = {"images": ["synthetic-image"], "audio": ["synthetic-voice"]}
        before = copy.deepcopy((state, shot, manifest))
        self.assertEqual(video_models.validate_selection(state, shot, manifest, STANDARD), STANDARD)
        self.assertEqual((state, shot, manifest), before)
        for model in ("new-unknown-model", "doubao-seedance-1-0-pro-250528", None, True, "api?key=SECRET"):
            with self.subTest(model=model), self.assertRaises(ValueError):
                video_models.validate_selection(state, shot, manifest, model)
        configured = self.state(video_model="ep-synthetic-existing")
        self.assertEqual(video_models.validate_selection(configured, shot, manifest, "ep-synthetic-existing"),
                         "ep-synthetic-existing")
        self.assertEqual(video_models.catalog(configured)["models"][0]["id"], "ep-synthetic-existing")
        with self.assertRaises(ValueError):
            video_models.validate_selection(configured, shot, manifest, "ep-other")

    def test_options_check_actual_candidate_reference_manifest(self):
        candidate = {"reference_manifest": {"images": [], "audio": ["synthetic-voice"], "context": {}}}
        options = video_models.options(self.state(), {"duration": 10}, candidate)
        self.assertTrue(all(not m["available"] for m in options))
        candidate["reference_manifest"]["context"]["previous_actual"] = {"reference_mode": "original_video"}
        options = video_models.options(self.state(), {"duration": 20}, candidate)
        self.assertEqual([m["id"] for m in options if m["available"]], [NEW])
        candidate["reference_manifest"]["images"] = ["synthetic-image"] * 10
        self.assertTrue(all(not m["available"] for m in video_models.options(self.state(), {"duration": 10}, candidate)))

    def test_capabilities_are_fresh_and_keep_conservative_reference_limits(self):
        caps = video_models.capabilities({"video_model": NEW})
        self.assertEqual(caps["max_duration"], 30)
        self.assertEqual(caps["max_video_images"], 9)
        self.assertEqual(caps["max_audio_references"], 3)
        caps["max_duration"] = 1
        self.assertEqual(video_models.capabilities({"video_model": NEW})["max_duration"], 30)
        self.assertEqual(video_models.capabilities({"video_model": FAST})["max_duration"], 15)
        video_models.validate_request({"video_model": "synthetic-existing-endpoint"}, 10, 0, 0, False)


class VideoModelBillingTests(unittest.TestCase):
    def test_no_video_budget_estimate_reserves_current_model_cost_not_fast_cost(self):
        self.assertEqual(estimate_reservation({"video_model": FAST}, 10), 8.0)
        self.assertEqual(estimate_reservation({"video_model": NEW}, 10), 15.12)
        self.assertEqual(estimate_reservation({"video_model": NEW}, 30), 45.36)
        self.assertEqual(estimate_reservation({"video_model": STANDARD, "resolution": "4k"}, 5), 25.28)
        self.assertEqual(estimate_reservation({"video_model": NEW, "resolution": "480p"}, 5), 3.37)
        for config, duration, reference in (({"video_model": NEW}, 10, 2),
                ({"video_model": NEW, "ratio": "adaptive"}, 10, 0),
                ({"video_model": "ep-synthetic-existing"}, 10, 0),
                ({"video_model": FAST, "resolution": "1080p"}, 10, 0),
                ({"video_model": NEW, "currency": "USD"}, 10, 0),
                ({"video_model": NEW}, True, 0), ({"video_model": NEW}, 31, 0)):
            self.assertIsNone(estimate_reservation(config, duration, reference))

    def test_release_id_rates_include_input_video_and_resolution(self):
        rows = ((FAST, "720p", 37, 22), (STANDARD, "480p", 46, 28),
                (STANDARD, "1080p", 51, 31), (STANDARD, "4k", 26, 16),
                (MINI, "720p", 23, 14), (NEW, "720p", 70, 42), (NEW, "1080p", 77, 46))
        for model, resolution, plain, referenced in rows:
            for has_video, rate in ((False, plain), (True, referenced)):
                with self.subTest(model=model, resolution=resolution, has_video=has_video):
                    snapshot = price_snapshot({"video_model": model, "resolution": resolution}, "video", has_video_input=has_video)
                    self.assertEqual(snapshot["rate"], rate)
                    self.assertEqual(usage_record({"model": model, "resolution": resolution,
                        "usage": {"completion_tokens": 1_000_000}, "provider_status": "succeeded"}, snapshot)["usage_cost"], rate)

    def test_unsupported_combinations_stay_unknown_and_do_not_use_fast_rate(self):
        for config, has_video in (({"video_model": MINI, "resolution": "1080p"}, False),
                                  ({"video_model": NEW, "resolution": "4k"}, False),
                                  ({"video_model": "doubao-seedance-1-0-pro-250528"}, True),
                                  ({"video_model": "ep-synthetic-existing"}, False)):
            self.assertNotIn("rate", price_snapshot(config, "video", has_video_input=has_video))
        snapshot = price_snapshot({"video_model": NEW}, "video")
        for extra in ({"model": FAST}, {"resolution": "480p"}, {"service_tier": "flex"}):
            result = usage_record({"model": NEW, "usage": {"completion_tokens": 1000},
                                   "provider_status": "succeeded", **extra}, snapshot)
            self.assertEqual(result["status"], "pricing_missing")

    def test_locked_old_snapshot_and_zero_usage_are_preserved(self):
        snapshot = price_snapshot({"video_model": NEW}, "video")
        snapshot["rate"] = 35  # Synthetic historical locked promotion, not today's rate.
        self.assertEqual(usage_record({"model": NEW, "usage": {"completion_tokens": 1_000_000},
                                      "provider_status": "succeeded"}, snapshot)["usage_cost"], 35)
        zero = usage_record({"model": NEW, "usage": {"completion_tokens": 0},
                             "provider_status": "failed"}, snapshot)
        jobs = {"synthetic": {"amount": 10, "accounting": "spent", "billing": zero}}
        self.assertEqual(summarize_costs(jobs, "CNY")["usage_cost"], 0)
        self.assertEqual(snapshot["rate"], 35)


if __name__ == "__main__":
    unittest.main()
