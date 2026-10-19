"""Synthetic accounting fixtures only; no provider calls or paid generations."""

import copy
import unittest

from norlan.billing import IMAGE_MODEL, VIDEO_MODEL, price_snapshot, summarize_costs, usage_record


class BillingTests(unittest.TestCase):
    def video_config(self, **extra):
        return {"video_model": VIDEO_MODEL, **extra}

    def image_snapshot(self):
        return price_snapshot({"image_model": IMAGE_MODEL}, "image")

    def video_record(self, usage=None, **extra):
        return usage_record({"model": VIDEO_MODEL, "provider_status": "succeeded", "usage": usage or {}, **extra},
                            price_snapshot(self.video_config(), "video"))

    def test_exact_models_and_input_video_price(self):
        self.assertEqual(self.image_snapshot()["rate"], 0.12)
        for kind in ("voice", "video"):
            for resolution in ("480p", "720p"):
                config = self.video_config(resolution=resolution)
                self.assertEqual(price_snapshot(config, kind)["rate"], 37)
                self.assertEqual(price_snapshot(config, kind, has_video_input=True)["rate"], 22)
        self.assertEqual(self.image_snapshot()["price_checked_at"], "2026-10-20")

    def test_unknown_configuration_does_not_guess_a_price(self):
        cases = [{"video_model": "ep-user-endpoint"}, {"video_model": "doubao-seedance-2-0-fast"},
                 {"base_url": "https://gateway.example/api/v3"}, {"currency": "USD"},
                 {"currency": None}, {"currency": "cny"}, {"service_tier": "flex"},
                 {"service_tier": "online"}, {"service_tier": None}, {"resolution": "1080p"},
                 {"provider": "other"}, {"base_url": "https://ark.cn-beijing.volces.com/api/v3?key=secret"}]
        for values in cases:
            with self.subTest(values=values):
                self.assertNotIn("rate", price_snapshot(self.video_config(**values), "video"))
        self.assertNotIn("rate", price_snapshot(self.video_config(), "music"))
        self.assertNotIn("rate", price_snapshot(self.video_config(), "video", has_video_input="false"))

    def test_snapshot_keeps_settings_without_raw_endpoint_or_credentials(self):
        result = price_snapshot(self.video_config(base_url="https://gateway.example?key=SECRET", currency="USD"), "voice")
        self.assertEqual(result["model"], VIDEO_MODEL)
        self.assertEqual(result["currency"], "USD")
        self.assertEqual(result["resolution"], "720p")
        self.assertNotIn("SECRET", str(result))

    def test_usage_pricing_uses_actual_quantity_and_six_decimal_rounding(self):
        result = self.video_record({"completion_tokens": 108108, "total_tokens": 999999})
        self.assertEqual(result["usage_cost"], 4.0 - 0.000004)
        self.assertEqual(result["status"], "usage_priced")
        image = usage_record({"model": IMAGE_MODEL, "provider_status": "succeeded", "usage": {"generated_images": 1}}, self.image_snapshot())
        self.assertEqual(image["usage_cost"], 0.12)
        self.assertNotIn("actual_cost", image)
        self.assertNotIn("bill_reference", image)

    def test_missing_usage_is_unknown_and_total_tokens_is_not_completion_tokens(self):
        for usage in ({}, {"total_tokens": 108108}, {"output_tokens": 108108}):
            result = self.video_record(usage)
            self.assertIsNone(result["usage_cost"])
            self.assertEqual(result["status"], "usage_missing")
        self.assertIsNone(usage_record({}, {})["usage_cost"])

    def test_usage_whitelist_rejects_bools_negative_nonfinite_fraction_and_strings(self):
        for value in (True, False, -1, float("nan"), float("inf"), 1.25, "100", None):
            with self.subTest(value=value):
                result = self.video_record({"completion_tokens": value, "prompt": "SECRET", "input_images": 2})
                self.assertEqual(result["usage"], {"input_images": 2})
                self.assertIsNone(result["usage_cost"])
                self.assertNotIn("SECRET", str(result))
        self.assertEqual(self.video_record({"completion_tokens": 1.0})["usage"], {"completion_tokens": 1})

    def test_model_mismatch_missing_model_or_tier_change_is_not_priced(self):
        snapshot = price_snapshot(self.video_config(), "video")
        for extra in ({"model": "different-model"}, {"model": None}, {"service_tier": "flex"},
                      {"service_tier": "online"}, {"service_tier": None}):
            result = usage_record({"model": VIDEO_MODEL, "provider_status": "succeeded", "usage": {"completion_tokens": 1000}, **extra}, snapshot)
            self.assertIsNone(result["usage_cost"])
            self.assertEqual(result["status"], "pricing_missing")
            self.assertEqual(result["usage"], {"completion_tokens": 1000})
        self.assertEqual(usage_record({"provider_status": "succeeded", "usage": {"completion_tokens": 1000}}, snapshot)["status"], "pricing_missing")

    def test_zero_generated_images_is_not_a_successful_free_image(self):
        for status in ("", "succeeded", "completed"):
            result = usage_record({"model": IMAGE_MODEL, "provider_status": status,
                                   "usage": {"generated_images": 0}}, self.image_snapshot())
            self.assertEqual(result["status"], "usage_missing")
            self.assertIsNone(result["usage_cost"])
        failed = usage_record({"model": IMAGE_MODEL, "provider_status": "failed",
                               "usage": {"generated_images": 0}}, self.image_snapshot())
        self.assertEqual(failed["usage_cost"], 0)

    def test_failed_video_positive_usage_is_preserved_but_not_priced(self):
        for status in ("failed", "cancelled", "expired"):
            with self.subTest(status=status):
                result = self.video_record({"completion_tokens": 1000}, provider_status=status)
                self.assertEqual(result["usage"], {"completion_tokens": 1000})
                self.assertEqual(result["status"], "usage_missing")
                self.assertIsNone(result["usage_cost"])
                missing = self.video_record({}, provider_status=status)
                self.assertIsNone(missing["usage_cost"])
                zero = self.video_record({"completion_tokens": 0}, provider_status=status)
                self.assertEqual(zero["usage_cost"], 0)
                self.assertEqual(zero["status"], "usage_priced")

    def test_video_resolution_mismatch_is_unknown_but_absent_uses_locked_request(self):
        for resolution in ("1080p", "480p", "unknown", None, [], {}):
            with self.subTest(resolution=resolution):
                result = self.video_record({"completion_tokens": 1000}, resolution=resolution)
                self.assertEqual(result["usage"], {"completion_tokens": 1000})
                self.assertEqual(result["status"], "pricing_missing")
                self.assertIsNone(result["usage_cost"])
        self.assertEqual(self.video_record({"completion_tokens": 1000}, resolution="720p")["usage_cost"], 0.037)
        self.assertEqual(self.video_record({"completion_tokens": 1000})["usage_cost"], 0.037)

    def test_response_metadata_cannot_carry_free_text_or_secret_urls(self):
        result = self.video_record({"completion_tokens": 1000}, request_id="https://example/?key=SECRET",
                                   provider_status="Bearer SECRET")
        self.assertEqual(result["request_id"], "")
        self.assertEqual(result["provider_status"], "")
        self.assertNotIn("SECRET", str(result))
        result = self.video_record({"completion_tokens": 1000}, request_id="request-123_abc", provider_status="succeeded")
        self.assertEqual(result["request_id"], "request-123_abc")
        self.assertEqual(result["provider_status"], "succeeded")

    def test_summary_partial_actual_usage_and_unknown_are_not_double_counted(self):
        jobs = {
            "actual": {"amount": 5, "accounting": "spent", "billing": {"actual_cost": 3.2, "bill_reference": "bill-1",
                "currency": "CNY", "usage_cost": 4, "status": "usage_priced"}},
            "usage": {"amount": 5, "accounting": "spent", "billing": {"currency": "CNY", "usage_cost": 3.999996, "status": "usage_priced"}},
            "unknown": {"amount": 5, "accounting": "unknown"},
            "reserved": {"amount": 5, "accounting": "reserved"},
            "released": {"amount": 5, "accounting": "released"},
            "demo": {"amount": 0, "accounting": "spent"},
        }
        before = copy.deepcopy(jobs)
        summary = summarize_costs(jobs, "CNY")
        self.assertEqual(summary, {"actual_cost": 3.2, "actual_cost_jobs": 1, "cost_pending_jobs": 2,
                                   "usage_cost": 3.999996, "usage_cost_jobs": 1, "cost_jobs": 3})
        self.assertEqual(summarize_costs(jobs, "CNY"), summary)
        self.assertEqual(jobs, before)

    def test_summary_verified_zero_has_precedence_over_usage_price(self):
        jobs = {"free": {"amount": 5, "accounting": "spent", "billing": {"currency": "CNY", "actual_cost": 0,
            "bill_reference": "bill-free-credit", "usage_cost": 4, "status": "usage_priced"}}}
        summary = summarize_costs(jobs, "CNY")
        self.assertEqual(summary["actual_cost"], 0)
        self.assertEqual(summary["actual_cost_jobs"], 1)
        self.assertEqual(summary["cost_pending_jobs"], 0)
        self.assertEqual(summary["usage_cost_jobs"], 0)
        self.assertIsNone(summary["usage_cost"])

    def test_summary_invalid_or_wrong_currency_is_unknown(self):
        invalid_billing = [
            {"currency": "USD", "actual_cost": 1, "bill_reference": "bill"},
            {"currency": "CNY", "actual_cost": 1},
            {"currency": "CNY", "actual_cost": 1, "bill_reference": "   "},
            {"currency": "CNY", "actual_cost": None, "bill_reference": "bill"},
            {"currency": "CNY", "actual_cost": True, "bill_reference": "bill"},
            {"currency": "CNY", "actual_cost": float("nan"), "bill_reference": "bill"},
            {"currency": "CNY", "usage_cost": -1, "status": "usage_priced"},
            {"currency": "CNY", "usage_cost": True, "status": "usage_priced"},
            {"currency": "CNY", "usage_cost": 0, "status": "usage_missing"},
            {"currency": "CNY", "usage_cost": 0, "status": "usage_priced"},
        ]
        jobs = {str(i): {"amount": 5, "accounting": "spent", "billing": billing} for i, billing in enumerate(invalid_billing)}
        summary = summarize_costs(jobs, "CNY")
        self.assertIsNone(summary["actual_cost"])
        self.assertIsNone(summary["usage_cost"])
        self.assertEqual(summary["cost_pending_jobs"], len(invalid_billing))
        self.assertEqual(summary["actual_cost_jobs"], 0)

    def test_summary_zero_usage_requires_valid_reported_usage(self):
        record = self.video_record({"completion_tokens": 0})
        jobs = {"zero": {"amount": 5, "accounting": "spent", "billing": record}}
        summary = summarize_costs(jobs, "CNY")
        self.assertEqual(summary["usage_cost"], 0)
        self.assertEqual(summary["usage_cost_jobs"], 1)
        self.assertIsNone(summary["actual_cost"])
        self.assertEqual(summary["cost_pending_jobs"], 1)

    def test_summary_unknown_currency_and_invalid_caps_remain_untrusted(self):
        jobs = {"unknown": {"amount": 5, "accounting": "unknown", "billing": {
            "currency": None, "actual_cost": 1, "bill_reference": "bill"}}}
        for key, value in enumerate((False, True, -1, float("nan"), float("inf"), "5")):
            jobs[str(key)] = {"amount": value, "accounting": "spent"}
        summary = summarize_costs(jobs, None)
        self.assertEqual(summary["cost_jobs"], 1)
        self.assertIsNone(summary["actual_cost"])
        self.assertEqual(summary["cost_pending_jobs"], 1)


if __name__ == "__main__":
    unittest.main()
