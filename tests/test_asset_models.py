"""Synthetic asset-model/price fixtures; never call a provider or read keys."""

import copy
import unittest

from norlan import asset_models, video_models
from norlan.billing import IMAGE_MODEL, price_snapshot, usage_record


PRO = "doubao-seedream-5-0-pro-260628"
OLD = "doubao-seedream-4-5-251128"
FAST = video_models.DEFAULT_MODEL
MINI = "doubao-seedance-2-0-mini-260615"


class AssetModelsTests(unittest.TestCase):
    def state(self, **ark):
        return {"settings": {"ark": {"image_model": IMAGE_MODEL, "video_model": FAST, **ark},
                             "caps": {"image": 0.15, "video_second": 0.1}},
                "budget": {"currency": "CNY"}, "plan": {"ratio": "16:9"},
                "assets": [{"id": "synthetic-character", "kind": "character", "status": "approved",
                            "file": "synthetic-character.png"}]}

    def image(self, refs=0):
        return {"id": "synthetic-image", "kind": "character", "creation_input": {
            "references": [{"file": f"synthetic-{index}.png"} for index in range(refs)]}}

    def voice(self):
        return {"id": "synthetic-voice", "kind": "voice", "character_id": "synthetic-character",
                "audition_text": "这个句子只用于自动化合成测试场景。",
                "audition_source": "这个句子只用于自动化合成测试场景。", "audition_reason": "合成测试。"}

    def test_selected_model_is_detached_and_never_changes_project_defaults(self):
        state = self.state(nested={"synthetic": [1]})
        asset = {**self.image(), "generation_model": PRO}
        before = copy.deepcopy((state, asset))
        config = asset_models.effective_config(state, asset)
        self.assertEqual(config["image_model"], PRO)
        self.assertEqual(config["video_model"], FAST)
        config["nested"]["synthetic"].append(2)
        self.assertEqual((state, asset), before)
        config = asset_models.effective_config(state, {**self.voice(), "generation_model": MINI})
        self.assertEqual(config["video_model"], MINI)
        self.assertTrue(config["generate_audio"])
        self.assertFalse(config["video_watermark"])
        self.assertEqual(config["resolution"], "720p")

    def test_image_catalog_and_existing_endpoint_are_explicit(self):
        catalog = asset_models.catalog(self.state(), self.image())
        self.assertEqual(catalog["default_model"], IMAGE_MODEL)
        self.assertEqual(len(catalog["models"]), 6)
        self.assertIn(PRO, [item["id"] for item in catalog["models"]])
        self.assertTrue(next(item for item in catalog["models"] if item["id"] == OLD)["reason"])
        state = self.state(image_model="ep-synthetic-existing")
        self.assertEqual(asset_models.catalog(state, self.image())["models"][0]["id"], "ep-synthetic-existing")
        self.assertEqual(asset_models.validate_selection(state, self.image(), {}, "ep-synthetic-existing"), "ep-synthetic-existing")
        for model in ("ep-other", FAST, "api?key=SECRET", True, None):
            with self.subTest(model=model), self.assertRaises(ValueError):
                asset_models.validate_selection(state, self.image(), {}, model)
        self.assertNotIn("SECRET", str(asset_models.catalog(self.state(image_model="https://bad?key=SECRET"), self.image())))

    def test_images_validate_exact_inputs_and_model_specific_size(self):
        state, image = self.state(image_size="4K"), self.image(9)
        before = copy.deepcopy((state, image))
        choices = asset_models.options(state, image)
        self.assertFalse(next(item for item in choices if item["id"] == PRO)["available"])
        self.assertTrue(next(item for item in choices if item["id"] == OLD)["available"])
        self.assertEqual((state, image), before)
        with self.assertRaisesRegex(ValueError, "不会静默丢弃"):
            asset_models.validate_choice(self.state(), self.image(10), IMAGE_MODEL)
        candidate = {"reference_manifest": {"images": [], "audio": ["synthetic-audio"]}}
        with self.assertRaisesRegex(ValueError, "声音参考"):
            asset_models.validate_choice(self.state(), image, IMAGE_MODEL, candidate=candidate)
        self.assertEqual(asset_models.validate_choice(self.state(image_size="1600x900"), image, PRO)["image_size"], "1600x900")
        with self.assertRaises(ValueError):
            asset_models.validate_choice(self.state(image_size="1600x900"), image, OLD)

    def test_voice_reuses_catalog_and_actual_reference_manifest(self):
        state, voice = self.state(), self.voice()
        self.assertEqual([item["id"] for item in asset_models.catalog(state, voice)["models"]],
                         [item["id"] for item in video_models.catalog(state)["models"]])
        candidate = {"duration": 20, "reference_manifest": {"images": ["synthetic-image"], "audio": []}}
        available = [item["id"] for item in asset_models.options(state, voice, candidate) if item["available"]]
        self.assertEqual(available, ["doubao-seedance-2-5-260628"])
        with self.assertRaisesRegex(ValueError, "原生声音"):
            asset_models.validate_choice(self.state(generate_audio=False), voice, FAST)
        with self.assertRaisesRegex(ValueError, "原生声音"):
            asset_models.validate_choice(self.state(generate_audio="false"), voice, FAST)
        candidate["reference_manifest"]["images"] *= 10
        self.assertTrue(all(not item["available"] for item in asset_models.options(state, voice, candidate)))

    def test_voice_only_source_has_no_invented_visual_reference(self):
        state, voice = self.state(), self.voice()
        state["assets"][0] = {"id": "synthetic-character", "kind": "character", "voice_only": True}
        self.assertEqual(asset_models.validate_choice(state, voice, FAST)["video_model"], FAST)
        state["assets"] = []
        self.assertTrue(all(not item["available"] for item in asset_models.options(state, voice)))
        candidate = {"duration": 5, "reference_manifest": {"images": [], "audio": []}}
        self.assertEqual(asset_models.validate_selection(state, voice, candidate, FAST), FAST)

    def test_unknown_image_model_requires_positive_operator_cap(self):
        state = self.state(image_model="ep-synthetic-existing")
        for cap in (None, False, 0, -1, float("nan")):
            state["settings"]["caps"]["image"] = cap
            with self.subTest(cap=cap), self.assertRaisesRegex(ValueError, "预算占用上限"):
                asset_models.validate_choice(state, self.image(), "ep-synthetic-existing")

    def test_reservation_uses_selected_model_and_actual_reference_inputs(self):
        state = self.state()
        self.assertEqual(asset_models.reserve_amount(state, self.image(), IMAGE_MODEL), 0.15)
        self.assertEqual(asset_models.reserve_amount(state, self.image(4), PRO), 0.66)
        self.assertEqual(asset_models.reserve_amount(self.state(image_size="1.5K"), self.image(1), PRO), 0.30)
        self.assertEqual(asset_models.reserve_amount(state, self.image(), OLD), 0.25)
        fast = asset_models.reserve_amount(state, self.voice(), FAST, duration=5)
        standard = asset_models.reserve_amount(state, self.voice(), "doubao-seedance-2-0-260128", duration=5)
        self.assertGreater(standard, fast)
        state["settings"]["caps"]["image"] = 1
        self.assertEqual(asset_models.reserve_amount(state, self.image(4), PRO), 1)


class AssetImageBillingTests(unittest.TestCase):
    def test_legacy_image_prices_are_exact_and_keep_alias_identity(self):
        cases = (("doubao-seedream-5-0-260128", .22), ("doubao-seedream-5-0-lite-260128", .22),
                 (OLD, .25), ("doubao-seedream-4-0-250828", .20), (IMAGE_MODEL, .12))
        for model, rate in cases:
            with self.subTest(model=model):
                snapshot = price_snapshot({"image_model": model}, "image")
                self.assertEqual(snapshot["model"], model)
                self.assertEqual(snapshot["rate"], rate)
                result = usage_record({"model": model, "provider_status": "succeeded",
                                       "usage": {"generated_images": 1}}, snapshot)
                self.assertEqual(result["usage_cost"], rate)
                self.assertEqual(result["status"], "usage_priced")

    def test_pro_single_image_rates_include_required_input_evidence(self):
        for size, rate in (("1K", .30), ("1.5K", .30), ("2K", .60),
                           ("1600x1600", .30), ("2048x2048", .60)):
            with self.subTest(size=size):
                snapshot = price_snapshot({"image_model": PRO, "image_size": size}, "image")
                self.assertEqual(snapshot["rate"], rate)
                result = usage_record({"model": PRO, "provider_status": "succeeded",
                                       "usage": {"generated_images": 1, "input_images": 4}}, snapshot)
                self.assertEqual(result["usage_cost"], round(rate + .06, 6))
                missing = usage_record({"model": PRO, "provider_status": "succeeded",
                                        "usage": {"generated_images": 1}}, snapshot)
                self.assertIsNone(missing["usage_cost"])
                self.assertEqual(missing["status"], "usage_missing")

    def test_pro_unknown_price_tier_and_alias_never_use_flash_price(self):
        for config in ({"image_model": PRO, "image_size": "4K"},
                       {"image_model": PRO, "image_size": "unknown"},
                       {"image_model": "ep-synthetic", "image_size": "2K"},
                       {"image_model": PRO, "currency": "USD"}):
            with self.subTest(config=config):
                self.assertNotIn("rate", price_snapshot(config, "image"))
        snapshot = price_snapshot({"image_model": PRO}, "image")
        wrong = usage_record({"model": IMAGE_MODEL, "provider_status": "succeeded",
                              "usage": {"generated_images": 1, "input_images": 0}}, snapshot)
        self.assertIsNone(wrong["usage_cost"])


if __name__ == "__main__":
    unittest.main()
