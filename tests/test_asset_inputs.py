"""Synthetic fixtures only: asset choices and local image validation, no API."""
import base64
import copy
from pathlib import Path
import struct
import tempfile
import unittest
from unittest.mock import patch
import zlib

from norlan import asset_inputs, core, pipeline


def png_data(color=b"\x11\x55\xaa"):
    def chunk(kind, data):
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))
    return (b"\x89PNG\r\n\x1a\n"
            + chunk(b"IHDR", struct.pack(">IIBBBBB", 2, 2, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress((b"\x00" + color * 2) * 2))
            + chunk(b"IEND", b""))


def data_url(data=None, mime="image/png"):
    return "data:" + mime + ";base64," + base64.b64encode(data if data is not None else png_data()).decode()


class AssetInputTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        core.create_project(self.root, "合成测试小说")
        core.install_plan(self.root, pipeline.demo_plan())
        core.approve(self.root, "plan", "plan", 1)

    def asset(self, asset_id="hero"):
        return core._find(core.load_project(self.root), "asset", asset_id)

    def test_new_plan_requires_human_choice_and_rejects_planner_injection(self):
        plan = pipeline.demo_plan()
        plan["assets"][0]["creation_input"] = {"mode": "gpt", "confirmed": True}
        state = core.install_plan(self.root, plan)
        self.assertEqual([a["id"] for a in asset_inputs.pending_assets(state)], ["hero", "station"])
        self.assertEqual(state["assets"][0]["creation_input"], asset_inputs.empty_input())

    def test_all_inputs_require_current_approved_plan(self):
        state = core.install_plan(self.root, pipeline.demo_plan())
        mutations = [lambda: core.set_asset_input(self.root, "hero", 0, 0, "gpt", ""),
                     lambda: core.add_asset_reference(self.root, "hero", 0, 0, data_url()),
                     lambda: core.remove_asset_reference(self.root, "hero", 0, 0, "missing"),
                     lambda: core.set_remaining_asset_inputs_gpt(self.root, state["plan"]["version"])]
        for mutate in mutations:
            with self.assertRaisesRegex(core.GateError, "先批准"):
                mutate()
            self.assertEqual(self.asset()["creation_input"], asset_inputs.empty_input())
        self.assertEqual(list(self.root.glob("inputs/assets/hero/*")), [])

    def test_bulk_gpt_preserves_custom_descriptions_references_and_existing_assets(self):
        with core.transaction(self.root) as state:
            template = state["assets"][0]
            variants = [
                ("draft_mode", {"creation_input": {**asset_inputs.empty_input(), "mode": "custom"}}),
                ("draft_description", {"creation_input": {**asset_inputs.empty_input(), "description": "白衬衫"}}),
                ("draft_picture", {"creation_input": {**asset_inputs.empty_input(), "references": [{"id": "draft"}]}}),
                ("started", {"version": 1}),
                ("empty_mode", {"creation_input": {**asset_inputs.empty_input(), "mode": ""}}),
            ]
            for asset_id, changes in variants:
                state["assets"].append({**copy.deepcopy(template), "id": asset_id, **changes})
            before = copy.deepcopy(state["assets"])
        state = core.set_remaining_asset_inputs_gpt(self.root, 1)
        selected = {"hero", "station", "empty_mode"}
        for previous, current in zip(before, state["assets"]):
            if current["id"] in selected:
                self.assertTrue(asset_inputs.confirmed(current))
                self.assertEqual(current["creation_input"]["revision"], 1)
                self.assertEqual(current["status"], "planned")
                self.assertEqual(current["version"], 0)
            else:
                self.assertEqual(current, previous)
        self.assertEqual(state["jobs"], {})
        once = copy.deepcopy(state["assets"])
        twice = core.set_remaining_asset_inputs_gpt(self.root, 1)
        self.assertEqual(twice["assets"], once)

    def test_bulk_gpt_rejects_stale_plan_running_and_inflight_atomically(self):
        for version in (0, 2, True):
            with self.assertRaises(core.GateError):
                core.set_remaining_asset_inputs_gpt(self.root, version)
        with core.run_lock(self.root), self.assertRaises(core.GateError):
            core.set_remaining_asset_inputs_gpt(self.root, 1)
        core.reserve(self.root, "unrelated", "video", "E001-S001", 0)
        with self.assertRaises(core.GateError):
            core.set_remaining_asset_inputs_gpt(self.root, 1)
        self.assertEqual(len(asset_inputs.pending_assets(core.load_project(self.root))), 2)

    def test_gpt_choice_confirms_input_without_approving_or_generating_asset(self):
        state = core.set_asset_input(self.root, "hero", 0, 0, "gpt", "")
        asset = state["assets"][0]
        self.assertTrue(asset_inputs.confirmed(asset))
        self.assertEqual(asset["status"], "planned")
        self.assertEqual(asset["version"], 0)
        self.assertEqual(asset["creation_input"]["revision"], 1)
        self.assertEqual(state["jobs"], {})
        self.assertEqual([a["id"] for a in asset_inputs.pending_assets(state)], ["station"])

    def test_description_validation_and_stale_versions_are_atomic(self):
        invalid = [(0, 0, "unknown", ""), (True, 0, "gpt", ""), (0, True, "gpt", ""),
                   (1, 0, "gpt", ""), (0, 1, "gpt", ""), (0, 0, "custom", " "),
                   (0, 0, "custom", "x" * 4001), (0, 0, "gpt", "自定义")]
        for version, revision, mode, description in invalid:
            with self.subTest(version=version, revision=revision, mode=mode), self.assertRaises(ValueError):
                core.set_asset_input(self.root, "hero", version, revision, mode, description)
            self.assertEqual(self.asset()["creation_input"], asset_inputs.empty_input())
        core.set_asset_input(self.root, "hero", 0, 0, "custom", "蓝色短发，白衬衫")
        with self.assertRaises(core.GateError):
            core.set_asset_input(self.root, "hero", 0, 0, "gpt", "")

    def test_upload_requires_confirmation_and_uses_immutable_hash(self):
        state = core.add_asset_reference(self.root, "hero", 0, 0, data_url())
        asset = state["assets"][0]
        self.assertFalse(asset_inputs.confirmed(asset))
        self.assertEqual(asset_inputs.reference_paths(self.root, asset), [])
        ref = asset["creation_input"]["references"][0]
        self.assertEqual(ref["bytes"], len(png_data()))
        self.assertEqual(ref["sha256"], core.digest(self.root / ref["file"]))
        self.assertEqual(ref["mime"], "image/png")
        self.assertTrue(ref["file"].startswith("inputs/assets/hero/"))
        state = core.set_asset_input(self.root, "hero", 0, 1, "custom", "")
        paths = asset_inputs.reference_paths(self.root, state["assets"][0])
        self.assertEqual(paths, [self.root / ref["file"]])
        (self.root / ref["file"]).write_bytes(png_data(b"\x66\x66\x66"))
        with self.assertRaises(core.GateError):
            asset_inputs.reference_paths(self.root, state["assets"][0])

    def test_upload_and_removal_invalidate_confirmation_and_stale_request(self):
        core.set_asset_input(self.root, "hero", 0, 0, "custom", "白衬衫")
        core.add_asset_reference(self.root, "hero", 0, 1, data_url())
        asset = self.asset()
        self.assertFalse(asset["creation_input"]["confirmed"])
        self.assertEqual(asset["creation_input"]["revision"], 2)
        ref = asset["creation_input"]["references"][0]
        with self.assertRaises(core.GateError):
            core.remove_asset_reference(self.root, "hero", 0, 1, ref["id"])
        core.set_asset_input(self.root, "hero", 0, 2, "custom", "白衬衫")
        core.remove_asset_reference(self.root, "hero", 0, 3, ref["id"])
        self.assertTrue((self.root / ref["file"]).is_file())
        self.assertEqual(self.asset()["creation_input"]["references"], [])
        self.assertFalse(self.asset()["creation_input"]["confirmed"])

    def test_corrupt_image_never_publishes_asset_input(self):
        with self.assertRaises(ValueError):
            core.add_asset_reference(self.root, "hero", 0, 0, data_url(b"\x89PNG\r\n\x1a\ninvalid"))
        self.assertEqual(self.asset()["creation_input"], asset_inputs.empty_input())
        self.assertEqual(list(self.root.glob("inputs/assets/hero/*")), [])

    def test_rejects_untrusted_formats_encoded_urls_and_oversize_payloads(self):
        for value in ("https://example.com/image.png", "data:image/svg+xml;base64,PHN2Zz4=",
                      "data:image/png;base64,@@@", data_url(mime="image/jpeg"),
                      data_url(b"<script>boom</script>"), data_url(b"")):
            with self.subTest(value=value[:50]), self.assertRaises(ValueError):
                asset_inputs.decode_reference(value)
        with patch.object(asset_inputs, "MAX_REFERENCE_BYTES", 10), self.assertRaises(ValueError):
            asset_inputs.decode_reference(data_url())

    def test_reference_limit_and_duplicate_do_not_publish_files(self):
        core.add_asset_reference(self.root, "hero", 0, 0, data_url())
        with self.assertRaises(ValueError):
            core.add_asset_reference(self.root, "hero", 0, 1, data_url())
        with patch.object(asset_inputs, "MAX_REFERENCES", 1), self.assertRaises(ValueError):
            core.add_asset_reference(self.root, "hero", 0, 1, data_url(png_data(b"\xff\x00\x00")))
        self.assertEqual(len(list(self.root.glob("inputs/assets/hero/*"))), 1)

    def test_running_or_inflight_job_blocks_all_input_mutations(self):
        with core.run_lock(self.root), self.assertRaises(core.GateError):
            core.set_asset_input(self.root, "hero", 0, 0, "gpt", "")
        with core.run_lock(self.root), self.assertRaises(core.GateError):
            core.add_asset_reference(self.root, "hero", 0, 0, data_url())
        core.reserve(self.root, "unrelated", "video", "E001-S001", 0)
        with self.assertRaises(core.GateError):
            core.set_asset_input(self.root, "hero", 0, 0, "gpt", "")
        self.assertEqual(self.asset()["creation_input"], asset_inputs.empty_input())

    def test_existing_assets_are_not_regated_or_editable(self):
        original = core.load_project(self.root)
        cases = [("file", "media/old.png"), ("history", [{"file": "media/old.png"}]), ("version", 1)]
        for field, value in cases:
            with self.subTest(field=field), core.transaction(self.root) as state:
                state["assets"] = copy.deepcopy(original["assets"])
                state["assets"][0].pop("creation_input")
                state["assets"][0][field] = value
            state = core.load_project(self.root)
            self.assertEqual([a["id"] for a in asset_inputs.pending_assets(state)], ["station"])
            with self.assertRaises(core.GateError):
                core.set_asset_input(self.root, "hero", state["assets"][0]["version"], 0, "gpt", "")
        with core.transaction(self.root) as state:
            state["assets"] = copy.deepcopy(original["assets"])
            for asset in state["assets"]:
                asset.pop("creation_input")
        self.assertEqual(len(asset_inputs.pending_assets(core.load_project(self.root))), 2)
        core.reserve(self.root, "old-attempt", "image", "hero", 0)
        core.update_job(self.root, "old-attempt", state="failed", accounting="released")
        self.assertEqual([a["id"] for a in asset_inputs.pending_assets(core.load_project(self.root))], ["station"])

    def test_voice_only_accepts_description_and_retains_audition_rules(self):
        with core.transaction(self.root) as state:
            voice = {"id": "voice", "kind": "voice", "name": "角色声音", "prompt": "中年男声，中低音",
                     "character_id": "hero", "status": "planned", "version": 0, "history": [],
                     "feedback": "", "creation_input": asset_inputs.empty_input()}
            state["assets"].append(voice)
        with self.assertRaises(ValueError):
            core.add_asset_reference(self.root, "voice", 0, 0, data_url())
        core.set_asset_input(self.root, "voice", 0, 0, "custom", "清澈、明亮，请大喊测试")
        voice = self.asset("voice")
        from norlan.voice import audition_profile
        effective = {**voice, "prompt": asset_inputs.compose_prompt(voice)}
        # The stable voice allowlist never copies acting instructions as traits.
        profile = audition_profile(effective)
        self.assertNotIn("大喊", str(profile.get("traits")))
        self.assertNotIn("audition_text", voice)

    def test_prompt_preserves_story_definition_and_exact_custom_description(self):
        core.set_asset_input(self.root, "hero", 0, 0, "custom", "蓝色短发，白色衬衫")
        asset = self.asset()
        prompt = asset_inputs.compose_prompt(asset, "水彩动画")
        self.assertIn(asset["prompt"], prompt)
        self.assertIn("蓝色短发，白色衬衫", prompt)
        self.assertIn("水彩动画", prompt)
        self.assertNotIn("真实视觉参考", prompt)


if __name__ == "__main__":
    unittest.main()
