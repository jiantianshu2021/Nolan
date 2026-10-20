"""Synthetic localhost asset prompt HTTP consent checks, never real production.

The server and core mutations are real; its worker launch and all provider/GPT
entry points are mocked. Every project belongs to a temporary test directory.
"""
import copy
import http.client
import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

from norlan import asset_inputs, asset_prompt_gate, core, dashboard, pipeline


IMAGE_CHOICE = "doubao-seedream-4-0-250828"
INCOMPATIBLE_VOICE = "doubao-seedance-1-0-pro-250528"


class DashboardAssetPromptApiTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        core.create_project(self.root, "SECRET-SYNTHETIC-SOURCE: 只保存在测试小说里。", mode="live")
        core.install_plan(self.root, pipeline.demo_plan())
        core.approve(self.root, "plan", "plan", 1)
        core.set_remaining_asset_inputs_gpt(self.root, 1)
        core.configure(self.root, {"caps": {"image": 1, "video_second": 1}}, limit=200)
        for asset in core.load_project(self.root)["assets"]:
            asset_prompt_gate.prepare(self.root, asset["id"], "合成完整提示词：" + asset["name"], [])
        self.server = dashboard.make_server(self.root, port=0)
        self.port = self.server.server_address[1]
        self.thread = threading.Thread(target=lambda: self.server.serve_forever(poll_interval=0.02), daemon=True)
        self.thread.start()
        self.launch = self.enterContext(patch.object(self.server.context, "launch", return_value=True))
        self.approve = self.enterContext(patch.object(core, "approve_asset_prompt", wraps=core.approve_asset_prompt))
        self.run = self.enterContext(patch.object(dashboard, "_run", side_effect=AssertionError("No production in HTTP tests")))
        self.ark = self.enterContext(patch.object(pipeline, "_ark", side_effect=AssertionError("No provider in HTTP tests")))
        self.brain = self.enterContext(patch.object(pipeline, "_brain", side_effect=AssertionError("No GPT in HTTP tests")))

    def tearDown(self):
        self.server.context.running = False
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)
        self.run.assert_not_called()
        self.ark.assert_not_called()
        self.brain.assert_not_called()

    def request(self, method, payload=None, *, headers=None, path="/api/action", omit_token=False):
        connection = http.client.HTTPConnection("127.0.0.1", self.port, timeout=3)
        options = {"Content-Type": "application/json", "X-Norlan-Token": self.server.context.token} if method == "POST" else {}
        options.update(headers or {})
        if omit_token:
            options.pop("X-Norlan-Token", None)
        body = json.dumps(payload, ensure_ascii=False).encode() if payload is not None else None
        try:
            connection.request(method, path, body=body, headers=options)
            response = connection.getresponse()
            return response.status, json.loads(response.read())
        finally:
            connection.close()

    def asset(self, identifier="hero"):
        return core._find(core.load_project(self.root), "asset", identifier)

    def payload(self, identifier="hero", **changes):
        candidate = self.asset(identifier)["generation_prompt"]
        result = {"action": "asset_prompt_approve", "id": identifier, "candidate_id": candidate["id"],
                  "sha256": candidate["request_hash"], "model": IMAGE_CHOICE}
        result.update(changes)
        return result

    def assert_rejected_without_mutation(self, payload, expected=409, **request_options):
        before = (self.root / "project.json").read_bytes()
        calls_before = self.launch.call_count
        status, result = self.request("POST", payload, **request_options)
        self.assertEqual(status, expected, result)
        self.assertTrue(result["error"])
        self.assertEqual((self.root / "project.json").read_bytes(), before)
        self.assertEqual(self.launch.call_count, calls_before)

    def reset_directions(self):
        with core.transaction(self.root) as state:
            for asset in state["assets"]:
                asset["status"] = "planned"
                asset["creation_input"] = asset_inputs.empty_input()
                for key in ("generation_prompt", "generation_prompt_approval", "generation_prompt_history"):
                    asset.pop(key, None)
            state["status"] = "awaiting_asset_input"

    def test_get_exposes_final_prompt_model_options_and_omits_config_source_and_private_fields(self):
        with core.transaction(self.root) as state:
            state["settings"]["private_note"] = "SECRET-SETTINGS"
            state["settings"]["ark"]["api_key"] = "SECRET-ARK-KEY"
            asset = core._find(state, "asset", "hero")
            asset["generation_prompt"]["generation_config"]["api_key"] = "SECRET-CANDIDATE-CONFIG"
            asset["generation_prompt"]["provider_response"] = {"token": "SECRET-PROVIDER"}
            asset["generation_prompt"]["source_manifest"]["private_note"] = "SECRET-SOURCE-MANIFEST"
        before = (self.root / "project.json").read_bytes()
        status, result = self.request("GET", path="/api/state")
        self.assertEqual(status, 200)
        hero = next(asset for asset in result["assets"] if asset["id"] == "hero")
        candidate = hero["generation_prompt"]
        self.assertEqual(candidate["prompt"], self.asset()["generation_prompt"]["prompt"])
        self.assertEqual(candidate["target_version"], 1)
        self.assertEqual(candidate["sha256"], self.asset()["generation_prompt"]["request_hash"])
        self.assertEqual(hero["selected_generation_model"], candidate["model"])
        self.assertTrue(any(item["id"] == IMAGE_CHOICE and item["available"] for item in hero["generation_models"]))
        self.assertNotIn("settings", result)
        self.assertNotIn("jobs", result)
        self.assertNotIn("source_manifest", candidate)
        self.assertNotIn("generation_config", candidate)
        self.assertNotIn("provider_response", candidate)
        self.assertNotIn("SECRET-", json.dumps(result, ensure_ascii=False))
        self.assertEqual((self.root / "project.json").read_bytes(), before)
        self.approve.assert_not_called()
        self.launch.assert_not_called()

    def test_human_exact_prompt_and_model_are_approved_once_then_worker_launches_once(self):
        prompt = "  合成人工最终提示词。\n保留首尾空格与换行。\n\n"
        payload = self.payload(prompt=prompt)
        original_settings = copy.deepcopy(core.load_project(self.root)["settings"])
        status, result = self.request("POST", payload, path="/api/prompt-edit")
        self.assertEqual(status, 202, result)
        self.approve.assert_called_once_with(self.root, "hero", payload["candidate_id"], payload["sha256"],
                                              model=IMAGE_CHOICE, prompt=prompt)
        self.launch.assert_called_once_with()
        state = core.load_project(self.root)
        hero = self.asset()
        self.assertEqual(hero["status"], "asset_prompt_approved")
        self.assertEqual(hero["version"], 0, "Approval itself does not create media or advance versions")
        self.assertEqual(hero["generation_prompt"]["prompt"], prompt)
        self.assertEqual(hero["generation_prompt"]["model"], IMAGE_CHOICE)
        self.assertEqual(hero["generation_prompt_approval"]["source"], "human_manual_edit")
        self.assertEqual(self.asset("station")["status"], "awaiting_asset_prompt_approval")
        self.assertEqual(state["jobs"], {})
        self.assertEqual(state["settings"], original_settings)
        self.assert_rejected_without_mutation(payload)
        self.assertEqual(self.launch.call_count, 1, "Duplicate clicks cannot launch a second worker")

    def test_large_chinese_prompt_uses_dedicated_endpoint_and_keeps_ordinary_action_cap(self):
        prompt = "  完整中文资产提示词，保留人物与场景细节。\n" * 2000 + "\n"
        payload = self.payload(prompt=prompt)
        encoded = json.dumps(payload, ensure_ascii=False).encode()
        self.assertGreater(len(encoded), dashboard.MAX_BODY)
        self.assertLess(len(encoded), dashboard.MAX_PROMPT_EDIT_BODY)
        self.assertLessEqual(len(prompt), 100000)
        self.assert_rejected_without_mutation(payload, expected=413)
        self.approve.assert_not_called()
        self.launch.assert_not_called()
        status, result = self.request("POST", payload, path="/api/prompt-edit")
        self.assertEqual(status, 202, result)
        self.assertEqual(self.asset()["generation_prompt"]["prompt"], prompt)
        self.launch.assert_called_once()
        self.assertEqual(self.approve.call_args.kwargs["prompt"], prompt)

    def test_asset_prompt_edit_retains_dedicated_body_and_text_bounds(self):
        self.assert_rejected_without_mutation(self.payload(), expected=413, path="/api/prompt-edit",
            headers={"Content-Length": str(dashboard.MAX_PROMPT_EDIT_BODY + 1)})
        self.assert_rejected_without_mutation(self.payload(prompt="字" * 100001), path="/api/prompt-edit")
        self.assert_rejected_without_mutation({**self.payload(), "action": "run"}, path="/api/prompt-edit")
        self.launch.assert_not_called()
        self.approve.assert_not_called()

    def test_unedited_prompt_approval_retains_exact_prepared_text(self):
        prepared = self.asset()["generation_prompt"]["prompt"]
        status, result = self.request("POST", self.payload())
        self.assertEqual(status, 202, result)
        self.assertEqual(self.asset()["generation_prompt"]["prompt"], prepared)
        self.assertEqual(self.asset()["generation_prompt_approval"]["source"], "human_confirmation")
        self.launch.assert_called_once()

    def test_invalid_models_ids_hashes_text_and_extra_fields_never_start_or_mutate(self):
        invalid = [{"model": value} for value in (None, True, 3, [], {}, "", "bad model", "模型", "x" * 201,
                                                 "https://private.invalid/api", "unknown-model")]
        invalid += [{"id": True}, {"id": ""}, {"candidate_id": "old-candidate"}, {"candidate_id": ""},
                    {"sha256": "b" * 64}, {"sha256": False}, {"sha256": "invalid"},
                    {"prompt": None}, {"prompt": " \n "}, {"prompt": True}, {"prompt": "x\x00y"},
                    {"settings": {"video_model": "unknown"}}, {"version": 1}, {"duration": 30},
                    {"reference_url": "https://private.invalid/SECRET"}, {"approve": True}]
        for changed in invalid:
            with self.subTest(changed=repr(changed)):
                self.assert_rejected_without_mutation(self.payload(**changed))
        self.launch.assert_not_called()
        self.assertNotIn("generation_prompt_approval", self.asset())

    def test_missing_required_fields_are_rejected_without_approval_or_launch(self):
        valid = self.payload()
        for key in valid:
            with self.subTest(missing=key):
                self.assert_rejected_without_mutation({name: value for name, value in valid.items() if name != key})
        self.launch.assert_not_called()
        self.assertNotIn("generation_prompt_approval", self.asset())

    def test_stale_asset_version_is_rejected_even_if_candidate_id_and_hash_match(self):
        payload = self.payload()
        with core.transaction(self.root) as state:
            core._find(state, "asset", "hero")["version"] = 1
        self.assert_rejected_without_mutation(payload)
        self.launch.assert_not_called()

    def test_csrf_host_and_origin_cannot_approve_or_launch(self):
        for headers in ({"X-Norlan-Token": ""}, {"X-Norlan-Token": "wrong"},
                        {"Host": "foreign.invalid"}, {"Origin": "https://foreign.invalid"}):
            with self.subTest(headers=headers):
                self.assert_rejected_without_mutation(self.payload(), expected=403, headers=headers)
        self.assert_rejected_without_mutation(self.payload(), expected=403, omit_token=True)
        self.approve.assert_not_called()
        self.launch.assert_not_called()

    def test_busy_external_run_and_pause_block_without_approval(self):
        self.server.context.running = True
        self.assert_rejected_without_mutation(self.payload())
        self.server.context.running = False
        with patch.object(self.server.context, "external_run_active", return_value=True):
            self.assert_rejected_without_mutation(self.payload())
        self.approve.assert_not_called()
        core.set_paused(self.root, True)
        self.assert_rejected_without_mutation(self.payload())
        self.assertNotIn("generation_prompt_approval", self.asset())
        self.launch.assert_not_called()

    def test_voice_model_compatibility_rejection_never_approves_or_starts(self):
        with core.transaction(self.root) as state:
            for asset in state["assets"]:
                path = self.root / "media" / (asset["id"] + ".png")
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"SYNTHETIC APPROVED IMAGE")
                asset.update(status="candidate", version=1, file=str(path.relative_to(self.root)), sha256=core.digest(path))
        for asset in core.load_project(self.root)["assets"]:
            core.approve(self.root, "asset", asset["id"], asset["version"])
        voice = {"id": "synthetic-voice", "kind": "voice", "name": "合成声音", "character_id": "hero",
            "prompt": "中性声音试听", "status": "planned", "version": 0, "history": [], "feedback": "",
            "creation_input": {**asset_inputs.empty_input(), "mode": "gpt", "confirmed": True, "revision": 1}}
        with core.transaction(self.root) as state:
            state["assets"].append(voice)
        asset_prompt_gate.prepare(self.root, voice["id"], "合成完整试听提示词", [self.root / "media/hero.png"], duration=6)
        status, result = self.request("GET", path="/api/state")
        self.assertEqual(status, 200)
        public = next(asset for asset in result["assets"] if asset["id"] == voice["id"])
        self.assertTrue(any(option["id"] == INCOMPATIBLE_VOICE and not option["available"] and option["reason"]
                            for option in public["generation_models"]))
        self.assert_rejected_without_mutation(self.payload(voice["id"], model=INCOMPATIBLE_VOICE))
        self.assertNotIn("generation_prompt_approval", self.asset(voice["id"]))
        self.launch.assert_not_called()

    def test_saving_live_direction_launches_only_preparation_without_provider_or_approval(self):
        self.reset_directions()
        status, result = self.request("POST", {"action": "asset_input", "id": "hero", "version": 0,
            "input_revision": 0, "mode": "custom", "description": "合成人物应穿白色衬衫"})
        self.assertEqual(status, 200, result)
        self.launch.assert_called_once_with()
        self.approve.assert_not_called()
        state = core.load_project(self.root)
        self.assertEqual(state["jobs"], {})
        self.assertEqual(self.asset()["version"], 0)
        self.assertNotIn("generation_prompt_approval", self.asset())
        self.assertEqual(self.asset()["creation_input"]["description"], "合成人物应穿白色衬衫")
        self.assertEqual(self.asset()["status"], "planned")

    def test_bulk_gpt_directions_launch_preparation_once_without_media_authorization(self):
        self.reset_directions()
        status, result = self.request("POST", {"action": "asset_inputs_gpt_remaining", "plan_version": 1})
        self.assertEqual(status, 200, result)
        self.launch.assert_called_once_with()
        self.approve.assert_not_called()
        state = core.load_project(self.root)
        self.assertEqual(state["jobs"], {})
        for asset in state["assets"]:
            self.assertEqual(asset["version"], 0)
            self.assertEqual(asset["status"], "planned")
            self.assertTrue(asset["creation_input"]["confirmed"])
            self.assertNotIn("generation_prompt_approval", asset)


if __name__ == "__main__":
    unittest.main()
