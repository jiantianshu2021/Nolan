"""Synthetic HTTP checks for editing a generated asset's exact prompt.

Every fixture and media byte is local and temporary. Worker/provider/GPT
entry points are forbidden; HTTP confirmation only records human consent.
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


IMAGE_MODEL = "doubao-seedream-4-0-250828"
ORIGINAL_PROMPT = "  合成已生成版本的完整提示词。\n原本的构图、服装与照明。\n\n"
EDITED_PROMPT = "  人工直接修改的完整提示词。\n保持原样的换行与前后空格。\n\n"


class DashboardAssetDirectEditTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        core.create_project(self.root, "PRIVATE-SYNTHETIC-NOVEL: 测试剧本原文。", mode="live")
        core.install_plan(self.root, pipeline.demo_plan())
        core.approve(self.root, "plan", "plan", 1)
        core.set_remaining_asset_inputs_gpt(self.root, 1)
        self.reference = self.root / "inputs/assets/hero/reference.png"
        self.reference.parent.mkdir(parents=True)
        self.reference.write_bytes(b"SYNTHETIC REFERENCE IMAGE")
        relative = str(self.reference.relative_to(self.root))
        with core.transaction(self.root) as state:
            hero = core._find(state, "asset", "hero")
            hero["creation_input"] = {
                **asset_inputs.empty_input(), "confirmed": True, "mode": "custom",
                "description": "合成参考人物",
                "references": [{"id": "synthetic-reference", "file": relative, "mime": "image/png",
                                "sha256": core.digest(self.reference), "bytes": self.reference.stat().st_size}],
            }
        candidate = asset_prompt_gate.prepare(self.root, "hero", ORIGINAL_PROMPT, [self.reference])
        asset_prompt_gate.approve(self.root, "hero", candidate["id"], candidate["request_hash"],
                                  model=IMAGE_MODEL)
        self.original_media = self.root / "media/assets/hero-v1.png"
        self.original_media.parent.mkdir(parents=True, exist_ok=True)
        self.original_media.write_bytes(b"SYNTHETIC COMPLETED IMAGE V1")
        with core.transaction(self.root) as state:
            hero = core._find(state, "asset", "hero")
            hero.update(status="candidate", version=1, file="media/assets/hero-v1.png",
                        sha256=core.digest(self.original_media),
                        generation_request=copy.deepcopy(hero["generation_prompt"]),
                        generation_model=IMAGE_MODEL)
            state["status"] = "awaiting_asset_approval"
        self.server = dashboard.make_server(self.root, port=0)
        self.port = self.server.server_address[1]
        self.thread = threading.Thread(
            target=lambda: self.server.serve_forever(poll_interval=0.02), daemon=True)
        self.thread.start()
        self.launch = self.enterContext(patch.object(self.server.context, "launch", return_value=True))
        self.run = self.enterContext(patch.object(dashboard, "_run",
                                                  side_effect=AssertionError("No production in HTTP tests")))
        self.ark = self.enterContext(patch.object(pipeline, "_ark",
                                                  side_effect=AssertionError("No provider in HTTP tests")))
        self.brain = self.enterContext(patch.object(pipeline, "_brain",
                                                    side_effect=AssertionError("No GPT in HTTP tests")))
        self.reject = self.enterContext(patch.object(core, "reject",
                                                     side_effect=AssertionError("No feedback/rewrite route")))

    def tearDown(self):
        self.server.context.running = False
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)
        self.run.assert_not_called()
        self.ark.assert_not_called()
        self.brain.assert_not_called()
        self.reject.assert_not_called()

    def request(self, method, payload=None, *, path="/api/prompt-edit", headers=None, omit_token=False):
        options = {"Content-Type": "application/json", "X-Norlan-Token": self.server.context.token}
        options.update(headers or {})
        if omit_token:
            options.pop("X-Norlan-Token")
        body = None if payload is None else json.dumps(payload, ensure_ascii=False).encode()
        connection = http.client.HTTPConnection("127.0.0.1", self.port, timeout=3)
        try:
            connection.request(method, path, body=body, headers=options)
            response = connection.getresponse()
            return response.status, json.loads(response.read())
        finally:
            connection.close()

    def asset(self):
        return core._find(core.load_project(self.root), "asset", "hero")

    def editor(self):
        status, result = self.request("GET", path="/api/asset-prompt-editor?id=hero&version=1")
        self.assertEqual(status, 200, result)
        return result

    def payload(self, **changes):
        snapshot = self.editor()
        payload = {"action": "asset_edit_and_generate", "id": "hero", "version": 1,
                   "sha256": snapshot["sha256"], "prompt": EDITED_PROMPT, "model": IMAGE_MODEL}
        payload.update(changes)
        return payload

    def assert_rejected_unchanged(self, payload, expected=409, **options):
        before = (self.root / "project.json").read_bytes()
        launches = self.launch.call_count
        status, result = self.request("POST", payload, **options)
        self.assertEqual(status, expected, result)
        self.assertTrue(result["error"])
        self.assertEqual((self.root / "project.json").read_bytes(), before)
        self.assertEqual(self.launch.call_count, launches)
        self.assertEqual(self.original_media.read_bytes(), b"SYNTHETIC COMPLETED IMAGE V1")

    def test_editor_prefills_actual_request_with_models_without_mutation_or_private_data(self):
        with core.transaction(self.root) as state:
            state["settings"]["private_note"] = "PRIVATE-SETTINGS"
            state["settings"]["ark"]["api_key"] = "PRIVATE-ARK-KEY"
            request = core._find(state, "asset", "hero")["generation_request"]
            request["provider_response"] = {"token": "PRIVATE-PROVIDER"}
        before = (self.root / "project.json").read_bytes()
        editor = self.editor()
        self.assertEqual(editor["id"], "hero")
        self.assertEqual(editor["version"], 1)
        self.assertEqual(editor["target_version"], 2)
        self.assertEqual(editor["prompt"], ORIGINAL_PROMPT)
        self.assertEqual(editor["model"], IMAGE_MODEL)
        self.assertRegex(editor["sha256"], r"^[0-9a-f]{64}$")
        self.assertTrue(any(item["id"] == IMAGE_MODEL and item["available"] for item in editor["models"]))
        self.assertEqual(editor["reference_manifest"]["images"], ["inputs/assets/hero/reference.png"])
        self.assertNotIn("PRIVATE-", json.dumps(editor, ensure_ascii=False))
        for field in ("settings", "jobs", "source_manifest", "generation_config", "provider_response"):
            self.assertNotIn(field, editor)
        self.assertEqual((self.root / "project.json").read_bytes(), before)
        self.assertEqual(self.editor(), editor, "Opening or reopening the editor cannot create a candidate")
        self.launch.assert_not_called()

    def test_human_exact_edit_authorizes_one_generation_without_rewrite_or_second_gate(self):
        before = core.load_project(self.root)
        payload = self.payload()
        status, result = self.request("POST", payload)
        self.assertEqual(status, 202, result)
        self.launch.assert_called_once_with()
        state, hero = core.load_project(self.root), self.asset()
        self.assertEqual(hero["status"], "asset_prompt_approved")
        self.assertEqual(hero["version"], 1, "Only a future worker may advance the media version")
        self.assertEqual(hero["generation_prompt"]["target_version"], 2)
        self.assertEqual(hero["generation_prompt"]["prompt"], EDITED_PROMPT)
        self.assertEqual(hero["generation_prompt"]["model"], IMAGE_MODEL)
        self.assertEqual(hero["generation_prompt_approval"]["decision"], "generate")
        self.assertIsNotNone(asset_prompt_gate.approved(self.root, "hero"))
        self.assertEqual(state["jobs"], before["jobs"])
        self.assertEqual(state["settings"], before["settings"])
        self.assertEqual(hero["file"], "media/assets/hero-v1.png")
        self.assertEqual(self.original_media.read_bytes(), b"SYNTHETIC COMPLETED IMAGE V1")
        self.assert_rejected_unchanged(payload)
        self.assertEqual(self.launch.call_count, 1, "A double click cannot authorize another version")

    def test_editor_requires_session_token_and_same_origin_and_host(self):
        path = "/api/asset-prompt-editor?id=hero&version=1"
        before = (self.root / "project.json").read_bytes()
        for options in ({"omit_token": True}, {"headers": {"X-Norlan-Token": "bad"}},
                        {"headers": {"Origin": "https://foreign.invalid"}},
                        {"headers": {"Host": "foreign.invalid"}}):
            with self.subTest(options=options):
                self.assertEqual(self.request("GET", path=path, **options)[0], 403)
        self.assertEqual((self.root / "project.json").read_bytes(), before)
        self.launch.assert_not_called()

    def test_editor_rejects_ambiguous_query_and_stale_version_without_mutation(self):
        before = (self.root / "project.json").read_bytes()
        invalid = ("", "?id=hero", "?id=hero&version=1&extra=1",
                   "?id=hero&id=station&version=1", "?id=hero&version=1&version=2",
                   "?id=../hero&version=1", "?id=hero&version=true",
                   "?id=hero&version=-1", "?id=hero&version=1234567890")
        for suffix in invalid:
            with self.subTest(suffix=suffix):
                self.assertEqual(self.request("GET", path="/api/asset-prompt-editor" + suffix)[0], 400)
        self.assertEqual(self.request("GET", path="/api/asset-prompt-editor?id=hero&version=2")[0], 409)
        self.assertEqual((self.root / "project.json").read_bytes(), before)
        self.launch.assert_not_called()

    def test_post_checks_csrf_origin_host_and_rejects_other_endpoints(self):
        payload = self.payload()
        for options in ({"omit_token": True}, {"headers": {"X-Norlan-Token": "bad"}},
                        {"headers": {"Origin": "https://foreign.invalid"}},
                        {"headers": {"Host": "foreign.invalid"}}):
            with self.subTest(options=options):
                self.assert_rejected_unchanged(payload, expected=403, **options)
        self.assert_rejected_unchanged(payload, path="/api/action")
        self.assert_rejected_unchanged({**payload, "action": "reject"})

    def test_post_requires_strict_fields_and_bounded_exact_text(self):
        payload = self.payload()
        invalid = ({"extra": True}, {"id": True}, {"id": "../hero"}, {"id": ""},
                   {"version": True}, {"version": -1}, {"version": "1"},
                   {"sha256": "broken"}, {"sha256": True},
                   {"prompt": " \n "}, {"prompt": True}, {"prompt": None},
                   {"prompt": "x\x00y"}, {"prompt": "x" * 100001},
                   {"model": True}, {"model": "bad model"}, {"model": "not-configured-model"})
        for changed in invalid:
            with self.subTest(changed=next(iter(changed))):
                self.assert_rejected_unchanged({**payload, **changed})
        for field in payload:
            with self.subTest(missing=field):
                self.assert_rejected_unchanged({key: value for key, value in payload.items() if key != field})

    def test_large_prompt_uses_dedicated_bounded_endpoint(self):
        prompt = "  完整中文资产提示词，保留人物与场景细节。\n" * 2000 + "\n"
        payload = self.payload(prompt=prompt)
        encoded = json.dumps(payload, ensure_ascii=False).encode()
        self.assertGreater(len(encoded), dashboard.MAX_BODY)
        self.assertLess(len(encoded), dashboard.MAX_PROMPT_EDIT_BODY)
        self.assertLessEqual(len(prompt), 100000)
        self.assert_rejected_unchanged(payload, expected=413, path="/api/action")
        self.assert_rejected_unchanged(payload, expected=413,
                                       headers={"Content-Length": str(dashboard.MAX_PROMPT_EDIT_BODY + 1)})
        status, result = self.request("POST", payload)
        self.assertEqual(status, 202, result)
        self.assertEqual(self.asset()["generation_prompt"]["prompt"], prompt)
        self.launch.assert_called_once_with()

    def test_busy_dashboard_external_worker_and_paused_project_preserve_draft(self):
        payload = self.payload()
        self.server.context.running = True
        self.assert_rejected_unchanged(payload)
        self.server.context.running = False
        with patch.object(self.server.context, "external_run_active", return_value=True):
            self.assert_rejected_unchanged(payload)
        core.set_paused(self.root, True)
        self.assert_rejected_unchanged(payload)

    def test_stale_editor_hash_and_changed_source_reject_without_generation(self):
        payload = self.payload()
        self.assert_rejected_unchanged({**payload, "sha256": "0" * 64})
        with core.transaction(self.root) as state:
            state["plan"]["style"] += " 合成新的画面风格。"
        self.assert_rejected_unchanged(payload)

    def test_changed_actual_reference_rejects_without_silent_omission_or_mutation(self):
        payload = self.payload()
        self.reference.write_bytes(b"SYNTHETIC REPLACED REFERENCE")
        self.assert_rejected_unchanged(payload)
        before = (self.root / "project.json").read_bytes()
        status, result = self.request("GET", path="/api/asset-prompt-editor?id=hero&version=1")
        self.assertEqual(status, 409, result)
        self.assertEqual((self.root / "project.json").read_bytes(), before)

    def test_unknown_prior_submission_blocks_resubmission_and_retains_original_media(self):
        payload = self.payload()
        with core.transaction(self.root) as state:
            state["jobs"]["synthetic-uncertain"] = {
                "id": "synthetic-uncertain", "kind": "image", "item_id": "hero",
                "state": "uncertain", "accounting": "unknown", "amount": 0,
            }
            core._find(state, "asset", "hero")["job_id"] = "synthetic-uncertain"
        self.assert_rejected_unchanged(payload)

    def test_another_failed_asset_blocks_before_saving_any_direct_generation_consent(self):
        with core.transaction(self.root) as state:
            core._find(state, "asset", "station")["status"] = "generation_failed"
        payload = self.payload()
        before = copy.deepcopy(self.asset())
        self.assert_rejected_unchanged(payload)
        self.assertEqual(self.asset(), before)
        self.assertIsNone(asset_prompt_gate.approved(self.root, "hero"))
        self.launch.assert_not_called()

    def test_explicit_retry_excludes_the_target_failed_asset_from_launch_precheck(self):
        with core.transaction(self.root) as state:
            state["jobs"]["synthetic-known-failure"] = {
                "id": "synthetic-known-failure", "kind": "image", "item_id": "hero",
                "state": "failed", "accounting": "spent", "amount": 0,
            }
            core._find(state, "asset", "hero").update(
                status="generation_failed", job_id="synthetic-known-failure",
                generation_error="合成测试：服务明确拒绝了前次生成。")
        payload = self.payload()
        status, result = self.request("POST", payload)
        self.assertEqual(status, 202, result)
        self.assertEqual(self.asset()["status"], "asset_prompt_approved")
        self.assertEqual(self.asset()["generation_prompt"]["prompt"], EDITED_PROMPT)
        self.assertIsNotNone(asset_prompt_gate.approved(self.root, "hero"))
        self.launch.assert_called_once_with()

    def test_multiple_known_failures_can_be_repaired_without_mutual_deadlock(self):
        with core.transaction(self.root) as state:
            state["jobs"]["synthetic-known-failure"] = {
                "id": "synthetic-known-failure", "kind": "image", "item_id": "hero",
                "state": "failed", "accounting": "spent", "amount": 0,
            }
            core._find(state, "asset", "hero").update(
                status="generation_failed", job_id="synthetic-known-failure",
                generation_error="合成测试：已核实失败。")
            core._find(state, "asset", "station")["status"] = "generation_failed"
        # Exercise the real launch guard, which stops before creating a worker.
        self.launch.side_effect = lambda: dashboard._Context.launch(self.server.context)
        status, result = self.request("POST", self.payload())
        self.assertEqual(status, 202, result)
        self.assertIn("已保存", result["dashboard"]["error"])
        self.assertIn("未启动", result["dashboard"]["error"])
        self.assertIsNotNone(asset_prompt_gate.approved(self.root, "hero"))
        self.assertEqual(core._find(core.load_project(self.root), "asset", "station")["status"],
                         "generation_failed")
        self.assertFalse(self.server.context.running)

    def assert_saved_consent_is_reported_when_launch_fails(self, exception):
        payload = self.payload()
        self.launch.side_effect = exception
        status, result = self.request("POST", payload)
        self.assertEqual(status, 202, result)
        self.assertNotIn("error", result, "A saved authorization is not an ordinary rejected edit")
        self.assertIsInstance(result.get("dashboard"), dict)
        message = result["dashboard"].get("error", "")
        self.assertIn("已保存", message)
        self.assertTrue("未启动" in message or "未能启动" in message, message)
        self.assertNotIn("PRIVATE-LAUNCH-DETAIL", message)
        self.assertFalse(result["dashboard"]["running"])
        hero = self.asset()
        self.assertEqual(hero["status"], "asset_prompt_approved")
        self.assertEqual(hero["generation_prompt"]["prompt"], EDITED_PROMPT)
        self.assertEqual(hero["generation_prompt"]["target_version"], 2)
        self.assertIsNotNone(asset_prompt_gate.approved(self.root, "hero"))
        self.assertEqual(core.load_project(self.root)["jobs"], {})
        self.assertEqual(self.original_media.read_bytes(), b"SYNTHETIC COMPLETED IMAGE V1")
        self.launch.assert_called_once_with()
        self.assert_rejected_unchanged(payload)
        self.assertEqual(self.launch.call_count, 1, "Launch failure cannot turn saved consent into a duplicate edit")

    def test_saved_consent_remains_explicitly_accepted_if_launch_validation_fails(self):
        self.assert_saved_consent_is_reported_when_launch_fails(ValueError("PRIVATE-LAUNCH-DETAIL"))

    def test_saved_consent_remains_explicitly_accepted_if_worker_start_fails(self):
        self.assert_saved_consent_is_reported_when_launch_fails(RuntimeError("PRIVATE-LAUNCH-DETAIL"))


if __name__ == "__main__":
    unittest.main()
