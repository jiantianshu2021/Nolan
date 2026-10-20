"""Synthetic credentials only: no real Keychain, model or network calls."""
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from norlan import core, mcp_server
from norlan.providers import ArkProvider, ProviderError


class ArkCredentialsFlowTests(unittest.TestCase):
    def test_provider_environment_precedes_keychain(self):
        with patch.dict(os.environ, {"ARK_API_KEY": "synthetic-env-secret"}, clear=True), \
                patch("norlan.credentials.load_ark_key", side_effect=AssertionError("no native access")):
            provider = ArkProvider({"video_model": "seedance-test"})
            self.assertEqual(provider._key(), "synthetic-env-secret")

    def test_new_provider_without_environment_reuses_saved_key_and_never_submits(self):
        with patch.dict(os.environ, {}, clear=True), \
                patch("norlan.credentials.load_ark_key", return_value="synthetic-stored-secret") as stored, \
                patch("norlan.providers._api_open") as network:
            first = ArkProvider({"video_model": "seedance-test"})
            second = ArkProvider({"video_model": "seedance-test"})
            first.preflight("video")
            self.assertEqual(second._key(), "synthetic-stored-secret")
            self.assertGreaterEqual(stored.call_count, 2)
            network.assert_not_called()

    def test_missing_saved_key_gives_one_time_login_instructions_before_network(self):
        with patch.dict(os.environ, {}, clear=True), patch("norlan.credentials.load_ark_key", return_value=None), \
                patch("norlan.providers._api_open") as network:
            with self.assertRaisesRegex(ProviderError, "nolan ark-login") as caught:
                ArkProvider({"video_model": "seedance-test"})
            self.assertFalse(caught.exception.uncertain)
            self.assertEqual(caught.exception.failure["stage"], "local_validation")
            network.assert_not_called()

    def test_locked_keychain_is_distinct_from_missing_key(self):
        from norlan.credentials import CredentialError
        with patch.dict(os.environ, {}, clear=True), \
                patch("norlan.credentials.load_ark_key", side_effect=CredentialError("macOS 钥匙串已锁定，请在本机解锁。")):
            with self.assertRaisesRegex(ProviderError, "钥匙串.*解锁") as caught:
                ArkProvider({"video_model": "seedance-test"})
        self.assertNotIn("未配置", str(caught.exception))
        self.assertFalse(caught.exception.uncertain)

    def test_missing_and_locked_diagnostics_give_different_remedies(self):
        with patch.dict(os.environ, {}, clear=True), patch("norlan.credentials.load_ark_key", return_value=None):
            with self.assertRaises(ProviderError) as caught:
                ArkProvider({"video_model": "seedance-test"})
        missing = mcp_server._diagnostic(str(caught.exception))
        self.assertIn("nolan ark-login", missing["message"])
        locked = mcp_server._diagnostic("macOS 钥匙串已锁定或需要授权；请解锁。（OSStatus -25308）")
        self.assertIn("解锁", locked["message"])
        self.assertNotIn("隐藏输入并保存一次", locked["message"])

    def test_custom_environment_never_uses_default_saved_key(self):
        with patch.dict(os.environ, {}, clear=True), \
                patch("norlan.credentials.load_ark_key", side_effect=AssertionError("wrong key scope")):
            with self.assertRaisesRegex(ProviderError, "指定.*环境变量"):
                ArkProvider({"api_key_env": "SYNTHETIC_CUSTOM_ARK", "video_model": "seedance-test"})
        with patch.dict(os.environ, {"SYNTHETIC_CUSTOM_ARK": "synthetic-custom"}, clear=True), \
                patch("norlan.credentials.load_ark_key", side_effect=AssertionError("wrong key scope")):
            self.assertEqual(ArkProvider({"api_key_env": "SYNTHETIC_CUSTOM_ARK"})._key(), "synthetic-custom")

    def test_mcp_status_has_no_secret_no_project_mutations_and_no_network(self):
        with tempfile.TemporaryDirectory(prefix="synthetic-keychain-mcp-") as temp:
            server = mcp_server.MCPServer(Path(temp) / "workspace")
            self.addCleanup(server.close)
            project = server.projects / "synthetic"
            core.create_project(project, "合成凭据状态测试，不调用供应商。")
            before = (project / "project.json").read_bytes()
            status = {"available": True, "source": "keychain", "message": "已保存，自动读取。",
                      "private_future_field": "synthetic-must-not-be-exposed"}
            with patch("norlan.credentials.ark_key_status", return_value=status), \
                    patch("norlan.providers._api_open") as network:
                result = server.call_tool("get_credentials_status", {})
                network.assert_not_called()
            self.assertEqual(result["ark"]["source"], "keychain")
            self.assertNotIn("synthetic-must-not-be-exposed", json.dumps(result))
            self.assertEqual((project / "project.json").read_bytes(), before)
            tool = mcp_server.TOOL_MAP["get_credentials_status"]
            self.assertTrue(tool["annotations"]["readOnlyHint"])


if __name__ == "__main__":
    unittest.main()
