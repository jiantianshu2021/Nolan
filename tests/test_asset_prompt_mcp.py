"""Synthetic status projection: no keys, external services or real approvals."""
import json
from pathlib import Path
import tempfile
import unittest

from norlan import asset_prompt_gate, core, mcp_server, pipeline


class AssetPromptMCPTests(unittest.TestCase):
    def test_conversation_gets_gate_counts_while_browser_keeps_complete_editor(self):
        with tempfile.TemporaryDirectory(prefix='synthetic-asset-mcp-') as temporary:
            root = Path(temporary) / 'project'
            core.create_project(root, 'SYNTHETIC LOCAL MCP PROJECTION TEST', mode='live')
            core.install_plan(root, pipeline.demo_plan())
            core.approve(root, 'plan', 'plan', 1)
            core.set_remaining_asset_inputs_gpt(root, 1)
            state = pipeline.run(root)
            candidate = state['assets'][0]['generation_prompt']
            marker = 'SYNTHETIC MANUAL FINAL ASSET PROMPT '
            manual = marker + '演示提示词。' * 3000
            asset_prompt_gate.approve(root, 'hero', candidate['id'], candidate['request_hash'], prompt=manual)
            state = core.load_project(root)
            before = (root / 'project.json').read_bytes()
            server = mcp_server.MCPServer(Path(temporary) / 'workspace', project=root)
            try:
                result = server._status(root)
            finally:
                server.close()
            projected = result['assets'][0]['generation_prompt']
            self.assertEqual(projected['prompt_chars'], len(manual))
            self.assertEqual(projected['target_version'], 1)
            self.assertEqual(projected['reference_image_count'], 0)
            self.assertNotIn(marker, json.dumps(result, ensure_ascii=False))
            self.assertNotIn('generation_models', result['assets'][0])
            from norlan import dashboard
            browser = dashboard.public_state(state)['assets'][0]
            self.assertEqual(browser['generation_prompt']['prompt'], manual)
            self.assertTrue(browser['generation_models'])
            self.assertEqual((root / 'project.json').read_bytes(), before)
            self.assertFalse(state['jobs'])


if __name__ == '__main__':
    unittest.main()
