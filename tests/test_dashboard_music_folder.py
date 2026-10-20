"""Synthetic local-folder UI contracts. No downloads, GPT or paid media calls."""
import copy
import http.client
import json
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import threading
import unittest
from unittest.mock import patch

from norlan import dashboard
from norlan.music_folder import MusicFolderError


def synthetic_state():
    return {"id": "synthetic-music-folder", "title": "合成本地曲库测试", "mode": "demo", "paused": True,
            "status": "paused", "plan": {"status": "approved", "version": 1}, "assets": [], "jobs": {},
            "budget": {}, "settings": {"postproduction": {"enabled": True}, "music": {
                "mode": "library", "folder": "/synthetic/music", "api_key": "PRIVATE-KEY",
                "folder_scan": {"folder": "/synthetic/music", "scanned_at": "2026-10-20T01:00:00+00:00",
                    "imported": [{"id": "theme", "title": "合成测试曲", "version": 1, "provider_raw": "PRIVATE-SCAN"}],
                    "duplicates": 1, "total_candidates": 3, "skipped": [{"name": "invalid.mp3", "reason": "音频无效"}],
                    "provider_raw": "PRIVATE-REPORT"}}},
            "music_library": {"tracks": [{"id": "theme", "title": "合成测试曲", "version": 1,
                "file": "music/library/theme.mp3", "sha256": "a" * 64, "duration": 60, "status": "awaiting_approval",
                "metadata": {"author": "Synthetic Composer", "source_url": "https://example.org/synthetic",
                    "license": {"name": "CC BY 4.0", "attribution_required": True, "api_key": "PRIVATE-LICENSE"},
                    "attribution": "Synthetic music credit", "editorial": {"moods": ["安静"],
                        "suggested_use": "合成测试", "provider_raw": "PRIVATE-EDITORIAL"},
                    "download_url": "https://example.org/private-download", "provider_raw": "PRIVATE-META"},
                "provenance": {"api_key": "PRIVATE-PROVENANCE"}}]},
            "episodes": [{"id": "E1", "title": "合成第一集", "status": "generating", "version": 0,
                "shots": [{"id": "S1", "status": "planned", "duration": 5, "version": 0}],
                "postproduction": {"music": {"credits_file": "exports/E1.credits.txt", "credits_sha256": "c" * 64}}}]}


class MusicFolderProjectionTests(unittest.TestCase):
    def test_saved_unconfigured_music_label_uses_actual_local_library_state(self):
        state = synthetic_state()
        state["episodes"][0]["postproduction"]["music"]["status"] = "unconfigured"
        self.assertEqual(dashboard.public_state(state)["episodes"][0]["postproduction"]["music"]["status"], "awaiting_approval")
        state["music_library"]["tracks"][0]["status"] = "approved"
        self.assertEqual(dashboard.public_state(state)["episodes"][0]["postproduction"]["music"]["status"], "library_ready")

    def test_local_folder_report_and_license_are_public_but_credentials_are_not(self):
        state = synthetic_state()
        before = copy.deepcopy(state)
        public = dashboard.public_state(state)
        library = public["music_library"]
        self.assertEqual(library["folder"], "/synthetic/music")
        self.assertEqual(library["mode"], "library")
        self.assertEqual(library["folder_scan"]["duplicates"], 1)
        self.assertEqual(library["tracks"][0]["metadata"]["license"]["name"], "CC BY 4.0")
        self.assertEqual(library["tracks"][0]["metadata"]["editorial"]["suggested_use"], "合成测试")
        self.assertNotIn("PRIVATE-", json.dumps(public))
        self.assertNotIn("private-download", json.dumps(public))
        self.assertFalse(public["episodes"][0]["postproduction"]["ready"])
        self.assertEqual(state, before)

    def test_only_explicit_music_credits_text_is_served(self):
        state = synthetic_state()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            (root / "exports").mkdir()
            credit = root / "exports/E1.credits.txt"
            credit.write_text("Synthetic credit", encoding="utf-8")
            (root / "exports/private.txt").write_text("private", encoding="utf-8")
            state["assets"] = [{"file": "exports/private.txt"}]
            self.assertEqual(dashboard.resolve_media(root, "exports/E1.credits.txt", state), credit)
            self.assertIsNone(dashboard.resolve_media(root, "exports/private.txt", state))
            self.assertIsNone(dashboard.resolve_media(root, "exports/../private.txt", state))
            credit.unlink()
            credit.symlink_to(root / "exports/private.txt")
            self.assertIsNone(dashboard.resolve_media(root, "exports/E1.credits.txt", state))

    def test_library_migration_hides_only_obsolete_music_api_notices(self):
        state = synthetic_state()
        post = state["episodes"][0]["postproduction"]
        post["missing_configuration"] = [
            "自动生成音乐需配置火山纯音乐服务（独立AK/SK）及单次费用上限；也可直接导入本地音乐。",
            "未找到火山音乐AK/SK；方舟API Key不能代替。可在本机配置，或导入本地音乐。",
            "需要首次导入音乐并试听确认；自动生成音乐另需火山音乐 AK/SK 和调用预算。",
            "本地字幕识别模型尚未安装，请先配置本地识别。",
            "字幕校对要求ChatGPT订阅连接；请在终端运行 nolan setup 选择订阅。",
            "音乐文件夹不存在或无法读取，请检查本机路径。"]
        post["music"].update(can_generate=True, error="音乐文件夹不存在或无法读取，请检查本机路径。")
        before = copy.deepcopy(state)
        with patch("norlan.episode_post.music_configuration", return_value=(True, "synthetic stale capability")):
            public = dashboard.public_state(state)["episodes"][0]["postproduction"]
        self.assertEqual(public["missing_configuration"], post["missing_configuration"][3:])
        self.assertFalse(public["music"]["can_generate"])
        self.assertIn("文件夹不存在", public["music"]["error"])
        self.assertEqual(state, before)
        state["settings"]["postproduction"]["enabled"] = False
        self.assertFalse(dashboard.public_state(state)["episodes"][0]["postproduction"]["music"]["can_generate"])


class MusicFolderHttpTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.state = synthetic_state()
        self.core = self.enterContext(patch.object(dashboard, "_core")).return_value
        self.core.load_project.return_value = self.state
        self.run = self.enterContext(patch.object(dashboard, "_run"))
        self.server = dashboard.make_server(self.root, port=0)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.addCleanup(self.shutdown)

    def shutdown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)

    def request(self, payload=None, path="/api/action", headers=None, method="POST"):
        connection = http.client.HTTPConnection("127.0.0.1", self.server.server_address[1], timeout=3)
        body = json.dumps(payload) if payload is not None else None
        request_headers = {"Content-Type": "application/json", "X-Norlan-Token": self.server.context.token}
        request_headers.update(headers or {})
        connection.request(method, path, body, request_headers)
        response = connection.getresponse()
        status, data, response_headers = response.status, response.read(), dict(response.getheaders())
        connection.close()
        return status, data, response_headers

    def test_scan_before_all_segments_approved_does_not_resume_video(self):
        status, data, _ = self.request({"action": "music_folder_scan", "folder": "/synthetic/music"})
        self.assertEqual(status, 200)
        self.core.scan_music_folder.assert_called_once_with(self.root, "/synthetic/music")
        self.run.assert_not_called()
        self.core.set_paused.assert_not_called()
        self.core.request_postproduction.assert_not_called()
        self.assertTrue(json.loads(data)["paused"])

    def test_saved_folder_scan_and_human_action_validation(self):
        self.assertEqual(self.request({"action": "music_folder_scan"})[0], 200)
        self.core.scan_music_folder.assert_called_once_with(self.root, None)
        for payload in ({"action": "music_folder_scan", "folder": ""},
                        {"action": "music_folder_scan", "folder": 3},
                        {"action": "music_folder_scan", "folder": "/synthetic", "recursive": True},
                        {"action": "music_folder_scan", "folder": "/synthetic", "approve": True}):
            self.assertEqual(self.request(payload)[0], 409)
        self.assertEqual(self.request({"action": "music_folder_scan"}, headers={"X-Norlan-Token": "invalid"})[0], 403)
        self.assertEqual(self.request({"action": "music_folder_scan"}, headers={"Origin": "https://invalid.example"})[0], 403)

    def test_scan_errors_are_actionable_and_active_production_is_not_interrupted(self):
        self.core.scan_music_folder.side_effect = MusicFolderError("音乐文件夹不存在或无法读取，请检查本机路径。")
        status, data, _ = self.request({"action": "music_folder_scan"})
        self.assertEqual(status, 409)
        self.assertIn("文件夹不存在", json.loads(data)["error"])
        self.core.scan_music_folder.reset_mock()
        self.server.context.running = True
        self.assertEqual(self.request({"action": "music_folder_scan"})[0], 409)
        self.core.scan_music_folder.assert_not_called()

    def test_music_credits_download_is_an_attachment(self):
        target = self.root / "exports/E1.credits.txt"
        target.parent.mkdir()
        target.write_text("Synthetic music attribution", encoding="utf-8")
        status, data, headers = self.request(None, "/media/exports/E1.credits.txt", method="GET")
        self.assertEqual(status, 200)
        self.assertEqual(data, b"Synthetic music attribution")
        self.assertIn("attachment", headers["Content-Disposition"])


@unittest.skipUnless(shutil.which("node"), "Node.js required for synthetic DOM test")
class MusicFolderRenderingTests(unittest.TestCase):
    def test_early_folder_setup_and_audition_do_not_unlock_post_or_paid_generation(self):
        html = Path(dashboard.__file__).with_name("dashboard.html").read_text()
        sources = []
        for name in ("el", "mediaURL", "appendPrompt", "postproductionFeedback", "musicTrack", "postproductionEditor", "subtitleSyncReport", "postproductionPanel"):
            match = re.search(r"(?:async )?function " + name + r"\(", html)
            following = re.search(r"\n(?:async )?function ", html[match.start() + 1:])
            end = match.start() + 1 + following.start() if following else len(html)
            sources.append(html[match.start():end])
        program = r'''
const assert=require('node:assert/strict'),vm=require('node:vm');
const sources=JSON.parse(require('node:fs').readFileSync(0,'utf8'));
class Element{constructor(tag){this.tag=tag;this.children=[];this.dataset={};this.listeners={};this.textContent='';}append(...nodes){this.children.push(...nodes);}setAttribute(k,v){this[k]=v;}addEventListener(k,v){this.listeners[k]=v;}}
const actions=[],track={id:'theme',title:'合成音乐',version:2,sha256:'a'.repeat(64),file:'music/library/theme.mp3',status:'awaiting_approval',metadata:{author:'Synthetic Author',license:{name:'CC BY 4.0',attribution_required:true},attribution:'Synthetic attribution',editorial:{moods:['克制'],suggested_use:'合成悬念'}}};
const state={paused:true,dashboard:{running:false},music_library:{mode:'library',folder:'/synthetic/music',tracks:[track],folder_scan:{imported:[track],duplicates:2,skipped:[{name:'bad.mp3',reason:'音频无效'}]}}};
const context={state,assert,actions,document:{createElement:tag=>new Element(tag)},perform:(...args)=>actions.push(args),uploadMusic:()=>{},toast:()=>{}};
vm.runInNewContext(sources.join('\n')+`
const walk=node=>[node,...(node.children||[]).flatMap(walk)];
const episode={id:'E1',title:'第一集',version:0,postproduction:{enabled:true,ready:false,subtitles:{},music:{can_generate:true,credits_file:'exports/E1.credits.txt'}}};
let nodes=walk(postproductionPanel(episode));
const scan=nodes.find(node=>node.tag==='button'&&node.textContent==='扫描音乐文件夹');assert.ok(scan);assert.equal(scan.disabled,false);scan.listeners.click();assert.equal(actions.at(-1)[0].action,'music_folder_scan');assert.equal(actions.at(-1)[0].folder,'/synthetic/music');assert.equal(actions.at(-1)[1],undefined);
assert.ok(nodes.some(node=>node.tag==='audio'));assert.ok(nodes.some(node=>node.textContent.includes('无需音乐 API 凭据')));
assert.equal(nodes.some(node=>node.tag==='button'&&node.textContent.includes('一首新音乐')),false);
assert.equal(nodes.some(node=>node.tag==='button'&&node.textContent.includes('本集后期')),false);
assert.equal(nodes.some(node=>node.tag==='textarea'),false);
const approve=nodes.find(node=>node.tag==='button'&&node.textContent==='确认这首音乐');assert.equal(approve.disabled,false);approve.listeners.click();assert.equal(actions.at(-1)[0].action,'music_approve');assert.equal(actions.at(-1)[0].sha256,'a'.repeat(64));assert.equal(actions.at(-1)[1],false);
assert.ok(nodes.some(node=>node.tag==='details'&&node.className==='music-source'));assert.ok(nodes.some(node=>node.textContent==='Synthetic attribution'));
assert.ok(nodes.some(node=>node.tag==='a'&&node.href==='/media/exports/E1.credits.txt'));
state.dashboard.running=true;nodes=walk(postproductionPanel(episode));assert.equal(nodes.find(node=>node.tag==='button'&&node.textContent==='扫描音乐文件夹').disabled,true);
`,context);
'''
        result = subprocess.run([shutil.which("node"), "-e", program], input=json.dumps(sources), text=True,
                                capture_output=True, timeout=15)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("Boolean(data.dashboard?.running),data.music_library,", html)
        self.assertIn('input[data-music-folder]', html)


if __name__ == "__main__":
    unittest.main()
