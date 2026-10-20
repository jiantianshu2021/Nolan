"""Synthetic direct-edit UI and HTTP contracts; no recognition or paid calls."""
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

from norlan import core, dashboard, subtitle_sync
from tests.test_dashboard_postproduction import synthetic_state


class PostEditHttpTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name).resolve()
        self.state = synthetic_state()
        self.core = self.enterContext(patch.object(dashboard, "_core")).return_value
        self.core.load_project.return_value = self.state
        self.apply = self.enterContext(patch("norlan.post_edits.apply_edits"))
        self.launch = self.enterContext(patch.object(dashboard._Context, "launch", return_value=True))
        self.server = dashboard.make_server(self.root, port=0)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.addCleanup(self.shutdown)

    def shutdown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)

    def request(self, data, *, token=True, path="/api/post-edit"):
        connection = http.client.HTTPConnection("127.0.0.1", self.server.server_address[1], timeout=3)
        headers = {"Content-Type": "application/json"}
        if token:
            headers["X-Norlan-Token"] = self.server.context.token
        connection.request("POST", path, json.dumps(data), headers)
        response = connection.getresponse()
        result = response.status, json.loads(response.read())
        connection.close()
        return result

    def payload(self, section="music"):
        row = {"index": 0, "start": 0.5, "end": 2}
        row.update({"gain_db": -22.5} if section == "music" else {"text": "合成测试字幕"})
        return {"action": "edit_postproduction", "id": "E1", "version": 2, "revision": 3,
                "section": section, "edits": [row]}

    def test_full_rows_bound_to_version_and_revision_launch_one_local_preview(self):
        for section in ("music", "subtitles"):
            with self.subTest(section=section):
                payload = self.payload(section)
                status, state = self.request(payload)
                self.assertEqual(status, 200)
                self.apply.assert_called_with(self.root, "E1", 2, 3, section, payload["edits"])
                self.assertEqual(state["id"], self.state["id"])
        self.assertEqual(self.launch.call_count, 2)
        self.core.approve.assert_not_called()
        self.core.reject.assert_not_called()
        self.core.request_postproduction.assert_not_called()

    def test_malformed_fields_arbitrary_media_and_nonfinite_values_never_reach_backend(self):
        changes = [{"version": True}, {"revision": -1}, {"section": []}, {"edits": []},
                   {"file": "/private/video.mp4"}]
        for change in changes:
            self.assertEqual(self.request({**self.payload(), **change})[0], 409)
        bad_rows = [{"track_id": "unapproved"}, {"file": "/private/audio.wav"}, {"source_start": 3},
                    {"index": True}, {"index": 1}, {"start": True}, {"end": 0},
                    {"start": float("nan")}, {"end": float("inf")}, {"gain_db": -61}, {"gain_db": 1}]
        for change in bad_rows:
            payload = self.payload()
            payload["edits"][0].update(change)
            self.assertEqual(self.request(payload)[0], 409)
        for text in ("", "  ", "x\x00y", 123, "字" * 4001):
            payload = self.payload("subtitles")
            payload["edits"][0]["text"] = text
            self.assertEqual(self.request(payload)[0], 409)
        self.apply.assert_not_called()
        self.launch.assert_not_called()

    def test_session_token_and_running_worker_guard_mutations(self):
        self.assertEqual(self.request(self.payload(), token=False)[0], 403)
        self.server.context.running = True
        self.assertEqual(self.request(self.payload())[0], 409)
        self.server.context.running = False
        with patch.object(self.server.context, "external_run_active", return_value=True):
            self.assertEqual(self.request(self.payload())[0], 409)
        self.apply.assert_not_called()
        self.launch.assert_not_called()

    def test_version_conflict_is_visible_and_never_launches(self):
        from norlan.post_edits import EditError
        for error in (EditError("后期版本已变化，请刷新页面后重新调整。"),
                      core.GateError("后期编辑版本已变化，请刷新后重新检查当前条目。"),
                      core.GateError("本集源片段已变化，请先更新后期，再编辑当前版本。")):
            self.apply.side_effect = error
            status, body = self.request(self.payload())
            self.assertEqual(status, 409)
            self.assertEqual(body["error"], str(error))
        self.launch.assert_not_called()

    def test_full_subtitle_payload_has_bounded_separate_size_without_widening_other_actions(self):
        payload = self.payload("subtitles")
        payload["edits"] = [{"index": index, "start": index * 2.0, "end": index * 2.0 + 1,
                             "text": "合成测试用较长字幕" * 20} for index in range(30)]
        self.assertGreater(len(json.dumps(payload)), dashboard.MAX_BODY)
        self.assertEqual(self.request(payload)[0], 200)
        self.assertEqual(self.request({"action": "run", "padding": "x" * dashboard.MAX_BODY}, path="/api/action")[0], 413)

    def test_normal_action_rejects_oversized_declared_body_without_waiting_for_bytes(self):
        connection = http.client.HTTPConnection("127.0.0.1", self.server.server_address[1], timeout=1)
        try:
            connection.putrequest("POST", "/api/action")
            connection.putheader("Content-Type", "application/json")
            connection.putheader("X-Norlan-Token", self.server.context.token)
            connection.putheader("Content-Length", dashboard.MAX_BODY + 1)
            connection.endheaders()
            response = connection.getresponse()
            self.assertEqual(response.status, 413)
            response.read()
        finally:
            connection.close()
        self.apply.assert_not_called()
        self.launch.assert_not_called()

    def test_post_edit_endpoint_rejects_other_actions_and_old_route_cannot_edit(self):
        for action in ("run", "approve", "postproduction", "music_generate", None, []):
            self.assertEqual(self.request({"action": action})[0], 409)
        self.assertEqual(self.request(self.payload(), path="/api/action")[0], 409)
        self.apply.assert_not_called()
        self.launch.assert_not_called()
        self.core.approve.assert_not_called()


class PostEditProjectionTests(unittest.TestCase):
    def test_bound_manual_adoption_enables_human_approval_without_claiming_machine_pass(self):
        state = synthetic_state()
        episode = state["episodes"][0]
        episode.update(status="awaiting_review", version=2, file="exports/E1-v2.mp4", sha256="c" * 64)
        post = episode["postproduction"]
        post.update(status="awaiting_final_approval", final_ready=True, manual_preview=False,
                    final_file=episode["file"], final_version=2, final_sha256=episode["sha256"],
                    source_key="f" * 64, revision=2)
        sub = post["subtitles"]
        sub["audio_sha256"] = "a" * 64
        sub["sync_check"] = subtitle_sync.make_manual_report(sub["segments"],
            audio_sha256=sub["audio_sha256"], source_key=post["source_key"], revision=2, manual_record_hash="d" * 64)
        sub["sync_check"].update(attempts=2, max_attempts=2, checked_cues=1, total_cues=1)
        sub["previous_sync_check"] = {"status": "passed", "summary": "历史合成检查通过", "issues": [],
                                      "evidence_hash": "PRIVATE-EVIDENCE", "raw": "PRIVATE-ASR"}
        public = dashboard.public_state(state)["episodes"][0]["postproduction"]
        self.assertTrue(public["final_available"])
        self.assertTrue(public["final_ready"])
        self.assertEqual(public["subtitles"]["sync_check"]["status"], "manual")
        self.assertNotIn("attempts", public["subtitles"]["sync_check"])
        self.assertNotIn("checked_cues", public["subtitles"]["sync_check"])
        self.assertEqual(public["subtitles"]["previous_sync_check"]["status"], "passed")
        self.assertNotIn("PRIVATE-", json.dumps(public))
        sub["segments"][0]["start"] += 0.1
        stale = dashboard.public_state(state)["episodes"][0]["postproduction"]
        self.assertFalse(stale["final_ready"])
        self.assertEqual(stale["subtitles"]["sync_check"]["status"], "needs_attention")
        self.assertIn("重新保存", stale["subtitles"]["sync_check"]["summary"])

    def test_pending_manual_rows_follow_current_revision_and_source_only(self):
        state = synthetic_state()
        post = state["episodes"][0]["postproduction"]
        post.update(source_key="f" * 64, revision=2, status="pending_manual_edits")
        post["manual_subtitles"] = {"source_key": post["source_key"], "revision": 2,
                                    "segments": [{"start": 1.2, "end": 3.4, "text": "用户已保存的字幕"}]}
        post["manual_music_plan"] = {"source_key": post["source_key"], "revision": 2,
            "summary": "用户已保存的配乐调整", "cues": [{"track_id": "theme-1", "start": 1, "end": 4, "gain_db": -31}]}
        before = copy.deepcopy(state)
        public = dashboard.public_state(state)["episodes"][0]["postproduction"]
        self.assertEqual(public["subtitles"]["segments"], post["manual_subtitles"]["segments"])
        self.assertEqual(public["music"]["cues"][0]["gain_db"], -31)
        self.assertEqual(public["music"]["summary"], "用户已保存的配乐调整")
        self.assertEqual(state, before)
        for change in ({"revision": 1}, {"source_key": "e" * 64}):
            post["manual_subtitles"].update(change)
            public = dashboard.public_state(state)["episodes"][0]["postproduction"]
            self.assertEqual(public["subtitles"]["segments"][0]["text"], "合成对白")

    def test_manual_preview_exposes_only_current_export_without_opening_approval(self):
        state = synthetic_state()
        episode = state["episodes"][0]
        episode.update(status="awaiting_review", version=2, file="exports/E1-v2.mp4", sha256="c" * 64)
        post = episode["postproduction"]
        post.update(status="subtitle_sync_needs_attention", final_ready=True, manual_preview=True,
                    final_file=episode["file"], final_version=2, final_sha256=episode["sha256"], duration=5)
        post["subtitles"]["sync_check"] = {"status": "needs_attention", "summary": "合成测试仍有疑点"}
        public = dashboard.public_state(state)["episodes"][0]["postproduction"]
        self.assertTrue(public["final_available"])
        self.assertTrue(public["manual_preview"])
        self.assertFalse(public["final_ready"])
        self.assertEqual(public["duration"], 5)
        for change in ({"manual_preview": False}, {"final_ready": False}, {"final_version": 1}):
            candidate = copy.deepcopy(state)
            candidate["episodes"][0]["postproduction"].update(change)
            self.assertFalse(dashboard.public_state(candidate)["episodes"][0]["postproduction"]["final_available"])


@unittest.skipUnless(shutil.which("node"), "Node.js required for synthetic DOM contracts")
class PostEditRenderingTests(unittest.TestCase):
    def test_action_router_sends_only_table_edits_to_separate_endpoint(self):
        html = Path(dashboard.__file__).with_name("dashboard.html").read_text()
        match = re.search(r"async function action\([^\n]+", html)
        script = r'''
const assert=require('node:assert/strict'),vm=require('node:vm'),source=JSON.parse(require('node:fs').readFileSync(0,'utf8'));
const calls=[],context={state:null,paint:()=>{},fetch:async(url,options)=>{calls.push({url,payload:JSON.parse(options.body)});return {ok:true,json:async()=>({episodes:[]})};},TOKEN:'synthetic-token',assert,calls};
vm.runInNewContext(source+`\n(async()=>{const edit={action:'edit_postproduction',id:'E1',version:4,revision:2,section:'music',edits:[{index:0,start:0,end:1,gain_db:-20}]};await action(edit);await action({action:'run'});assert.equal(calls[0].url,'/api/post-edit');assert.deepEqual(JSON.stringify(calls[0].payload),JSON.stringify(edit));assert.equal(calls[1].url,'/api/action');})().catch(error=>{process.stderr.write(error.stack);process.exitCode=1;});`,context);
'''
        result = subprocess.run([shutil.which("node"), "-e", script], input=json.dumps(match.group()),
                                text=True, capture_output=True, timeout=10)
        self.assertEqual(result.returncode, 0, result.stderr)

    def run_dom(self, assertions):
        html = Path(dashboard.__file__).with_name("dashboard.html").read_text()
        sources = []
        for name in ("el", "postproductionEditor"):
            match = re.search(r"(?:async )?function " + name + r"\(", html)
            following = re.search(r"\n(?:async )?function ", html[match.start() + 1:])
            end = match.start() + 1 + following.start() if following else len(html)
            sources.append(html[match.start():end])
        program = r'''
const assert=require('node:assert/strict'),vm=require('node:vm');
const {sources,assertions}=JSON.parse(require('node:fs').readFileSync(0,'utf8'));
class Element{constructor(tag){this.tag=tag;this.children=[];this.dataset={};this.listeners={};this.textContent='';this.value='';}append(...nodes){this.children.push(...nodes);}setAttribute(k,v){this[k]=v;}addEventListener(k,v){this.listeners[k]=v;}}
const actions=[],messages=[],state={paused:false,dashboard:{running:false},music_library:{tracks:[{id:'theme',title:'合成测试音乐'}]}};
const context={state,assert,actions,messages,requestBusy:false,document:{createElement:tag=>new Element(tag),createTextNode:text=>({tag:'text',textContent:text})},action:async payload=>actions.push(payload),toast:(text,error)=>messages.push({text,error}),refresh:async()=>{},updateInputControls:()=>{}};
vm.runInNewContext(sources.join('\n')+`
const walk=node=>[node,...(node.children||[]).flatMap(walk)],field=(row,key)=>walk(row).find(node=>node.dataset?.postEditField===key),rows=editor=>walk(editor).filter(node=>node.className==='post-edit-row'),save=editor=>walk(editor).find(node=>node.className==='quiet post-edit-save');
const episode={id:'E1',version:4,postproduction:{ready:true,revision:2,duration:10,music:{summary:'合成配乐安排',cues:[{track_id:'theme',start:0,end:4,gain_db:-18,source_start:1,fade_in:.2,fade_out:.5,reason:'合成说明'},{track_id:'theme',start:5,end:9,gain_db:-23}]},subtitles:{segments:[{start:.5,end:2,text:'合成字幕一'},{start:3,end:4,text:'合成字幕二'}]}}};state.episodes=[episode];
(async()=>{`+assertions+`})().catch(error=>{process.stderr.write(error.stack);process.exitCode=1;});`,context);
'''
        result = subprocess.run([shutil.which("node"), "-e", program], input=json.dumps({"sources": sources, "assertions": assertions}),
                                text=True, capture_output=True, timeout=15)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_music_controls_submit_all_rows_without_media_identity_or_generation(self):
        self.run_dom(r'''
const editor=postproductionEditor(episode,'music'),items=rows(editor),range=field(items[0],'gain_db');
assert.equal(editor.children[0].textContent,'查看本集配乐安排');assert.equal(range.type,'range');assert.equal(range.min,'-60');assert.equal(range.max,'0');
assert.equal(save(editor).disabled,true);range.value='-29.5';range.listeners.input();
assert.ok(walk(items[0]).some(node=>node.tag==='output'&&node.textContent==='3.3% · -29.5 dB'));
field(items[0],'start').value='1.25';field(items[0],'start').listeners.input();
assert.equal(field(items[0],'start').step,'0.01');assert.equal(save(editor).textContent,'确认调整并试看');
await save(editor).listeners.click();assert.equal(actions.length,1);
assert.deepEqual(JSON.parse(JSON.stringify(actions[0])),{action:'edit_postproduction',id:'E1',version:4,revision:2,section:'music',edits:[{index:0,start:1.25,end:4,gain_db:-29.5},{index:1,start:5,end:9,gain_db:-23}]});
assert.equal(actions.some(item=>item.action==='approve'||item.action==='run'),false);
''')

    def test_subtitle_text_and_times_submit_full_rows(self):
        self.run_dom(r'''
const editor=postproductionEditor(episode,'subtitles'),items=rows(editor),text=field(items[0],'text');
text.value='已更正的合成字幕';text.listeners.input();field(items[0],'end').value='2.25';field(items[0],'end').listeners.input();
await save(editor).listeners.click();assert.equal(actions[0].section,'subtitles');assert.equal(actions[0].edits.length,2);
assert.deepEqual(JSON.parse(JSON.stringify(actions[0].edits[0])),{index:0,start:.5,end:2.25,text:'已更正的合成字幕'});
assert.ok(walk(editor).some(node=>node.textContent.includes('不再识别、校对或同步复检')));
''')

    def test_music_original_excerpt_range_follows_draft_duration_without_changing_source_identity(self):
        self.run_dom(r'''
const editor=postproductionEditor(episode,'music'),items=rows(editor),original=walk(items[0]).find(node=>node.className==='muted music-source-window');
assert.equal(original.textContent,'原曲截取：00:01.0–00:05.0（共 4.0 秒）');
field(items[0],'end').value='3';field(items[0],'end').listeners.input();assert.equal(original.textContent,'原曲截取：00:01.0–00:04.0（共 3.0 秒）');
assert.ok(walk(editor).some(node=>node.textContent.includes('以统一后的音源为准')));
assert.ok(walk(editor).some(node=>node.textContent.includes('不再自动复检、改音量或调用 GPT')));
await save(editor).listeners.click();assert.equal(actions.length,1);assert.equal(actions[0].edits[0].end,3);assert.equal('source_start' in actions[0].edits[0],false);assert.equal(episode.postproduction.music.cues[0].source_start,1);
''')

    def test_drafts_survive_same_version_refresh_but_reset_on_new_revision_or_export(self):
        self.run_dom(r'''
let editor=postproductionEditor(episode,'music');let range=field(rows(editor)[0],'gain_db');range.value='-32';range.listeners.input();
episode.postproduction.music.cues[0].gain_db=-12;
editor=postproductionEditor(episode,'music');assert.equal(field(rows(editor)[0],'gain_db').value,'-32');assert.equal(save(editor).disabled,false);
episode.postproduction.revision++;editor=postproductionEditor(episode,'music');assert.equal(field(rows(editor)[0],'gain_db').value,'-12');assert.equal(save(editor).disabled,true);
range=field(rows(editor)[0],'gain_db');range.value='-28';range.listeners.input();episode.version++;
editor=postproductionEditor(episode,'music');assert.equal(field(rows(editor)[0],'gain_db').value,'-12');assert.equal(save(editor).disabled,true);
''')

    def test_busy_and_unready_disable_controls_and_prevent_submit(self):
        self.run_dom(r'''
for(const update of ['running','paused','unready']){
 state.dashboard.running=update==='running';state.paused=update==='paused';episode.postproduction.ready=update!=='unready';
 const editor=postproductionEditor(episode,'music');assert.ok(walk(editor).filter(node=>['input','button'].includes(node.tag)).every(node=>node.disabled));
 await save(editor).listeners.click();
}assert.equal(actions.length,0);
''')

    def test_duplicate_click_is_blocked_and_failed_save_retains_draft_and_error(self):
        self.run_dom(r'''
let release;action=payload=>{actions.push(payload);return new Promise(resolve=>release=resolve);};
let editor=postproductionEditor(episode,'music'),range=field(rows(editor)[0],'gain_db');range.value='-31';range.listeners.input();
const first=save(editor).listeners.click();assert.equal(save(editor).disabled,true);await save(editor).listeners.click();assert.equal(actions.length,1);release();await first;assert.equal(requestBusy,false);
action=async()=>{throw new Error('后期版本已变化，请刷新页面后重新调整。');};
await save(editor).listeners.click();assert.ok(messages.some(item=>item.error&&item.text.includes('版本已变化')));
editor=postproductionEditor(episode,'music');assert.equal(field(rows(editor)[0],'gain_db').value,'-31');assert.equal(save(editor).disabled,false);
''')

    def test_invalid_blank_and_reverse_times_do_not_submit(self):
        self.run_dom(r'''
const editor=postproductionEditor(episode,'subtitles'),input=field(rows(editor)[0],'start');input.value='';input.listeners.input();
await save(editor).listeners.click();assert.equal(actions.length,0);assert.ok(messages.at(-1).error);
input.value='3';input.listeners.input();await save(editor).listeners.click();assert.equal(actions.length,0);
''')


if __name__ == "__main__":
    unittest.main()
