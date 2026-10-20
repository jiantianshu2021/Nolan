"""Synthetic direct prompt editing checks; never submit a real model request."""
import copy
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import unittest
from unittest.mock import patch

from norlan import core, dashboard
from tests import test_dashboard_prompt_gate as prompt_gate_fixture


class DashboardManualPromptHttpTests(unittest.TestCase):
    setUp_base = prompt_gate_fixture.DashboardPromptGateTests.setUp
    tearDown = prompt_gate_fixture.DashboardPromptGateTests.tearDown
    request = prompt_gate_fixture.DashboardPromptGateTests.request

    def setUp(self):
        self.setUp_base()
        self.core.GateError = core.GateError
        self.edits = []
        self.core.edit_prompt_and_approve = lambda *args: self.edits.append(args)

    def payload(self):
        return {"action": "edit_prompt_and_generate", "id": "S1", "candidate_id": "candidate-one",
                "request_hash": "a" * 64, "prompt": "  完整提示词\n保留换行、空格和最后空行。\n\n"}

    def submit(self, payload=None, headers=None, path="/api/prompt-edit"):
        return self.request("POST", path, self.payload() if payload is None else payload, headers)

    def test_exact_edited_text_dispatches_once_without_starting_production_in_handler(self):
        before = copy.deepcopy(self.state)
        status, _ = self.submit()
        self.assertEqual(status, 200)
        self.assertEqual(self.edits, [(self.root.resolve(), "S1", "candidate-one", "a" * 64, self.payload()["prompt"])])
        self.assertEqual(self.core.calls, [], "No reject/rewrite endpoint must be used")
        self.run.assert_not_called()
        self.assertEqual(self.state, before)

    def test_large_complete_prompt_has_bounded_dedicated_endpoint(self):
        payload = {**self.payload(), "prompt": "完整中文提示词，保留镜头细节。\n" * 1500}
        self.assertGreater(len(json.dumps(payload).encode()), dashboard.MAX_BODY)
        self.assertEqual(self.submit(payload)[0], 200)
        self.assertEqual(self.edits[0][-1], payload["prompt"])
        self.assertEqual(self.submit(payload, path="/api/action")[0], 413,
                         "Other action payloads retain their original body bound")
        self.assertEqual(self.submit(headers={"Content-Length": str(dashboard.MAX_PROMPT_EDIT_BODY + 1)})[0], 413)
        self.assertEqual(len(self.edits), 1)

    def test_only_exact_fields_valid_ids_hash_and_nonblank_bounded_text_are_accepted(self):
        invalid = [{"extra": True}, {"id": True}, {"id": ""}, {"id": "x" * 129},
                   {"candidate_id": ""}, {"candidate_id": False}, {"candidate_id": "x" * 129},
                   {"request_hash": "broken"}, {"request_hash": True}, {"prompt": " \n "},
                   {"prompt": True}, {"prompt": None}, {"prompt": "x\x00y"}, {"prompt": "x" * 100001}]
        for changed in invalid:
            with self.subTest(changed=next(iter(changed))):
                self.assertEqual(self.submit({**self.payload(), **changed})[0], 409)
        for key in self.payload():
            with self.subTest(missing=key):
                self.assertEqual(self.submit({k: v for k, v in self.payload().items() if k != key})[0], 409)
        self.assertEqual(self.edits, [])

    def test_csrf_host_origin_and_endpoint_action_are_checked(self):
        for headers in ({"X-Norlan-Token": "bad"}, {"Origin": "https://foreign.invalid"},
                        {"Host": "foreign.invalid"}):
            self.assertEqual(self.submit(headers=headers)[0], 403)
        for action in ("run", "reject_prompt", "approve_prompt", "edit_postproduction"):
            self.assertEqual(self.submit({**self.payload(), "action": action})[0], 409)
        self.assertEqual(self.edits, [])

    def test_dashboard_and_external_production_locks_block_edits(self):
        self.server.context.running = True
        self.assertEqual(self.submit()[0], 409)
        self.server.context.running = False
        with patch.object(self.server.context, "external_run_active", return_value=True):
            self.assertEqual(self.submit()[0], 409)
        self.assertEqual(self.edits, [])

    def test_stale_candidate_gate_is_a_clear_conflict_without_production(self):
        self.core.edit_prompt_and_approve = lambda *args: (_ for _ in ()).throw(core.GateError("提示词候选已过期，请重新准备。"))
        status, result = self.submit()
        self.assertEqual(status, 409)
        self.assertIn("提示词候选已过期", result["error"])
        self.run.assert_not_called()


HTML = Path(dashboard.__file__).with_name("dashboard.html").read_text()


def function_source(name):
    start = re.search(r"(?:async )?function " + re.escape(name) + r"\(", HTML).start()
    following = re.search(r"\n(?:async )?function ", HTML[start + 1:])
    return HTML[start:start + 1 + following.start()] if following else HTML[start:]


@unittest.skipUnless(shutil.which("node"), "Node.js required for synthetic DOM checks")
class DashboardManualPromptDomTests(unittest.TestCase):
    def run_dom(self, assertions):
        sources = [function_source(name) for name in ("el", "showRevision", "submitRevision", "promptApproval",
                                                     "perform", "action", "appendPrompt", "subshotPlan", "shotCard")]
        program = r'''
const assert=require('node:assert/strict'),vm=require('node:vm');
const {sources,assertions}=JSON.parse(require('node:fs').readFileSync(0,'utf8'));
class Element{
 constructor(tag='div'){this.tag=tag;this.children=[];this.dataset={};this.listeners={};this.textContent='';this.value='';}
 append(...nodes){this.children.push(...nodes);}
 addEventListener(key,handler){this.listeners[key]=handler;}
 setAttribute(key,value){this[key]=value;}
 showModal(){this.shown=true;}close(){this.shown=false;}focus(){this.focused=true;}
}
const nodes=new Map(),calls=[],messages=[];
const $=id=>{if(!nodes.has(id))nodes.set(id,new Element());return nodes.get(id);};
const shot={id:'synthetic-s1',version:1,status:'awaiting_prompt_approval',duration:10,
 prompt_candidate:{id:'candidate-2',request_hash:'a'.repeat(64),prompt:'  原始完整提示词\n镜头：1。\n对白：你好。\n\n'}};
const context={assert,$,nodes,calls,messages,shot,setTimeout,currentAsset:null,requestBusy:false,TOKEN:'synthetic-token',
 state:{assets:[],paused:false,episodes:[{id:'synthetic-e1',shots:[shot]}],dashboard:{running:false}},
 document:{createElement:tag=>new Element(tag)},toast:(...args)=>messages.push(args),
 updateInputControls:()=>{},paint:()=>{},refresh:async()=>{},failureDetails:()=>null,
 advisoryReport:()=>null,shotReview:()=>new Element(),mediaURL:value=>'/media/'+value,
 fetch:async(url,options)=>{const payload=JSON.parse(options.body);calls.push({url,payload,options});
  if(context.fail||(context.failRun&&payload.action==='run'))return{ok:false,json:async()=>({error:'合成测试：候选已变更，请刷新。'})};
  return{ok:true,json:async()=>context.state};},
 walk:node=>[node,...(node.children||[]).flatMap(context.walk)]};
vm.runInNewContext(sources.join('\n')+'\n(async()=>{'+assertions+'\n})().catch(error=>{throw error;});',context);
'''
        result = subprocess.run([shutil.which("node"), "-e", program],
                                input=json.dumps({"sources": sources, "assertions": assertions}),
                                text=True, capture_output=True, timeout=15)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_edit_prefills_complete_candidate_without_submitting_and_uses_large_editor(self):
        self.assertIn('dialog.prompt-editor{max-width:1040px', HTML)
        self.assertIn('dialog.prompt-editor textarea{min-height:clamp(260px,52dvh,680px)', HTML)
        self.run_dom(r'''
const content=walk(promptApproval(shot));
content.find(node=>node.textContent==='修改提示词').listeners.click();
assert.equal($('asset-feedback').value,shot.prompt_candidate.prompt);
assert.equal($('asset-dialog').className,'prompt-editor');assert.equal($('asset-dialog').shown,true);
assert.equal($('asset-feedback').ariaLabel,'完整提示词');
assert.equal($('dialog-title').textContent,'编辑完整提示词');
assert.ok($('dialog-note').textContent.includes('不再由 GPT 重写'));
assert.ok($('dialog-note').textContent.includes('当前片段时长和已批准参考素材'));
assert.equal($('dialog-submit').textContent,'保存提示词并生成');assert.equal(calls.length,0);
''')

    def test_submit_preserves_exact_user_edit_and_continues_once_without_rewrite_or_second_gate(self):
        self.run_dom(r'''
walk(promptApproval(shot)).find(node=>node.textContent==='修改提示词').listeners.click();
const edited='  人工完整提示词\n保留我写的空格。\n\n';$('asset-feedback').value=edited;
submitRevision();await new Promise(resolve=>setTimeout(resolve,0));
assert.equal(calls.length,2);assert.equal(calls[0].url,'/api/prompt-edit');
assert.equal(calls[0].payload.action,'edit_prompt_and_generate');assert.equal(calls[0].payload.prompt,edited);
assert.equal(calls[0].payload.candidate_id,shot.prompt_candidate.id);
assert.equal(calls[0].payload.request_hash,shot.prompt_candidate.request_hash);
assert.deepEqual(Object.keys(calls[0].payload).sort(),['action','candidate_id','id','prompt','request_hash']);
assert.equal(calls[0].options.headers['X-Norlan-Token'],'synthetic-token');
assert.equal(calls[1].url,'/api/action');assert.equal(JSON.stringify(calls[1].payload),JSON.stringify({action:'run'}));
assert.equal(calls.some(call=>['reject_prompt','approve_prompt','planning_feedback'].includes(call.payload.action)),false);
assert.equal($('asset-dialog').shown,false);
''')

    def test_blank_or_busy_submission_never_sends_and_existing_approval_remains_single_action(self):
        self.run_dom(r'''
const all=walk(promptApproval(shot));all.find(node=>node.textContent==='修改提示词').listeners.click();
$('asset-feedback').value=' \n ';submitRevision();assert.equal(calls.length,0);assert.equal($('asset-dialog').shown,true);
$('asset-feedback').value='完整提示词';requestBusy=true;submitRevision();assert.equal(calls.length,0);
requestBusy=false;all.find(node=>node.textContent==='确认提示词并生成').listeners.click();
await new Promise(resolve=>setTimeout(resolve,0));assert.equal(calls[0].payload.action,'approve_prompt');
assert.equal(calls[1].payload.action,'run');assert.equal(calls.length,2);
''')

    def test_saved_video_feedback_keeps_existing_gpt_revision_route_and_small_dialog(self):
        self.run_dom(r'''
walk(promptApproval(shot)).find(node=>node.textContent==='修改提示词').listeners.click();
showRevision({...shot,file:'media/synthetic-v1.mp4'},'shot','返修片段');
assert.equal($('asset-dialog').className,'');assert.equal($('asset-feedback').value,'');
assert.equal($('asset-feedback').ariaLabel,'修改意见');
assert.ok($('dialog-note').textContent.includes('GPT 会判断'));
$('asset-feedback').value='  增加一个镜头，先确认新声音资产。  ';submitRevision();
await new Promise(resolve=>setTimeout(resolve,0));assert.equal(calls[0].url,'/api/action');
assert.equal(calls[0].payload.action,'reject');assert.equal(calls[0].payload.kind,'shot');
assert.equal(calls[0].payload.feedback,'增加一个镜头，先确认新声音资产。');assert.equal(calls[1].payload.action,'run');
''')

    def test_failed_save_does_not_launch_generation(self):
        self.run_dom(r'''
walk(promptApproval(shot)).find(node=>node.textContent==='修改提示词').listeners.click();
const draft='  不应丢失的人工修改\n保留末尾换行。\n\n';$('asset-feedback').value=draft;
fail=true;submitRevision();await new Promise(resolve=>setTimeout(resolve,0));
assert.equal(calls.length,1);assert.equal(calls[0].payload.action,'edit_prompt_and_generate');
assert.ok(messages.some(message=>message[1]===true));assert.equal(requestBusy,false);
assert.equal($('asset-dialog').shown,true,'failed save keeps the editor open, without a resubmission');
assert.equal($('asset-feedback').value,draft,'refresh does not replace edits with the old candidate');
''')


    def test_generated_manual_prompt_shows_actual_text_duration_and_old_storyboard_only_as_comparison(self):
        self.run_dom(r'''
const manual={id:'manual-s1',status:'generating',version:4,duration:10,manual_prompt:true,
 prompt:'  人工完整定稿：两个镜头，先快速下坠，再切父亲表情。\n',
 storyboard_revision:{summary:'过期分镜草稿'},
 subshots:[{start:0,end:3,prompt:'原计划镜头一'},{start:3,end:6,prompt:'原计划镜头二'},
           {start:6,end:10,prompt:'原计划镜头三'}]};
const all=walk(shotCard(manual,0)),text=all.map(node=>node.textContent).join(' · ');
assert.ok(all.some(node=>node.textContent==='手动定稿 · 按原文提交'));
assert.ok(all.some(node=>node.textContent==='视频生成时长 10 秒 · 一次生成'));
assert.ok(all.some(node=>node.textContent===manual.prompt),'the actual submitted text remains available');
assert.ok(all.some(node=>node.textContent==='查看提示词'));
const comparison=all.find(node=>node.className==='subshot-plan original-subshot-plan');
assert.ok(comparison);assert.equal(comparison.tag,'details');assert.equal(Boolean(comparison.open),false);
assert.ok(walk(comparison).some(node=>node.textContent==='原分镜（供对照）'));
assert.ok(text.includes('以手动定稿和实际视频为准'));
assert.equal(text.includes('3 个镜头 · 一次生成'),false,'old cuts are not claimed as actual manual cuts');
assert.equal(text.includes('这是生成时的镜头安排'),false);
assert.equal(text.includes('原始片段说明（供对照）'),false,'manual text is never hidden as an old draft');
assert.equal(calls.length,0);
''')

    def test_successful_manual_approval_then_run_failure_keeps_editor_closed_without_resubmission(self):
        self.run_dom(r'''
walk(promptApproval(shot)).find(node=>node.textContent==='修改提示词').listeners.click();
$('asset-feedback').value='已批准的人工完整提示词。';failRun=true;submitRevision();
await new Promise(resolve=>setTimeout(resolve,0));
assert.equal(calls.length,2);assert.equal(calls[0].payload.action,'edit_prompt_and_generate');
assert.equal(calls[1].payload.action,'run');assert.ok(messages.some(message=>message[1]===true));
assert.equal($('asset-dialog').shown,false,'a saved manual approval is not an obsolete draft');
assert.equal(requestBusy,false);await new Promise(resolve=>setTimeout(resolve,0));
assert.equal(calls.length,2,'a run error never resubmits or reopens the old candidate');
''')


class ManualPromptLatestFeedbackTests(unittest.TestCase):
    def test_manual_edit_event_replaces_older_feedback_without_exposing_prompt(self):
        episode = {"id": "E1", "feedback": "过时整集意见", "shots": [{"id": "S1", "feedback": "过时片段意见"}]}
        state = {"events": [
            {"action": "human_rejected", "kind": "shot", "id": "S1", "feedback": "旧意见"},
            {"action": "human_edited_prompt", "id": "S1", "version": 4,
             "candidate_id": "PRIVATE-ID", "request_hash": "a" * 64},
            {"action": "human_edited_prompt", "id": "OTHER-EPISODE", "version": 99}]}
        before = copy.deepcopy(state)
        self.assertEqual(dashboard._public_latest_episode_feedback(state, episode),
                         "已手动编辑完整提示词并确认直接生成（第4版），以本版实际提交内容为准。")
        self.assertEqual(state, before)
        state["events"].append({"action": "human_rejected", "kind": "shot", "id": "S1", "feedback": "更新意见"})
        self.assertEqual(dashboard._public_latest_episode_feedback(state, episode), "更新意见")


class ManualPromptProjectionTests(unittest.TestCase):
    def state(self, status="generating", version=4):
        prompt = "  实际手动提示词：保留原文。\n\n"
        shot = {"id": "S1", "status": status, "version": version, "duration": 10,
                "prompt": "旧的 GPT 来源稿", "prompt_history": [{"internal": "PRIVATE-HISTORY"}],
                "prompt_candidate": {"id": "manual-four", "status": "approved", "version": 4,
                    "request_hash": "a" * 64, "prompt": prompt,
                    "manual_prompt": {"mode": "exact_text", "source_candidate_id": "PRIVATE-SOURCE-ID",
                        "source_prompt_sha256": "PRIVATE-SOURCE-HASH",
                        "edited_prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest()}},
                "prompt_approval": {"candidate_id": "manual-four", "request_hash": "a" * 64,
                                    "source": "human_manual_edit", "decision": "generate"}}
        return {"id": "synthetic", "status": "ready", "plan": {"status": "approved"},
                "budget": {}, "assets": [], "jobs": {}, "episodes": [{"id": "E1", "shots": [shot]}]}

    def test_current_manual_text_is_projected_before_during_and_after_generation_without_private_marker(self):
        for status, version in (("prompt_approved", 3), ("generating", 4),
                                ("review_required", 4), ("accepted", 4)):
            with self.subTest(status=status):
                state = self.state(status, version)
                before = copy.deepcopy(state)
                shot = state["episodes"][0]["shots"][0]
                public = dashboard.public_state(state)["episodes"][0]["shots"][0]
                self.assertIs(public["manual_prompt"], True)
                self.assertEqual(public["prompt"], shot["prompt_candidate"]["prompt"])
                self.assertEqual(public["duration"], 10)
                self.assertNotIn("PRIVATE-", json.dumps(public))
                self.assertNotIn("edited_prompt_sha256", json.dumps(public))
                self.assertEqual(state, before)

    def test_stale_unbound_or_tampered_manual_text_is_never_exposed_as_current(self):
        mutations = [lambda shot: shot["prompt_candidate"].update(version=3),
                     lambda shot: shot["prompt_candidate"].update(version=True),
                     lambda shot: shot["prompt_candidate"].update(status="candidate"),
                     lambda shot: shot["prompt_candidate"].update(prompt="被篡改的文本"),
                     lambda shot: shot["prompt_candidate"]["manual_prompt"].update(edited_prompt_sha256="b" * 64),
                     lambda shot: shot["prompt_candidate"]["manual_prompt"].update(mode="other"),
                     lambda shot: shot["prompt_approval"].update(candidate_id="old-manual"),
                     lambda shot: shot["prompt_approval"].update(request_hash="b" * 64),
                     lambda shot: shot["prompt_approval"].update(source="automatic"),
                     lambda shot: shot.pop("prompt_approval")]
        for mutate in mutations:
            with self.subTest(mutation=mutate):
                state = self.state()
                shot = state["episodes"][0]["shots"][0]
                mutate(shot)
                public = dashboard.public_state(state)["episodes"][0]["shots"][0]
                self.assertNotIn("manual_prompt", public)
                self.assertEqual(public["prompt"], "旧的 GPT 来源稿")
        state = self.state("prompt_approved", 4)
        self.assertIsNone(dashboard._public_manual_prompt(state["episodes"][0]["shots"][0]),
                          "A previous version is not a new manual authorization")


if __name__ == "__main__":
    unittest.main()
