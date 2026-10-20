"""Synthetic model selection UI checks; no real model or browser requests."""
import json
from pathlib import Path
import re
import shutil
import subprocess
import unittest

from norlan import dashboard


HTML = Path(dashboard.__file__).with_name("dashboard.html").read_text()


def function_source(name):
    start = re.search(r"(?:async )?function " + re.escape(name) + r"\(", HTML).start()
    following = re.search(r"\n(?:async )?function ", HTML[start + 1:])
    return HTML[start:start + 1 + following.start()] if following else HTML[start:]


@unittest.skipUnless(shutil.which("node"), "Node.js required for synthetic DOM checks")
class DashboardVideoModelDomTests(unittest.TestCase):
    def run_dom(self, assertions):
        sources = [function_source(name) for name in (
            "el", "showRevision", "submitRevision", "promptApproval", "perform", "action",
            "appendPrompt", "subshotPlan", "shotCard", "videoModelChoice", "videoModelField",
            "videoModelName", "updateVideoModelControls")]
        program = r'''
const assert=require('node:assert/strict'),vm=require('node:vm');
const {sources,assertions}=JSON.parse(require('node:fs').readFileSync(0,'utf8'));
class Element{
 constructor(tag='div'){this.tag=tag;this.children=[];this.dataset={};this.listeners={};this.textContent='';this.value='';}
 append(...nodes){this.children.push(...nodes);}
 replaceChildren(...nodes){this.children=[...nodes];}
 addEventListener(key,handler){this.listeners[key]=handler;}
 setAttribute(key,value){this[key]=value;}
 showModal(){this.shown=true;}close(){this.shown=false;}focus(){this.focused=true;}
}
const nodes=new Map(),calls=[],messages=[],roots=[];
const $=id=>{if(!nodes.has(id))nodes.set(id,new Element());return nodes.get(id);};
const walk=node=>[node,...(node.children||[]).flatMap(walk)];
const modelOptions=[{id:'fast',label:'Seedance Fast',available:true,reason:''},
 {id:'pro',label:'Seedance Pro',available:true,reason:''},
 {id:'legacy',label:'旧视频模型',available:false,reason:'不支持当前声音参考'}];
const shot={id:'synthetic-s1',version:1,status:'awaiting_prompt_approval',duration:10,
 prompt_candidate:{id:'candidate-2',request_hash:'a'.repeat(64),video_model:'fast',
  model_options:modelOptions,prompt:'  原始完整提示词\n镜头：1。\n对白：你好。\n\n'}};
const context={assert,$,nodes,calls,messages,shot,modelOptions,roots,walk,setTimeout,currentAsset:null,
 requestBusy:false,TOKEN:'synthetic-token',
 state:{assets:[],paused:false,episodes:[{id:'synthetic-e1',shots:[shot]}],dashboard:{running:false},
  video_models:{default_model:'pro',models:modelOptions}},
 document:{createElement:tag=>new Element(tag),querySelectorAll:selector=>{
  if(selector==='select[data-video-model]')return [...roots,...nodes.values()].flatMap(walk).filter(node=>node.tag==='select'&&node.dataset.videoModel);
  return [];
 }},toast:(...args)=>messages.push(args),updateInputControls:()=>{},paint:()=>{},refresh:async()=>{},
 failureDetails:()=>null,advisoryReport:()=>null,shotReview:()=>new Element(),mediaURL:value=>'/media/'+value,
 fetch:async(url,options)=>{const payload=JSON.parse(options.body);calls.push({url,payload,options});
  if(context.fail)return{ok:false,json:async()=>({error:'合成测试：当前模型不可用'})};
  return{ok:true,json:async()=>context.state};}};
vm.runInNewContext(sources.join('\n')+'\n(async()=>{'+assertions+'\n})().catch(error=>{throw error;});',context);
'''
        result = subprocess.run([shutil.which("node"), "-e", program],
                                input=json.dumps({"sources": sources, "assertions": assertions}),
                                text=True, capture_output=True, timeout=15)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_bound_model_and_incompatible_reasons_are_visible_without_generation(self):
        self.run_dom(r'''
const card=promptApproval(shot);roots.push(card);const all=walk(card),select=all.find(node=>node.tag==='select');
assert.ok(all.some(node=>node.textContent==='本次视频生成模型'));
assert.equal(select.value,'fast','the candidate binding wins over the project default');
const legacy=select.children.find(node=>node.value==='legacy');
assert.equal(legacy.disabled,true);assert.ok(legacy.textContent.includes('不支持当前声音参考'));
assert.equal(select.children.find(node=>node.value==='pro').disabled,false);
select.value='pro';select.listeners.change();
assert.equal(calls.length,0,'changing a local model selection never submits or approves');
assert.equal(shot.prompt_candidate.video_model,'fast');
assert.ok(all.some(node=>node.textContent===shot.prompt_candidate.prompt));
assert.equal(all.filter(node=>node.tag==='button').length,2,'the single full prompt gate is retained');
''')

    def test_ordinary_confirmation_submits_selected_model_with_candidate_hash_once(self):
        self.run_dom(r'''
const card=promptApproval(shot);roots.push(card);const all=walk(card),select=all.find(node=>node.tag==='select');
select.value='pro';select.listeners.change();
all.find(node=>node.textContent==='确认提示词并生成').listeners.click();
await new Promise(resolve=>setTimeout(resolve,0));
assert.equal(calls.length,2);assert.equal(calls[0].url,'/api/action');
assert.equal(calls[0].payload.action,'approve_prompt');assert.equal(calls[0].payload.video_model,'pro');
assert.equal(calls[0].payload.candidate_id,shot.prompt_candidate.id);
assert.equal(calls[0].payload.request_hash,shot.prompt_candidate.request_hash);
assert.equal(calls[1].payload.action,'run');
assert.equal(calls.some(call=>['reject_prompt','planning_feedback'].includes(call.payload.action)),false);
''')

    def test_manual_editor_can_choose_model_and_preserves_exact_prompt_without_rewrite(self):
        self.run_dom(r'''
const card=promptApproval(shot);roots.push(card);const all=walk(card),select=all.find(node=>node.tag==='select');
select.value='pro';select.listeners.change();all.find(node=>node.textContent==='修改提示词').listeners.click();
assert.equal($('asset-feedback').value,shot.prompt_candidate.prompt);
const editor=walk($('prompt-model-editor')).find(node=>node.tag==='select');assert.equal(editor.value,'pro');
editor.value='fast';editor.listeners.change();assert.equal(select.value,'fast','the underlying selection follows the editor');
const edited='  人工完整提示词\n保留空格。\n\n';$('asset-feedback').value=edited;submitRevision();
await new Promise(resolve=>setTimeout(resolve,0));
assert.equal(calls.length,2);assert.equal(calls[0].url,'/api/prompt-edit');
assert.equal(calls[0].payload.action,'edit_prompt_and_generate');assert.equal(calls[0].payload.video_model,'fast');
assert.equal(calls[0].payload.prompt,edited);assert.equal(calls[0].payload.candidate_id,shot.prompt_candidate.id);
assert.equal(calls[1].payload.action,'run');assert.equal($('asset-dialog').shown,false);
assert.equal(calls.some(call=>['reject_prompt','planning_feedback','approve_prompt'].includes(call.payload.action)),false);
''')

    def test_refresh_and_failed_save_retain_model_and_human_text(self):
        self.run_dom(r'''
let card=promptApproval(shot);roots.push(card);let all=walk(card),select=all.find(node=>node.tag==='select');
select.value='pro';select.listeners.change();
state.video_models.default_model='fast';card=promptApproval(shot);roots.push(card);all=walk(card);
assert.equal(all.find(node=>node.tag==='select').value,'pro','polling must not replace an unsubmitted model');
all.find(node=>node.textContent==='修改提示词').listeners.click();
const draft='  保留我的人工定稿。\n\n';$('asset-feedback').value=draft;fail=true;submitRevision();
await new Promise(resolve=>setTimeout(resolve,0));
assert.equal(calls.length,1);assert.equal(calls[0].payload.video_model,'pro');
assert.equal($('asset-dialog').shown,true);assert.equal($('asset-feedback').value,draft);
assert.equal(walk($('prompt-model-editor')).find(node=>node.tag==='select').value,'pro');
assert.equal(videoModelChoice(shot,shot.prompt_candidate).value,'pro');
assert.equal(requestBusy,false);assert.ok(messages.some(message=>message[1]===true));
''')

    def test_lost_compatibility_keeps_selection_and_blocks_without_silent_fallback(self):
        self.run_dom(r'''
let all=walk(promptApproval(shot)),select=all.find(node=>node.tag==='select');
select.value='pro';select.listeners.change();
shot.prompt_candidate.model_options=shot.prompt_candidate.model_options.map(item=>item.id==='pro'?{...item,available:false,reason:'当前参考视频不受支持'}:item);
all=walk(promptApproval(shot));select=all.find(node=>node.tag==='select');
assert.equal(select.value,'pro');assert.equal(all.find(node=>node.textContent==='确认提示词并生成').disabled,true);
assert.ok(all.some(node=>node.textContent.includes('当前参考视频不受支持')));
all.find(node=>node.textContent==='修改提示词').listeners.click();assert.equal($('dialog-submit').disabled,true);
$('asset-feedback').value='人工稿';submitRevision();assert.equal(calls.length,0);
const editor=walk($('prompt-model-editor')).find(node=>node.tag==='select');editor.value='fast';editor.listeners.change();
assert.equal($('dialog-submit').disabled,false);assert.equal(currentAsset.selected_video_model,'fast');
''')

    def test_new_candidate_does_not_inherit_an_old_unsubmitted_selection(self):
        self.run_dom(r'''
let all=walk(promptApproval(shot)),select=all.find(node=>node.tag==='select');
select.value='pro';select.listeners.change();
shot.prompt_candidate={...shot.prompt_candidate,id:'candidate-3',request_hash:'b'.repeat(64),video_model:'fast'};
assert.equal(walk(promptApproval(shot)).find(node=>node.tag==='select').value,'fast');
shot.prompt_candidate={...shot.prompt_candidate,request_hash:'c'.repeat(64),video_model:'pro'};
assert.equal(walk(promptApproval(shot)).find(node=>node.tag==='select').value,'pro');
assert.equal(calls.length,0);
''')

    def test_current_and_historical_videos_show_their_actual_models(self):
        self.run_dom(r'''
const saved={id:'saved',status:'accepted',version:3,duration:10,file:'media/v3.mp4',prompt:'实际提示词',
 generation_model:'fast',history:[{version:1,file:'media/v1.mp4',generation_model:'legacy'},
 {version:2,file:'media/v2.mp4'}]};
const all=walk(shotCard(saved,0)),modelLines=all.filter(node=>node.className==='muted generation-model').map(node=>node.textContent);
assert.equal(modelLines.length,2,'unknown historical models must not be fabricated from current settings');
assert.ok(modelLines.some(text=>text==='本版生成模型：Seedance Fast · fast'));
assert.ok(modelLines.some(text=>text==='生成模型：旧视频模型 · legacy'));
assert.equal(modelLines.some(text=>text.includes('Seedance Pro')),false);
assert.equal(calls.length,0);
''')

    def test_busy_pause_and_running_disable_the_selector_without_changing_selection(self):
        self.run_dom(r'''
const card=promptApproval(shot);roots.push(card);const select=walk(card).find(node=>node.tag==='select');
for(const flag of ['busy','paused','running']){
 requestBusy=flag==='busy';state.paused=flag==='paused';state.dashboard.running=flag==='running';
 updateVideoModelControls();assert.equal(select.disabled,true);assert.equal(select.value,'fast');
}
requestBusy=false;state.paused=false;state.dashboard.running=false;updateVideoModelControls();
assert.equal(select.disabled,false);assert.equal(calls.length,0);
''')

    def test_catalog_changes_repaint_and_ordinary_revision_hides_prompt_controls(self):
        self.assertIn("data.music_library,data.video_models,data.asset_amendments", function_source("paint"))
        self.assertIn("dialog:not(.prompt-editor) #prompt-model-editor{display:none}", HTML)
        self.run_dom(r'''
const all=walk(promptApproval(shot));all.find(node=>node.textContent==='修改提示词').listeners.click();
$('dialog-submit').disabled=true;showRevision({...shot,file:'media/original.mp4'},'shot','返修片段');
assert.equal($('asset-dialog').className,'');assert.equal($('dialog-submit').disabled,false);
assert.equal($('asset-feedback').value,'');assert.equal(currentAsset.reviewKind,'shot');assert.equal(calls.length,0);
''')


if __name__ == "__main__":
    unittest.main()
