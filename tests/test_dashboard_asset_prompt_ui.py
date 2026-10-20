"""Synthetic DOM checks for asset consent; no real approval or paid calls."""
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
class DashboardAssetPromptDomTests(unittest.TestCase):
    def run_dom(self, assertions):
        sources = [function_source(name) for name in (
            "el", "mediaURL", "previewInfo", "assetPromptDraft", "assetPromptApproval", "updateAssetPromptControls",
            "nextHumanNavigationGate", "paintApprovalGate", "perform", "action", "assetEditSession",
            "openAssetPromptEditor", "submitAssetPromptEdit", "assetDirectEditor", "assetCard",
            "inputDraft", "inputCard", "paintIntake")]
        program = r'''
const assert=require('node:assert/strict'),vm=require('node:vm');
const {sources,assertions}=JSON.parse(require('node:fs').readFileSync(0,'utf8'));
class Element{
 constructor(tag='div'){this.tag=tag;this.children=[];this.dataset={};this.listeners={};this.textContent='';this.value='';this.classList={add:()=>{},remove:()=>{},toggle:()=>{}};}
 append(...nodes){this.children.push(...nodes);}
 replaceChildren(...nodes){this.children=[...nodes];}
 addEventListener(key,handler){this.listeners[key]=handler;}
 setAttribute(key,value){this[key]=value;}
 focus(){this.focused=true;}
 querySelector(selector){return walk(this).find(node=>node.className===selector.slice(1));}
}
const nodes=new Map(),calls=[],messages=[],roots=[],previews=[];
const $=id=>{if(!nodes.has(id))nodes.set(id,new Element());return nodes.get(id);};
const walk=node=>[node,...(node.children||[]).flatMap(walk)];
const asset={id:'synthetic-hero',kind:'character',name:'合成人物',version:1,status:'awaiting_asset_prompt_approval',
 generation_prompt:{id:'synthetic-p2',sha256:'a'.repeat(64),request_hash:'a'.repeat(64),model:'image-fast',
 prompt:'  完整人物提示词\n白衬衫，黑色短发。\n\n',target_version:2,reference_manifest:{images:['inputs/hero/synthetic-reference.png'],audio:[]}},
 generation_models:[{id:'image-fast',label:'图片 Fast',available:true},{id:'image-pro',label:'图片 Pro',available:true},
 {id:'legacy',label:'旧模型',available:false,reason:'不支持参考图'}],selected_generation_model:'image-fast',
 creation_input:{references:[{name:'人物参考',file:'inputs/hero/synthetic-reference.png'}]}};
const context={assert,$,nodes,calls,messages,asset,roots,previews,walk,setTimeout,currentAsset:null,previewReady:false,previewAsset:null,showImage:asset=>previews.push(asset),
 requestBusy:false,TOKEN:'synthetic-token',signature:'',inputDrafts:new Map(),kinds:{character:'人物',voice:'声音'},
 badge:status=>new Element('span'),appendPrompt:()=>{},failureDetails:()=>null,uploadReferences:()=>{},removeReference:()=>{},saveInput:()=>{},
 state:{id:'synthetic-project',assets:[asset],paused:false,episodes:[],plan:{status:'approved'},dashboard:{running:false}},
 document:{createElement:tag=>new Element(tag),createTextNode:text=>({tag:"text",textContent:text}),querySelectorAll:selector=>selector==='[data-asset-prompt-control]'?
 roots.flatMap(walk).filter(node=>node.dataset.assetPromptControl):[]},
 toast:(...args)=>messages.push(args),updateInputControls:()=>{},paint:()=>{},refresh:async()=>{},
 fetch:async(url,options)=>{calls.push({url,headers:options.headers,payload:options.body?JSON.parse(options.body):null});
  if(!options.body)return{ok:!context.readFail,json:async()=>context.readFail?({error:"合成只读检查失败"}):context.snapshot};
  if(context.fail)return{ok:false,json:async()=>({error:'合成测试：候选已经更新'})};
  return{ok:true,json:async()=>context.state};}};
vm.runInNewContext(sources.join('\n')+'\n(async()=>{'+assertions+'\n})().catch(error=>{throw error;});',context);
'''
        result = subprocess.run([shutil.which("node"), "-e", program],
                                input=json.dumps({"sources": sources, "assertions": assertions}),
                                text=True, capture_output=True, timeout=15)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_complete_editable_prompt_and_compatible_models_are_shown_without_submission(self):
        self.run_dom(r'''
const all=walk(assetPromptApproval(asset)),editor=all.find(node=>node.tag==='textarea'),select=all.find(node=>node.tag==='select');
assert.equal(editor.value,asset.generation_prompt.prompt);assert.equal(select.value,'image-fast');
assert.ok(all.some(node=>node.textContent==='资产生成前确认 · 第 2 版'));
assert.ok(all.some(node=>node.textContent==='本次资产生成模型'));
assert.ok(all.some(node=>node.textContent==='将使用的参考图片 · 1 张'));
assert.equal(all.find(node=>node.tag==='textarea').maxLength,100000);
assert.equal(all.find(node=>node.tag==='img').src,'/media/inputs/hero/synthetic-reference.png');
assert.equal(select.children.find(node=>node.value==='legacy').disabled,true);
assert.ok(select.children.find(node=>node.value==='legacy').textContent.includes('不支持参考图'));
select.value='image-pro';select.listeners.change();assert.equal(calls.length,0);
assert.equal(asset.selected_generation_model,'image-fast');assert.equal(asset.generation_prompt.model,'image-fast');
''')

    def test_one_confirmation_submits_bound_candidate_and_selected_model_only_once(self):
        self.run_dom(r'''
const card=assetPromptApproval(asset);roots.push(card);const all=walk(card),select=all.find(node=>node.tag==='select');
select.value='image-pro';select.listeners.change();all.find(node=>node.textContent==='确认提示词并生成资产').listeners.click();
await new Promise(resolve=>setTimeout(resolve,0));
assert.equal(calls.length,1);assert.equal(calls[0].url,'/api/prompt-edit');
assert.equal(JSON.stringify(calls[0].payload),JSON.stringify({action:'asset_prompt_approve',id:asset.id,candidate_id:'synthetic-p2',sha256:'a'.repeat(64),model:'image-pro'}));
assert.equal(calls.some(call=>['run','approve','reject','planning_feedback'].includes(call.payload.action)),false);
''')

    def test_human_prompt_uses_exact_text_without_rewrite_or_second_confirmation(self):
        self.run_dom(r'''
const all=walk(assetPromptApproval(asset)),editor=all.find(node=>node.tag==='textarea');
const edited='  人工修改的完整提示词。\n保留空格和换行。\n\n';editor.value=edited;editor.listeners.input();
assert.ok(all.some(node=>node.textContent==='已修改：确认后直接使用以上原文。'));
all.find(node=>node.textContent==='确认提示词并生成资产').listeners.click();
await new Promise(resolve=>setTimeout(resolve,0));assert.equal(calls.length,1);
assert.equal(calls[0].payload.action,'asset_prompt_approve');assert.equal(calls[0].payload.prompt,edited);
assert.equal(asset.generation_prompt.prompt.startsWith('  完整人物'),true,'the prepared candidate stays immutable');
''')

    def test_polling_and_failed_save_keep_prompt_and_model_drafts(self):
        self.run_dom(r'''
let all=walk(assetPromptApproval(asset)),editor=all.find(node=>node.tag==='textarea'),select=all.find(node=>node.tag==='select');
const edited='  本地未提交草稿。\n\n';editor.value=edited;editor.listeners.input();select.value='image-pro';select.listeners.change();
asset.selected_generation_model='image-fast';all=walk(assetPromptApproval(asset));
assert.equal(all.find(node=>node.tag==='textarea').value,edited);assert.equal(all.find(node=>node.tag==='select').value,'image-pro');
fail=true;all.find(node=>node.textContent==='确认提示词并生成资产').listeners.click();
await new Promise(resolve=>setTimeout(resolve,0));
assert.equal(calls.length,1);assert.equal(assetPromptDraft(asset,asset.generation_prompt).draft.prompt,edited);
assert.equal(assetPromptDraft(asset,asset.generation_prompt).draft.model,'image-pro');
assert.equal(requestBusy,false);assert.ok(messages.some(message=>message[1]===true));
''')

    def test_new_candidate_hash_uses_new_prompt_and_model_while_previous_draft_survives(self):
        self.run_dom(r'''
let all=walk(assetPromptApproval(asset)),editor=all.find(node=>node.tag==='textarea');
editor.value='第一份未提交草稿';editor.listeners.input();const old=asset.generation_prompt;
asset.generation_prompt={...old,request_hash:'b'.repeat(64),sha256:'b'.repeat(64),prompt:'新候选完整提示词'};
asset.selected_generation_model='image-pro';all=walk(assetPromptApproval(asset));
assert.equal(all.find(node=>node.tag==='textarea').value,'新候选完整提示词');assert.equal(all.find(node=>node.tag==='select').value,'image-pro');
assert.equal(assetPromptDraft(asset,old).draft.prompt,'第一份未提交草稿');assert.equal(calls.length,0);
''')

    def test_incompatible_model_stays_selected_and_blocks_without_silent_fallback(self):
        self.run_dom(r'''
let all=walk(assetPromptApproval(asset)),select=all.find(node=>node.tag==='select');select.value='image-pro';select.listeners.change();
asset.generation_models=asset.generation_models.filter(model=>model.id!=='image-pro');
all=walk(assetPromptApproval(asset));select=all.find(node=>node.tag==='select');
assert.equal(select.value,'image-pro');assert.equal(all.find(node=>node.textContent==='确认提示词并生成资产').disabled,true);
all.find(node=>node.textContent==='确认提示词并生成资产').listeners.click();assert.equal(calls.length,0);
select.value='image-fast';select.listeners.change();assert.equal(all.find(node=>node.textContent==='确认提示词并生成资产').disabled,false);
''')

    def test_busy_paused_running_and_empty_prompt_cannot_submit(self):
        self.run_dom(r'''
const card=assetPromptApproval(asset);roots.push(card);const all=walk(card),editor=all.find(node=>node.tag==='textarea'),select=all.find(node=>node.tag==='select'),approve=all.find(node=>node.textContent==='确认提示词并生成资产');
for(const flag of ['busy','paused','running']){
 requestBusy=flag==='busy';state.paused=flag==='paused';state.dashboard.running=flag==='running';
 updateAssetPromptControls();assert.equal(select.disabled,true);assert.equal(editor.disabled,true);assert.equal(approve.disabled,true);approve.listeners.click();
}
requestBusy=false;state.paused=false;state.dashboard.running=false;updateAssetPromptControls();assert.equal(approve.disabled,false);
editor.value='   \n';editor.listeners.input();assert.equal(approve.disabled,true);approve.listeners.click();assert.equal(calls.length,0);
''')

    def test_voice_asset_has_same_prompt_and_model_gate_with_audition_duration(self):
        self.run_dom(r'''
asset.kind='voice';asset.name='合成播报员';asset.generation_prompt.duration=8;
asset.generation_models=[{id:'voice-model',label:'声音模型',available:true}];asset.selected_generation_model='voice-model';
const all=walk(assetPromptApproval(asset));assert.ok(all.some(node=>node.textContent==='本次试听时长：8 秒。'));
assert.equal(all.find(node=>node.tag==='select').value,'voice-model');assert.equal(calls.length,0);
asset.status='candidate';assert.equal(assetPromptApproval(asset),null);
''')

    def test_continue_focuses_asset_prompt_and_sidebar_explains_pre_generation_gate(self):
        self.run_dom(r'''
const gate=nextHumanNavigationGate(state);assert.equal(gate.route,'#assets');assert.equal(gate.target,'asset-prompt-approval-'+asset.id);
assert.equal(gate.label,'前往确认资产提示词与模型');paintApprovalGate(state);
assert.equal($('status').textContent,'等待资产提示词确认');assert.ok($('status-note').textContent.includes('确认后才生成该资产'));
assert.equal($('run').disabled,false);assert.equal(calls.length,0);
state.paused=true;assert.equal(nextHumanNavigationGate(state),null);paintApprovalGate(state);assert.equal($('run').textContent,'恢复制作');
''')

    def test_actual_voice_character_reference_thumbnails_open_without_submitting(self):
        self.run_dom(r'''
asset.kind='voice';asset.generation_prompt.reference_manifest={images:['media/hero-approved.png'],audio:[]};asset.creation_input.references=[];
const all=walk(assetPromptApproval(asset)),image=all.find(node=>node.tag==='img'),open=all.find(node=>node.tag==='button'&&node.className==='asset-image-button');
assert.equal(image.src,'/media/media/hero-approved.png');assert.ok(all.some(node=>node.textContent==='将使用的参考图片 · 1 张'));
assert.equal(all.some(node=>node.textContent.includes('hero-approved.png')),false,'raw paths stay out of visible copy');
open.listeners.click();assert.equal(previews.length,1);assert.equal(previews[0].file,'media/hero-approved.png');
assert.equal(previews[0].reference,true);assert.equal(previews[0].sourceAssetId,asset.id);assert.equal(previews[0].candidateId,asset.generation_prompt.id);assert.equal(calls.length,0);
previewAsset=previews[0];previewInfo();assert.equal($('image-info').textContent,'本次生成参考图');assert.equal($('image-source').hidden,false);
asset.generation_prompt={...asset.generation_prompt,id:'another-candidate'};previewInfo();assert.ok($('image-info').textContent.includes('当前参考清单已更新'));assert.equal($('image-source').hidden,true);
''')

    def test_empty_voice_manifest_is_authoritative_and_explains_offscreen_voice_without_stale_images(self):
        self.run_dom(r'''
asset.kind='voice';asset.generation_prompt.reference_manifest={images:[],audio:[]};
asset.creation_input.references=[{file:'inputs/old-stale-reference.png'}];
const all=walk(assetPromptApproval(asset));assert.equal(all.some(node=>node.tag==='img'),false);
assert.ok(all.some(node=>node.textContent==='本次声音试片没有图片参考；按已确认的画外声源制作。'));assert.equal(calls.length,0);
''')

    def test_direction_initial_prompt_and_model_share_one_card_without_duplicate_intake(self):
        self.run_dom(r"""
asset.input_editable=true;asset.input_required=false;asset.creation_input={mode:'gpt',confirmed:true,revision:1,references:[]};
const card=assetCard(asset),all=walk(card);assert.equal(card.dataset.assetCard,asset.id);
assert.equal(all.filter(node=>node.tag==='article').length,1);
assert.equal(all.filter(node=>node.tag==='h3'&&node.textContent===asset.name).length,1);
assert.equal(all.filter(node=>node.tag==='textarea'&&node.id==='asset-generation-prompt-'+asset.id).length,1);
assert.equal(all.filter(node=>node.tag==='select').length,1);
assert.ok(all.some(node=>node.textContent==='生成方向与参考图'));
assert.ok(all.some(node=>node.textContent==='确认提示词并生成资产'));
paintIntake(state);assert.equal(calls.length,0);assert.equal(nodes.has('asset-inputs'),false);
""")
        self.assertNotIn('id="asset-inputs"', HTML)
        self.assertNotIn('.asset:has(.asset-prompt-approval){grid-column:1/-1}', HTML)
        self.assertGreater(HTML.index('id="intake-panel"'), HTML.index('id="assets-panel"'))

    def test_asset_edit_reads_original_prompt_then_one_exact_submission_without_gpt_feedback(self):
        self.run_dom(r"""
asset.status='candidate';asset.file='media/synthetic-v1.png';
snapshot={id:asset.id,version:1,target_version:2,sha256:'c'.repeat(64),prompt:'  原始实际提交提示词\n含全部样式与规则。\n',model:'image-fast',models:asset.generation_models,reference_manifest:{images:[],audio:[]}};
let content=walk(assetCard(asset));assert.equal(content.some(node=>node.textContent==='提出修改'),false);
content.find(node=>node.textContent==='修改提示词').listeners.click();
await new Promise(resolve=>setTimeout(resolve,0));
assert.equal(calls.length,1);assert.ok(calls[0].url.startsWith('/api/asset-prompt-editor?id='));assert.equal(calls[0].headers['X-Norlan-Token'],'synthetic-token');assert.equal(calls[0].payload,null);
content=walk(assetCard(asset));const editor=content.find(node=>node.tag==='textarea'),select=content.find(node=>node.tag==='select');
assert.equal(editor.value,snapshot.prompt);assert.equal(select.value,'image-fast');
const exact='  人工直接修改，原样发给图片引擎。\n\n';editor.value=exact;editor.listeners.input();select.value='image-pro';select.listeners.change();
const submit=content.find(node=>node.textContent==='提交提示词并生成');submit.listeners.click();submit.listeners.click();
await new Promise(resolve=>setTimeout(resolve,0));
assert.equal(calls.length,2);assert.equal(calls[1].url,'/api/prompt-edit');
assert.equal(JSON.stringify(calls[1].payload),JSON.stringify({action:'asset_edit_and_generate',id:asset.id,version:1,sha256:snapshot.sha256,prompt:exact,model:'image-pro'}));
assert.equal(assetEditSession(asset).open,false);assert.equal(asset.generation_prompt.prompt.startsWith('  完整人物'),true);
assert.equal(calls.some(call=>['reject','run','planning_feedback'].includes(call.payload?.action)),false);
""")

    def test_asset_edit_cancel_poll_and_failed_submission_keep_exact_draft(self):
        self.run_dom(r"""
asset.status='candidate';snapshot={id:asset.id,version:1,sha256:'d'.repeat(64),prompt:'原始提示词',model:'image-fast',models:asset.generation_models,reference_manifest:{images:[],audio:[]}};
await openAssetPromptEditor(asset);let all=walk(assetDirectEditor(asset)),editor=all.find(node=>node.tag==='textarea'),select=all.find(node=>node.tag==='select');
const exact='  未提交人工草稿\n';editor.value=exact;editor.listeners.input();select.value='image-pro';select.listeners.change();
all=walk(assetDirectEditor(asset));assert.equal(all.find(node=>node.tag==='textarea').value,exact);
all.find(node=>node.textContent==='取消修改').listeners.click();assert.equal(assetEditSession(asset).open,false);assert.equal(calls.length,1);
await openAssetPromptEditor(asset);all=walk(assetDirectEditor(asset));assert.equal(all.find(node=>node.tag==='textarea').value,exact);assert.equal(all.find(node=>node.tag==='select').value,'image-pro');
fail=true;all.find(node=>node.textContent==='提交提示词并生成').listeners.click();await new Promise(resolve=>setTimeout(resolve,0));
assert.equal(assetEditSession(asset).open,true);assert.equal(requestBusy,false);all=walk(assetDirectEditor(asset));assert.equal(all.find(node=>node.tag==='textarea').value,exact);assert.equal(all.find(node=>node.tag==='select').value,'image-pro');
assert.equal(calls.filter(call=>call.payload).length,1);assert.ok(all.some(node=>node.textContent==='合成测试：候选已经更新'));
""")

    def test_direct_editor_reference_preview_is_bound_to_loaded_request(self):
        self.run_dom(r"""
asset.status='candidate';snapshot={id:asset.id,version:1,sha256:'f'.repeat(64),prompt:'原始提示词',model:'image-fast',models:asset.generation_models,reference_manifest:{images:['media/synthetic-approved.png'],audio:[]}};
await openAssetPromptEditor(asset);const all=walk(assetDirectEditor(asset));all.find(node=>node.className==='asset-image-button').listeners.click();
previewAsset=previews[0];previewInfo();assert.equal($('image-info').textContent,'本次生成参考图');assert.equal($('image-source').hidden,false);
asset.version=2;previewInfo();assert.ok($('image-info').textContent.includes('当前参考清单已更新'));assert.equal($('image-source').hidden,true);assert.equal(calls.length,1);
""")

    def test_asset_editor_stale_version_pause_and_loading_failure_never_generate(self):
        self.run_dom(r"""
asset.status='candidate';snapshot={id:asset.id,version:1,sha256:'e'.repeat(64),prompt:'原始提示词',model:'image-fast',models:asset.generation_models,reference_manifest:{images:[],audio:[]},source_notice:'合成旧版资产提示词说明'};
await openAssetPromptEditor(asset);let all=walk(assetDirectEditor(asset));assert.ok(all.some(node=>node.textContent===snapshot.source_notice));
state.paused=true;all=walk(assetDirectEditor(asset));const button=all.find(node=>node.textContent==='提交提示词并生成');assert.equal(button.disabled,true);button.listeners.click();state.paused=false;
asset.version=2;all=walk(assetDirectEditor(asset));assert.equal(all.find(node=>node.tag==='textarea').value,snapshot.prompt);assert.equal(all.find(node=>node.textContent==='提交提示词并生成').disabled,true);all.find(node=>node.textContent==='提交提示词并生成').listeners.click();assert.equal(calls.length,1);
readFail=true;await openAssetPromptEditor(asset);all=walk(assetDirectEditor(asset));assert.equal(all.some(node=>node.textContent==='提交提示词并生成'),false);assert.ok(all.some(node=>node.textContent==='合成只读检查失败'));
assert.equal(calls.every(call=>call.payload===null),true);
""")

    def test_intake_labels_only_prepare_prompts_and_keep_reference_upload_and_paste(self):
        self.assertIn('准备资产提示词', HTML)
        self.assertIn('准备声音试听提示词', function_source('updateInputControls'))
        self.assertNotIn('按以上选择生成资产', HTML)
        self.assertIn('先准备完整提示词', HTML)
        self.assertIn('uploadReferences(asset.id,files)', function_source('inputCard'))
        self.assertIn('addEventListener("paste"', function_source('inputCard'))
        self.assertIn('asset.status==="awaiting_asset_prompt_approval"?assetPromptApproval(asset):null', function_source('assetCard'))
        self.assertIn('"awaiting_asset_prompt_approval"].includes(item.status)', function_source('applyEpisodeRoute'))


if __name__ == '__main__':
    unittest.main()
