"""Final review must target the current postproduction export, never the dry cut."""
import json
from pathlib import Path
import re
import shutil
import subprocess
import unittest

from norlan import dashboard


@unittest.skipUnless(shutil.which("node"), "Node.js required for synthetic DOM contract")
class FinalPreviewRenderingTests(unittest.TestCase):
    def run_dom(self, assertions):
        html = Path(dashboard.__file__).with_name("dashboard.html").read_text()
        sources = []
        for name in ("el", "mediaURL", "appendPrompt", "episodeCard", "postproductionFeedback",
                     "musicTrack", "postproductionEditor", "subtitleSyncReport", "postproductionPanel", "paintPostproductionStatus"):
            match = re.search(r"(?:async )?function " + name + r"\(", html)
            following = re.search(r"\n(?:async )?function ", html[match.start() + 1:])
            end = match.start() + 1 + following.start() if following else len(html)
            sources.append(html[match.start():end])
        # Exercise the actual run-button listener; rendering final review must not
        # change its meaning back to submitting another production call.
        listener_start = html.index('$("run").addEventListener("click",')
        listener_end = html.index(';$("pause").addEventListener', listener_start)
        sources.append(html[listener_start:listener_end] + ";")
        program = r'''
const assert=require('node:assert/strict'),vm=require('node:vm');
const {sources,assertions}=JSON.parse(require('node:fs').readFileSync(0,'utf8'));
class Element{
 constructor(tag){this.tag=tag;this.children=[];this.dataset={};this.listeners={};this.textContent='';this.value='';this.scrolls=[];this.focuses=[];}
 append(...nodes){this.children.push(...nodes);for(const node of nodes)node.parent=this;}
 setAttribute(key,value){this[key]=value;}
 addEventListener(key,handler){this.listeners[key]=handler;}
 matches(selector){
  if(selector==='input:checked')return this.tag==='input'&&this.checked;
  const match=/^([\w-]+)?(?:\.([\w-]+))?$/.exec(selector);
  if(!match)throw new Error('Unsupported selector '+selector);
  return (!match[1]||this.tag===match[1])&&(!match[2]||String(this.className||'').split(' ').includes(match[2]));
 }
 querySelectorAll(selector){const found=[];for(const child of this.children){if(child.matches?.(selector))found.push(child);found.push(...(child.querySelectorAll?.(selector)||[]));}return found;}
 querySelector(selector){return this.querySelectorAll(selector)[0]||null;}
 scrollIntoView(options){this.scrolls.push(options);}
 focus(options){this.focuses.push(options);}
}
const nodes=new Map(),actions=[];
const state={paused:false,dashboard:{running:false},episodes:[{id:'E1'}],music_library:{mode:'library',folder:'/synthetic/music',tracks:[]}};
const context={assert,state,actions,nodes,document:{createElement:tag=>new Element(tag),createTextNode:text=>Object.assign(new Element('text'),{textContent:text})},
 $:id=>{if(!nodes.has(id))nodes.set(id,new Element('div'));return nodes.get(id);},
 badge:()=>new Element('badge'),reviewDetails:()=>null,
 appendEpisodeProduction:card=>{const rough=new Element('section');rough.className='synthetic-rough-cut';card.append(rough);},
 perform:(...args)=>actions.push(args),uploadMusic:()=>{},showRevision:()=>{},toast:()=>{}};
vm.runInNewContext(sources.join('\n')+`
const walk=node=>[node,...(node.children||[]).flatMap(walk)];
const episode={id:'E1',title:'合成测试第一集',version:4,status:'awaiting_review',file:'exports/E1-v4.mp4',has_subtitles:true,subtitles_burned:true,music_status:'approved_library_mixed',shots:[{id:'S1'},{id:'S2'}],postproduction:{enabled:true,ready:true,final_available:true,final_ready:true,final_file:'exports/E1-v4.mp4',final_version:4,final_sha256:'f'.repeat(64),dry_file:'exports/E1-original-audio.mp4',status:'awaiting_final_approval',subtitles:{status:'corrected',sync_check:{status:'passed',summary:'合成字幕已按独立音频证据检查。',attempts:1,max_attempts:2,checked_cues:3,total_cues:3,issues:[]}},music:{status:'mixed'}}};state.episodes=[episode];
`+assertions,context);
'''
        result = subprocess.run([shutil.which("node"), "-e", program],
                                input=json.dumps({"sources": sources, "assertions": assertions}),
                                text=True, capture_output=True, timeout=15)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_final_player_and_unique_version_bound_approval_are_adjacent(self):
        self.run_dom(r'''
const card=episodeCard(episode,0),post=card.querySelector('.postproduction'),preview=post.querySelector('.postproduction-final-preview');
assert.ok(preview);assert.equal(post.querySelector('.final-preview-title').textContent,'最终成片预览');
assert.equal(post.querySelector('.final-media-status').textContent,'字幕已烧录 · 配乐已混入');
assert.equal(preview.querySelector('video').src,'/media/exports/E1-v4.mp4');
assert.equal(preview.querySelector('video').dataset.mediaKind,'postproduction-final');
assert.equal(preview.dataset.finalVersion,'4');
assert.equal(card.querySelectorAll('video').length,1,'the final is not duplicated at the episode top');
assert.equal(card.querySelectorAll('video').some(video=>video.src.includes('original-audio')),false);
assert.equal(card.querySelectorAll('.final-approval').length,1);
assert.equal(preview.querySelectorAll('.postproduction-feedback').length,2,'subtitle and music rework stay next to preview');
assert.equal(preview.querySelector('a').href,preview.querySelector('video').src,'download targets the final too');
preview.querySelector('.final-approval').querySelector('button').listeners.click();
assert.equal(actions.at(-1)[0].action,'approve');assert.equal(actions.at(-1)[0].kind,'episode');assert.equal(actions.at(-1)[0].version,4);
assert.equal(card.querySelector('.episode-video-rework').tag,'details');
assert.equal(Boolean(card.querySelector('.episode-video-rework').open),false,'video rework is a separate collapsed action');
''')

    def test_subtitle_feedback_includes_audio_realignment_and_keeps_human_final_review(self):
        self.run_dom(r'''
const card=episodeCard(episode,0),preview=card.querySelector('.postproduction-final-preview');
const subtitleFeedback=preview.querySelector('.postproduction-feedback'),feedback=subtitleFeedback.querySelector('textarea');
assert.ok(feedback.placeholder.includes('前五条字幕出现太早'));
assert.ok(walk(subtitleFeedback).some(node=>node.textContent.includes('按实际人声重新识别并对齐字幕')));
assert.ok(walk(subtitleFeedback).some(node=>node.textContent.includes('保留已有视频、音乐和片段审批')));
feedback.value='前五条字幕比人声提前，尤其第一条，请按实际声音重新对齐。';
subtitleFeedback.querySelector('button').listeners.click();
assert.deepEqual(JSON.parse(JSON.stringify(actions.at(-1)[0])),{action:'postproduction',id:'E1',version:4,target:'subtitles',feedback:feedback.value});
assert.equal(actions.some(action=>action[0].action==='approve'),false);
''')

    def test_rework_and_inconsistent_final_never_present_old_video_as_current(self):
        self.run_dom(r'''
for(const update of [{final_available:false,final_ready:false},{ready:false},{final_file:'exports/E1-v3.mp4'},{final_version:3},{final_file:null}]){
 const changed={...episode,postproduction:{...episode.postproduction,...update}};
 const card=episodeCard(changed,0);
 assert.equal(card.querySelector('.postproduction-final-preview'),null);
 assert.equal(card.querySelector('.final-approval'),null);
 assert.equal(card.querySelectorAll('video').length,0);
}
const pending=episodeCard({...episode,postproduction:{...episode.postproduction,final_available:false,final_ready:false}},0);
assert.ok(pending.querySelector('.postproduction-preview-pending'));
assert.ok(pending.querySelector('.postproduction-preview-pending').textContent.includes('新的最终成片'));
''')

    def test_subtitle_and_music_feedback_keep_postproduction_routes_and_shot_selection_is_separate(self):
        self.run_dom(r'''
const card=episodeCard(episode,0),preview=card.querySelector('.postproduction-final-preview');
for(const target of ['subtitles','music']){
 const feedback=preview.querySelectorAll('.postproduction-feedback').find(node=>node.dataset.uiKey==='post-feedback:E1:'+target);
 feedback.querySelector('textarea').value='合成后期调整意见';feedback.querySelector('button').listeners.click();
 assert.equal(actions.at(-1)[0].action,'postproduction');assert.equal(actions.at(-1)[0].target,target);assert.equal(actions.at(-1)[0].version,4);
 assert.equal(actions.at(-1)[0].kind,undefined);assert.equal(actions.at(-1)[0].shot_ids,undefined);
}
const picture=card.querySelector('.episode-video-rework');picture.querySelector('textarea').value='第二个片段画面需要修改';
picture.querySelectorAll('input')[1].checked=true;picture.querySelector('button').listeners.click();
assert.equal(actions.at(-1)[0].action,'reject');assert.equal(actions.at(-1)[0].kind,'episode');assert.equal(actions.at(-1)[0].shot_ids.join(','),'S2');
''')

    def test_final_review_navigation_never_starts_production(self):
        self.run_dom(r'''
const card=episodeCard(episode,0),preview=card.querySelector('.postproduction-final-preview');nodes.set(preview.id,preview);
paintPostproductionStatus(state);assert.equal(nodes.get('run').textContent,'查看最终成片');assert.equal(nodes.get('run').disabled,false);
nodes.get('run').listeners.click();assert.equal(preview.scrolls.length,1);assert.equal(preview.focuses.length,1);assert.equal(actions.length,0);
state.episodes=[{...episode,status:'approved'}];nodes.get('run').listeners.click();assert.equal(actions.at(-1)[0].action,'run','approved episodes do not capture the next workflow');
''')

    def test_zero_dialogue_metadata_does_not_claim_subtitles_were_added(self):
        self.run_dom(r'''
const card=episodeCard({...episode,has_subtitles:false,subtitles_burned:false},0);
assert.equal(card.querySelector('.final-media-status').textContent,'无实际对白字幕 · 配乐已混入');
''')

    def test_sync_report_is_collapsed_and_passed_requires_human_verification(self):
        self.run_dom(r'''
const card=episodeCard(episode,0),report=card.querySelector('.subtitle-sync-report');
assert.ok(report);assert.equal(report.tag,'details');assert.equal(Boolean(report.open),false);
assert.equal(report.id,'subtitle-sync-E1');assert.equal(report.dataset.uiKey,'subtitle-sync:E1');
assert.ok(walk(report).some(node=>node.textContent==='字幕自动同步检查通过'));
assert.ok(walk(report).some(node=>node.textContent.includes('重新对齐 1 / 2 次')));
assert.ok(walk(report).some(node=>node.textContent.includes('已检查 3 / 3 条字幕')));
assert.ok(walk(report).some(node=>node.textContent.includes('请完整播放成片')));
assert.equal(actions.length,0,'automatic pass does not perform human approval');
''')

    def test_unchecked_checking_repairing_and_uncertain_keep_preview_but_block_approval(self):
        self.run_dom(r'''
for(const status of [null,'checking','repairing','needs_attention']){
 const sync_check=status?{status,summary:'合成检查说明',attempts:2,max_attempts:2,checked_cues:2,total_cues:3,issues:[{start:6.56,end:8.9,reason:'字幕开始时间偏早'}]}:undefined;
 const changed={...episode,postproduction:{...episode.postproduction,final_ready:false,subtitles:{...episode.postproduction.subtitles,sync_check}}};
 const card=episodeCard(changed,0),preview=card.querySelector('.postproduction-final-preview'),report=card.querySelector('.subtitle-sync-report');
 assert.ok(preview,'the same current export remains available');
 assert.equal(preview.querySelector('video').src,'/media/exports/E1-v4.mp4');
 assert.equal(card.querySelector('.final-approval'),null);
 assert.ok(card.querySelector('.subtitle-sync-approval-blocked'));
 if(status)assert.equal(report.querySelector('time').textContent,'00:06.6–00:08.9 · ');
 assert.equal(preview.querySelectorAll('.postproduction-feedback').length,2,'the same feedback route remains usable');
}
assert.equal(actions.length,0);
''')

    def test_attention_navigation_does_not_restart_checks_or_production(self):
        self.run_dom(r'''
episode.postproduction.final_ready=false;
episode.postproduction.subtitles.sync_check={status:'needs_attention',summary:'两轮后仍有疑点',attempts:2,max_attempts:2,issues:[{start:6,end:8,reason:'仍无法验证同步'}]};
const card=episodeCard(episode,0),report=card.querySelector('.subtitle-sync-report');nodes.set(report.id,report);
const locate=walk(card).find(node=>node.tag==='button'&&node.textContent==='查看字幕检查报告');
assert.ok(locate);locate.listeners.click();assert.equal(report.open,true);assert.equal(report.scrolls.length,1);assert.equal(actions.length,0);
paintPostproductionStatus(state);assert.equal(nodes.get('status').textContent,'字幕同步需要处理');assert.equal(nodes.get('run').textContent,'查看字幕检查报告');
nodes.get('run').listeners.click();assert.equal(report.scrolls.length,2);assert.equal(actions.length,0);
''')

    def test_live_check_progress_is_distinct_and_no_final_gate_is_opened(self):
        self.run_dom(r'''
state.dashboard.running=true;
for(const status of ['checking','repairing']){
 episode.postproduction.subtitles.sync_check={status,summary:'当前检查说明'};
 paintPostproductionStatus(state);
 assert.ok(nodes.get('status').textContent.includes(status==='checking'?'检查字幕同步':'修复字幕同步'));
 assert.equal(nodes.get('run').disabled,true);
 const card=episodeCard(episode,0);assert.equal(card.querySelector('.final-approval'),null,'status itself also blocks a racing final_ready flag');
}
assert.equal(actions.length,0);
''')

    def test_manual_revision_can_be_previewed_with_uncertain_sync_without_approval(self):
        self.run_dom(r'''
episode.postproduction.manual_preview=true;episode.postproduction.final_ready=false;
episode.postproduction.status='subtitle_sync_needs_attention';
episode.postproduction.subtitles.sync_check={status:'needs_attention',summary:'手动时间已保留，独立证据仍有疑点',attempts:0,max_attempts:2,issues:[]};
const card=episodeCard(episode,0),preview=card.querySelector('.postproduction-final-preview');
assert.ok(preview);assert.equal(preview.querySelector('video').src,'/media/exports/E1-v4.mp4');
assert.equal(preview.querySelector('.final-preview-title').textContent,'手动调整成片预览');
assert.ok(preview.querySelector('.subtitle-sync-approval-blocked').textContent.includes('字幕检查仍有疑点'));
assert.equal(card.querySelector('.final-approval'),null);assert.equal(actions.length,0);
''')

    def test_manual_adoption_is_final_human_verification_without_automatic_recheck_claim(self):
        self.run_dom(r'''
episode.postproduction.subtitles.sync_check={status:'manual',summary:'采用你输入的字幕文字和时间，不再自动复检。',attempts:2,max_attempts:2,checked_cues:3,total_cues:3,issues:[]};
episode.postproduction.subtitles.previous_sync_check={status:'passed',summary:'调整前的自动检查记录',issues:[{start:6,end:8,reason:'调整前检查说明'}]};
const card=episodeCard(episode,0),report=card.querySelector('.subtitle-sync-report'),history=report.querySelector('.subtitle-sync-history'),preview=card.querySelector('.postproduction-final-preview');
assert.equal(report.children[0].textContent,'手动调整版 · 不再自动复检，请试看后最终确认');
assert.ok(walk(report).some(node=>node.textContent.includes('不再调用识别、GPT 校对或同步复检')));
assert.equal(walk(report).some(node=>node.textContent.includes('本轮已自动重新对齐')),false);
assert.equal(walk(report).some(node=>node.textContent.includes('已检查 3 / 3')),false);
assert.ok(history);assert.equal(Boolean(history.open),false);
assert.ok(walk(history).some(node=>node.textContent.includes('不代表本次手动调整重新经过自动检查')));
assert.ok(preview.querySelector('.final-approval'));assert.equal(actions.length,0);
assert.equal(walk(preview).some(node=>node.textContent.includes('字幕自动检查通过')),false);
paintPostproductionStatus(state);assert.ok(nodes.get('status-note').textContent.includes('不再自动复检'));
''')


if __name__ == "__main__":
    unittest.main()
