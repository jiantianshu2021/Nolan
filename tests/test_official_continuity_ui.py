"""Synthetic official-reference projection/prompt tests; no provider requests."""
import copy
import json
from pathlib import Path
import shutil
import subprocess
import unittest
from unittest.mock import patch

from norlan import dashboard
from norlan.brain import Brain


class OfficialContinuityProjectionTests(unittest.TestCase):
    def setUp(self):
        self.source = {"id": "S1", "version": 1, "status": "accepted", "duration": 10,
                       "file": "media/E1/S1-v1.mp4", "sha256": "a" * 64}
        self.shot = {"id": "S2", "version": 2, "status": "review_required", "duration": 10,
                     "file": "media/E1/S2-v2.mp4", "sha256": "b" * 64, "job_id": "job-two"}
        self.job = {"kind": "video", "item_id": "S2", "state": "completed", "accounting": "spent",
                    "output": "media/E1/S2-v2.mp4", "task_id": "synthetic-task-2", "amount": 0}
        self.state = {"status": "review_required", "assets": [], "plan": {"status": "approved"},
                      "budget": {"currency": "CNY"}, "jobs": {"job-two": self.job},
                      "episodes": [{"id": "E1", "status": "generating", "shots": [self.source, self.shot]}]}

    def public(self):
        return dashboard.public_state(self.state)["episodes"][0]["shots"][1]

    def set_references(self, mode):
        key, role = (("video_references", "previous_original_video") if mode == "original_video"
                     else ("image_references", "previous_official_tail"))
        self.shot["reference_manifest"] = {"context": {
            "previous_actual": {"shot_id": "S1", "reference_mode": mode},
            key: [{"index": 1, "role": role, "shot_id": "S1", "sha256": "a" * 64,
                   "url": "https://synthetic.invalid/private.mp4?token=do-not-publish"}],
            "observation_images": [{"index": 2, "role": "previous_observation_tail", "shot_id": "S1",
                                    "sent_to_video_engine": False, "file": "evidence/local-tail.png"}]}}

    def test_saved_reference_modes_get_short_labels_without_private_metadata(self):
        for mode, label in (("original_video", "衔接参考：第 1 段原始视频"),
                            ("official_tail", "衔接参考：第 1 段官方尾帧")):
            with self.subTest(mode=mode):
                self.set_references(mode)
                before = copy.deepcopy(self.state)
                public = self.public()
                self.assertEqual(public["continuity_reference"], {"mode": mode, "label": label})
                self.assertNotIn("synthetic.invalid", json.dumps(public))
                self.assertNotIn("do-not-publish", json.dumps(public))
                self.assertNotIn("local-tail.png", json.dumps(public))
                self.assertEqual(self.state, before)

    def test_observation_only_tail_cannot_be_claimed_as_an_engine_reference(self):
        self.set_references("official_tail")
        context = self.shot["reference_manifest"]["context"]
        context.pop("image_references")
        self.assertNotIn("continuity_reference", self.public())
        self.set_references("original_video")
        context = self.shot["reference_manifest"]["context"]
        context["video_references"][0]["shot_id"] = "different-shot"
        self.assertNotIn("continuity_reference", self.public())
        context["video_references"][0]["shot_id"] = "S1"
        context["video_references"][0]["sha256"] = "invalid"
        self.assertNotIn("continuity_reference", self.public())

    def test_tail_download_warning_keeps_successful_video_and_human_gate(self):
        self.job["official_tail_error"] = {"stage": "download", "message": "合成测试：官方尾帧下载失败",
                                           "private_url": "https://synthetic.invalid/private?token=do-not-publish"}
        public = self.public()
        self.assertIn("本段视频可正常审阅", public["official_tail_warning"])
        self.assertEqual(public["file"], self.shot["file"])
        self.assertEqual(public["status"], "review_required")
        self.assertTrue(public["progress"]["waiting_for_human"])
        self.assertNotIn("failure", public)
        self.assertNotIn("do-not-publish", json.dumps(public))

    def test_wrong_job_or_stale_version_cannot_attach_tail_warning(self):
        self.job["official_tail_error"] = {"message": "合成测试"}
        self.job["item_id"] = "unrelated-shot"
        self.assertNotIn("official_tail_warning", self.public())
        self.job["item_id"] = "S2"
        self.job["output"] = "media/E1/S2-v1.mp4"
        self.assertNotIn("official_tail_warning", self.public())


class OfficialContinuityPromptTests(unittest.TestCase):
    def test_prompt_distinguishes_engine_references_from_gpt_observation(self):
        context = {
            "previous_actual": {"shot_id": "S1", "reference_mode": "original_video"},
            "video_references": [{"index": 1, "role": "previous_original_video", "shot_id": "S1"}],
            "image_references": [{"index": 1, "role": "approved_asset"}],
            "observation_images": [
                {"index": 1, "generation_image_index": 1, "role": "approved_asset"},
                {"index": 2, "role": "previous_observation_tail", "sent_to_video_engine": False}],
        }
        before = copy.deepcopy(context)
        with patch.object(Brain, "_call", return_value={"prompt": "合成提示词", "dialogue": "", "continuity_notes": ""}) as call:
            Brain().refine_shot({"id": "S2", "duration": 10}, context,
                                [Path("synthetic-asset.png"), Path("synthetic-observation.png")])
        instruction, data, _, images = call.call_args.args
        self.assertEqual(context, before)
        self.assertEqual(data["context"], before)
        self.assertEqual(len(images), 2)
        for rule in ("context.image_references", "context.observation_images", "sent_to_video_engine=false",
                     "不得把它写成已经上传给视频引擎的图片参考", "generation_image_index",
                     "context.video_references", "视频 1 / video1", "从视频结尾继续", "不重复已说台词",
                     "previous_official_tail", "没有播放完整视频或听到音轨"):
            self.assertIn(rule, instruction)


@unittest.skipUnless(shutil.which("node"), "Node.js required for pure DOM rendering test")
class OfficialContinuityRenderingTests(unittest.TestCase):
    def test_source_and_warning_do_not_hide_video_or_add_actions(self):
        html = Path(dashboard.__file__).with_name("dashboard.html").read_text()
        sources = []
        for name in ("el", "mediaURL", "appendPrompt", "failureDetails", "shotReview", "advisoryReport", "subshotPlan", "shotCard"):
            start = html.index("function " + name + "(")
            sources.append(html[start:html.index("\nfunction ", start + 1)])
        program = r'''
const assert = require("node:assert/strict"), vm = require("node:vm");
const sources = JSON.parse(require("node:fs").readFileSync(0,"utf8"));
class Element {
 constructor(tag){this.tag=tag;this.children=[];this.dataset={};this.listeners={};this.attributes={};this.className="";this.textContent="";}
 append(...nodes){this.children.push(...nodes);}
 setAttribute(key,value){this.attributes[key]=value;}
 addEventListener(key,handler){this.listeners[key]=handler;}
}
vm.runInNewContext(sources.join("\n")+`
const walk=node=>[node,...(node.children||[]).flatMap(walk)];
const shot={id:'S2',version:2,status:'review_required',duration:10,file:'media/E1/S2-v2.mp4'};
const before=walk(shotCard(shot,1,null));
shot.continuity_reference={mode:'original_video',label:'衔接参考：第 1 段原始视频'};
shot.official_tail_warning='本段视频可正常审阅；官方尾帧尚未保存，后续衔接需先恢复原任务素材。';
const nodes=walk(shotCard(shot,1,null));
assert.equal(nodes.filter(n=>n.tag==='video').length,1);
assert.equal(nodes.some(n=>n.textContent===shot.continuity_reference.label),true);
assert.equal(nodes.some(n=>n.textContent===shot.official_tail_warning),true);
assert.deepEqual(nodes.filter(n=>n.tag==='button').map(n=>n.textContent),before.filter(n=>n.tag==='button').map(n=>n.textContent));
assert.equal(nodes.some(n=>n.className==='generation-failure'),false);
`,{document:{createElement:tag=>new Element(tag),createTextNode:text=>({tag:'text',textContent:text})},assert});
'''
        done = subprocess.run([shutil.which("node"), "-e", program], input=json.dumps(sources), text=True,
                              capture_output=True, timeout=15)
        self.assertEqual(done.returncode, 0, done.stderr)


if __name__ == "__main__":
    unittest.main()
