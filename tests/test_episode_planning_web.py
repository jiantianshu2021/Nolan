"""Synthetic HTTP human-plan gate, independent of media generation."""
import copy
import http.client
import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

from norlan import core, dashboard, episode_planning, pipeline


class EpisodePlanningWebTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        core.create_project(self.root, "synthetic source")
        plan = pipeline.demo_plan()
        core.install_plan(self.root, plan)
        core.approve(self.root, "plan", "plan", 1)
        source = self.root / "source/ep02.md"
        source.write_text("第二集合成剧本，不是真实作品。", encoding="utf-8")
        final = self.root / "exports/synthetic.mp4"
        final.write_bytes(b"synthetic fixture")
        with core.transaction(self.root) as state:
            state["episodes"][0].update(status="approved", file="exports/synthetic.mp4", sha256=core.digest(final),
                                        approved_sha256=core.digest(final))
            state["episodes"][1].update(status="awaiting_episode_plan", shots=[], source_file="source/ep02.md",
                                        source_sha256=core.digest(source), source_duration=8)
        self.result = {"episode": copy.deepcopy(plan["episodes"][1]), "new_assets": []}
        for shot in self.result["episode"]["shots"]:
            shot["subshots"] = [{"start": 0, "end": shot["duration"], "prompt": "合成测试镜头。"}]
        episode_planning.prepare_episode(self.root, "E002", planner=lambda *_: self.result)
        self.server = dashboard.make_server(self.root, port=0)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.addCleanup(self.close_server)

    def close_server(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)

    def send(self, **changes):
        pending = core.load_project(self.root)["episodes"][1]["planning"]
        data = {"action": "episode_plan_decide", "id": "E002", "revision": pending["revision"],
                "signature": pending["signature"], "decision": "approve", **changes}
        connection = http.client.HTTPConnection("127.0.0.1", self.server.server_address[1], timeout=3)
        connection.request("POST", "/api/action", json.dumps(data),
            {"Content-Type": "application/json", "X-Norlan-Token": self.server.context.token})
        response = connection.getresponse()
        status, body = response.status, json.loads(response.read())
        connection.close()
        return status, body

    def test_human_approval_installs_plan_without_automatic_media_submission(self):
        before = core.load_project(self.root)
        with patch.object(dashboard, "_run", side_effect=AssertionError("approval must not run production")):
            status, response = self.send()
        self.assertEqual(status, 200)
        self.assertEqual(response["episodes"][1]["status"], "planned")
        after = core.load_project(self.root)
        self.assertEqual(after["episodes"][0], before["episodes"][0])
        self.assertEqual(after["jobs"], before["jobs"])
        self.assertEqual(after["episodes"][1]["shots"][0]["status"], "planned")

    def test_stale_hash_and_injected_approval_are_rejected_without_state_changes(self):
        before = (self.root / "project.json").read_bytes()
        for changes in ({"signature": "f"*64}, {"revision": True}, {"approved": True}, {"decision": "reject", "feedback": ""}):
            with self.subTest(changes=changes):
                status, _ = self.send(**changes)
                self.assertEqual(status, 409)
                self.assertEqual((self.root / "project.json").read_bytes(), before)

    def test_feedback_returns_to_planning_without_overwriting_source(self):
        before = (self.root / "source/ep02.md").read_bytes()
        status, response = self.send(decision="reject", feedback="第二段先停顿再转身。")
        self.assertEqual(status, 200)
        self.assertEqual(response["episodes"][1]["planning"]["status"], "rejected")
        self.assertEqual(response["episodes"][1]["shots"], [])
        self.assertEqual((self.root / "source/ep02.md").read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
