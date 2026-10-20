import copy
import io
import json
from pathlib import Path
import tempfile
import threading
import unittest
import subprocess
import sys
from unittest.mock import patch

from norlan import core, mcp_server, pipeline


class MCPTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.workspace = Path(self.temp.name).resolve() / "workspace"
        self.last_patch = patch.object(mcp_server, "_last_project", return_value=None)
        self.last_patch.start()
        self.addCleanup(self.last_patch.stop)
        self.server = mcp_server.MCPServer(self.workspace)
        self.addCleanup(self.server.close)
        self.root = self.server.projects / "story"
        self.initialize()

    def initialize(self, protocol="2025-06-18"):
        return self.server.handle({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {
            "protocolVersion": protocol, "capabilities": {}, "clientInfo": {"name": "test", "version": "1"}}})

    def call(self, name, **arguments):
        return self.server.handle({"jsonrpc": "2.0", "id": 2, "method": "tools/call", "params": {
            "name": name, "arguments": arguments}})["result"]

    def value(self, result):
        self.assertFalse(result["isError"], result)
        return json.loads(result["content"][0]["text"])

    def create(self, mode="live"):
        return self.value(self.call("create_project", project=str(self.root), novel_text="原创测试小说。", mode=mode))

    def save_plan(self):
        return self.value(self.call("save_plan", project=str(self.root), plan=pipeline.demo_plan()))

    def test_protocol_versions_and_unknown_version_negotiation(self):
        for protocol in mcp_server.PROTOCOLS:
            self.assertEqual(self.initialize(protocol)["result"]["protocolVersion"], protocol)
        self.assertEqual(self.initialize("future")["result"]["protocolVersion"], "2025-06-18")
        tools = self.server.handle({"jsonrpc": "2.0", "id": 3, "method": "tools/list"})["result"]["tools"]
        self.assertEqual(len(tools), 18)
        self.assertIn("import_season", {tool["name"] for tool in tools})
        self.assertTrue({"list_directory", "find_files"} <= {tool["name"] for tool in tools})
        self.assertTrue(all(t["inputSchema"]["additionalProperties"] is False for t in tools))
        self.assertEqual(self.server.handle({"jsonrpc": "2.0", "id": 4, "method": "ping"})["result"], {})
        self.assertIsNone(self.server.handle({"jsonrpc": "2.0", "method": "notifications/initialized"}))

    def test_invalid_rpc_and_unknown_method_do_not_expose_internals(self):
        for invalid in ([], {"jsonrpc": "2.0", "id": True, "method": "ping"}, {"id": 1, "method": "ping"}):
            self.assertEqual(self.server.handle(invalid)["error"]["code"], -32600)
        response = self.server.handle({"jsonrpc": "2.0", "id": 1, "method": "shell"})
        self.assertEqual(response["error"]["code"], -32601)
        response = self.server.handle({"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": []})
        self.assertEqual(response["error"]["code"], -32602)

    def test_stdio_only_json_and_notifications_silent(self):
        source = [b"not-json\n", json.dumps({"jsonrpc": "2.0", "id": 1, "method": "ping"}).encode() + b"\n",
                  b'{"jsonrpc":"2.0","method":"notifications/initialized"}\n']
        output = io.StringIO()
        self.server.serve(io.BytesIO(b"".join(source)), output)
        responses = [json.loads(line) for line in output.getvalue().splitlines()]
        self.assertEqual(len(responses), 2)
        self.assertEqual(responses[0]["error"]["code"], -32700)
        self.assertEqual(responses[1]["result"], {})
        self.assertTrue(self.server.closed)

    def test_module_stdio_entrypoint_has_clean_protocol_stdout(self):
        messages = [
            {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {
                "protocolVersion": "2024-11-05", "capabilities": {}, "clientInfo": {"name": "test", "version": "1"}}},
            {"jsonrpc": "2.0", "method": "notifications/initialized"},
            {"jsonrpc": "2.0", "id": 2, "method": "tools/list"},
        ]
        result = subprocess.run([sys.executable, "-m", "norlan.mcp_server", "--workspace", str(self.workspace)],
                                input="".join(json.dumps(message) + "\n" for message in messages),
                                text=True, capture_output=True, timeout=10,
                                cwd=Path(__file__).resolve().parents[1])
        self.assertEqual(result.returncode, 0, result.stderr)
        responses = [json.loads(line) for line in result.stdout.splitlines()]
        self.assertEqual([response["id"] for response in responses], [1, 2])
        self.assertEqual(len(responses[1]["result"]["tools"]), 18)
        self.assertEqual(result.stderr, "")

    def test_module_stdio_entrypoint_passes_explicit_workdir(self):
        creative = self.workspace.parent / "终端创作目录"
        creative.mkdir()
        messages = [
            {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {
                "protocolVersion": "2025-06-18", "capabilities": {}, "clientInfo": {"name": "synthetic", "version": "1"}}},
            {"jsonrpc": "2.0", "id": 2, "method": "tools/call", "params": {"name": "list_projects", "arguments": {}}},
        ]
        result = subprocess.run([sys.executable, "-m", "norlan.mcp_server", "--workspace", str(self.workspace), "--workdir", str(creative)],
                                input="".join(json.dumps(message) + "\n" for message in messages),
                                text=True, capture_output=True, timeout=10,
                                cwd=Path(__file__).resolve().parents[1])
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stderr, "")
        response = json.loads(result.stdout.splitlines()[1])["result"]
        self.assertEqual(self.value(response)["creation_directory"], str(creative))

    def test_unknown_and_privileged_tools_are_not_exposed(self):
        self.create()
        before = (self.root / "project.json").read_bytes()
        for name in ("approve", "approve_segments", "configure", "reconcile", "shell", "write_file", "increase_budget"):
            self.assertNotIn(name, mcp_server.TOOL_MAP)
            self.assertTrue(self.call(name, project=str(self.root))["isError"])
        self.assertEqual((self.root / "project.json").read_bytes(), before)

    def test_import_season_registers_sources_without_approval_or_model_calls(self):
        self.create()
        plan = pipeline.demo_plan()
        plan["episodes"] = [plan["episodes"][0]]
        plan["episodes"][0].update(id="ep01", title="合成第一集")
        core.install_plan(self.root, plan)
        core.approve(self.root, "plan", "plan", 1)
        before = core.load_project(self.root)
        folder = Path(self.temp.name) / "provided-scripts"
        folder.mkdir()
        (folder / "synthetic.md").write_text("## 第01集 合成第一集\n预计时长：约120秒\n合成正文。\n"
            "## 第02集 合成第二集\n预计时长：约135秒\n合成后续正文。\n", encoding="utf-8")
        with patch.object(pipeline, "_brain", side_effect=AssertionError("import must not call model")):
            result = self.call("import_season", project=str(self.root), source_folder=str(folder))
        self.assertFalse(result["isError"])
        state = core.load_project(self.root)
        self.assertEqual(state["episodes"][0], before["episodes"][0])
        self.assertEqual(state["assets"], before["assets"])
        self.assertEqual(state["jobs"], before["jobs"])
        self.assertEqual(state["episodes"][1]["status"], "awaiting_episode_plan")
        self.assertEqual(state["episodes"][1]["source_duration"], 135)

    def test_create_requires_exactly_one_source_and_refuses_overwrite(self):
        for extra in ({}, {"novel_text": "正文", "source_path": "/a.txt"}):
            self.assertTrue(self.call("create_project", project=str(self.root), **extra)["isError"])
        self.assertFalse(self.root.exists())
        state = self.create()
        self.assertEqual(state["mode"], "live")
        self.assertFalse(state["budget"]["approved"])
        original = (self.root / "source/novel.txt").read_bytes()
        self.assertTrue(self.call("create_project", project=str(self.root), novel_text="覆盖")["isError"])
        self.assertEqual((self.root / "source/novel.txt").read_bytes(), original)
        other = self.server.projects / "nonempty"
        other.mkdir()
        (other / "keep.txt").write_text("keep")
        self.assertTrue(self.call("create_project", project=str(other), novel_text="覆盖")["isError"])
        self.assertEqual((other / "keep.txt").read_text(), "keep")

    def test_create_defaults_live_and_never_calls_model_or_generation(self):
        with patch("norlan.pipeline.run", side_effect=AssertionError("must not run")), patch("norlan.brain.Brain.call", create=True, side_effect=AssertionError("must not call")):
            result = self.value(self.call("create_project", project=str(self.root), novel_text="正文", brief="黑白漫画"))
        self.assertEqual(result["mode"], "live")
        self.assertEqual(result["creative_brief"], "黑白漫画")
        self.assertEqual(result["plan"]["status"], "pending")

    def test_strict_argument_types_extra_params_and_nested_state_injection(self):
        self.create()
        for name, args in (
            ("project_status", {"project": str(self.root), "settings": {}}),
            ("set_creative_brief", {"project": str(self.root), "brief": [], "budget": 100}),
            ("request_revision", {"project": str(self.root), "kind": "plan", "id": "plan", "version": True, "feedback": "重写"}),
            ("run_production", {"project": str(self.root), "approve": True}),
        ):
            self.assertTrue(self.call(name, **args)["isError"])
        for target, injected in (("plan", {"status": "approved"}), ("asset", {"file": "../../secret.txt"}),
                                 ("episode", {"approved_sha256": "a"}), ("shot", {"status": "accepted"})):
            plan = pipeline.demo_plan()
            item = {"plan": plan, "asset": plan["assets"][0], "episode": plan["episodes"][0],
                    "shot": plan["episodes"][0]["shots"][0]}[target]
            item.update(injected)
            self.assertTrue(self.call("save_plan", project=str(self.root), plan=plan)["isError"])
        self.assertEqual(core.load_project(self.root)["plan"]["version"], 0)

    def test_save_plan_validates_references_and_never_approves(self):
        self.create()
        bad = pipeline.demo_plan()
        bad["episodes"][0]["shots"][0]["asset_ids"] = ["nonexistent"]
        self.assertTrue(self.call("save_plan", project=str(self.root), plan=bad)["isError"])
        state = self.save_plan()
        self.assertEqual(state["plan"]["status"], "candidate")
        self.assertEqual(state["plan"]["version"], 1)
        self.assertEqual(state["episodes"][0]["shots"][0]["asset_ids"], ["hero", "station"])
        self.assertFalse(state["budget"]["approved"])
        self.assertEqual(core.load_project(self.root)["jobs"], {})

    def test_plan_voice_errors_are_actionable_and_do_not_change_project(self):
        self.create()
        before = (self.root / "project.json").read_bytes()
        plan = pipeline.demo_plan()
        plan["assets"].append({"id": "voice", "kind": "voice", "name": "画外声音", "prompt": "仅画外呼声"})
        result = self.call("save_plan", project=str(self.root), plan=plan)
        self.assertTrue(result["isError"])
        error = json.loads(result["content"][0]["text"])
        self.assertEqual(error["category"], "invalid_plan")
        self.assertEqual(error["code"], "voice_character_required")
        self.assertEqual(error["path"], "plan.assets[2].character_id")
        self.assertIn("character", error["hint"])

        plan["assets"][-1]["character_id"] = "hero"
        plan["assets"][0]["character_id"] = "hero"
        result = self.call("save_plan", project=str(self.root), plan=plan)
        self.assertTrue(result["isError"])
        error = json.loads(result["content"][0]["text"])
        self.assertEqual(error["code"], "nonvoice_character_forbidden")
        self.assertEqual(error["path"], "plan.assets[0].character_id")

        del plan["assets"][0]["character_id"]
        shot = plan["episodes"][0]["shots"][0]
        shot["asset_ids"] = ["station", "voice"]
        result = self.call("save_plan", project=str(self.root), plan=plan)
        self.assertTrue(result["isError"])
        error = json.loads(result["content"][0]["text"])
        self.assertEqual(error["code"], "voice_character_not_referenced")
        self.assertEqual(error["path"], "plan.episodes[0].shots[0].asset_ids[1]")
        self.assertIn("画外", error["hint"])
        self.assertEqual((self.root / "project.json").read_bytes(), before)
        self.assertEqual(list((self.root / "plans").iterdir()), [])

        shot["asset_ids"].append("hero")
        shot["prompt"] += "；hero 仅为画外声源，不出现在画面。"
        state = self.value(self.call("save_plan", project=str(self.root), plan=plan))
        self.assertEqual(state["plan"]["status"], "candidate")
        self.assertEqual(state["plan"]["version"], 1)
        self.assertFalse(state["budget"]["approved"])
        self.assertEqual(core.load_project(self.root)["jobs"], {})

    def test_plan_validation_never_echoes_submitted_values(self):
        self.create()
        marker = "sensitive-value-must-not-be-echoed"
        cases = []
        voice_plan = pipeline.demo_plan()
        voice_plan["assets"].append({"id": "voice", "kind": "voice", "name": marker,
                                     "prompt": marker, "character_id": marker})
        cases.append((voice_plan, "voice_character_required", "plan.assets[2].character_id"))
        refs_plan = pipeline.demo_plan()
        refs_plan["episodes"][0]["shots"][0]["asset_ids"] = [marker]
        cases.append((refs_plan, "asset_references_invalid", "plan.episodes[0].shots[0].asset_ids"))
        for plan, code, path in cases:
            with self.subTest(code=code):
                result = self.call("save_plan", project=str(self.root), plan=plan)
                self.assertTrue(result["isError"])
                text = result["content"][0]["text"]
                error = json.loads(text)
                self.assertEqual(error["code"], code)
                self.assertEqual(error["path"], path)
                self.assertNotIn(marker, text)

    def test_schema_errors_identify_known_paths_without_echoing_unknown_fields(self):
        self.create()
        before = (self.root / "project.json").read_bytes()
        cases = []
        missing = pipeline.demo_plan()
        del missing["episodes"][0]["shots"][0]["scene_id"]
        cases.append((missing, "field_required", "arguments.plan.episodes[0].shots[0].scene_id"))
        invalid_type = pipeline.demo_plan()
        invalid_type["episodes"][0]["shots"][0]["duration"] = True
        cases.append((invalid_type, "type_invalid", "arguments.plan.episodes[0].shots[0].duration"))
        unknown = pipeline.demo_plan()
        unknown["assets"][0]["sensitive-unknown-key"] = "sensitive-unknown-value"
        cases.append((unknown, "unknown_field", "arguments.plan.assets[0]"))
        for plan, code, path in cases:
            with self.subTest(code=code):
                result = self.call("save_plan", project=str(self.root), plan=plan)
                self.assertTrue(result["isError"])
                text = result["content"][0]["text"]
                error = json.loads(text)
                self.assertEqual(error["category"], "invalid_arguments")
                self.assertEqual(error["code"], code)
                self.assertEqual(error["path"], path)
                self.assertNotIn("sensitive-unknown", text)
        self.assertEqual((self.root / "project.json").read_bytes(), before)

    def test_unknown_plan_exception_stays_private(self):
        self.create()
        before = (self.root / "project.json").read_bytes()
        with patch.object(core, "install_plan", side_effect=ValueError("private-provider-response-and-credential")):
            result = self.call("save_plan", project=str(self.root), plan=pipeline.demo_plan())
        self.assertTrue(result["isError"])
        self.assertNotIn("private-provider-response-and-credential", str(result))
        self.assertIn("原始异常未传给模型", result["content"][0]["text"])
        self.assertEqual((self.root / "project.json").read_bytes(), before)

    def test_new_project_paths_are_confined_and_symlinks_rejected(self):
        for root in ("relative", str(self.workspace / "elsewhere"), str(self.server.projects), str(self.server.projects / "nested/story")):
            self.assertTrue(self.call("create_project", project=root, novel_text="正文")["isError"])
        other = self.workspace / "elsewhere"
        other.mkdir()
        (self.server.projects / "link").symlink_to(other, target_is_directory=True)
        self.assertTrue(self.call("create_project", project=str(self.server.projects / "link"), novel_text="正文")["isError"])

    def test_explicit_existing_project_and_last_project_are_listed_without_scan(self):
        outside = self.workspace.parent / "explicit"
        unrelated = self.workspace.parent / "unrelated"
        core.create_project(outside, "指定项目")
        core.create_project(unrelated, "不得扫描")
        self.assertTrue(self.call("project_status", project=str(outside))["isError"])
        selected = mcp_server.MCPServer(self.workspace, project=outside)
        self.addCleanup(selected.close)
        listed = selected.call_tool("list_projects", {})
        self.assertEqual([p["project"] for p in listed["projects"]], [str(outside)])
        self.assertEqual(listed["projects_directory"], str(self.server.projects))
        with patch.object(mcp_server, "_last_project", return_value=outside):
            listed = self.value(self.call("list_projects"))
            self.assertEqual([p["project"] for p in listed["projects"]], [str(outside)])
            self.assertFalse(self.call("project_status", project=str(outside))["isError"])

    def test_existing_project_cannot_redirect_novel_or_plan_writes(self):
        self.create()
        outside = self.workspace.parent / "private.txt"
        outside.write_text("preserve", encoding="utf-8")
        destination = self.root / "plans/plan-v1.json"
        destination.symlink_to(outside)
        self.assertTrue(self.call("save_plan", project=str(self.root), plan=pipeline.demo_plan())["isError"])
        self.assertEqual(outside.read_text(), "preserve")
        self.assertEqual(core.load_project(self.root)["plan"]["version"], 0)
        original = self.root / "source/novel.txt"
        original.unlink()
        original.symlink_to(outside)
        self.assertTrue(self.call("run_production", project=str(self.root))["isError"])
        self.assertEqual(self.server.workers, {})

    def test_read_novel_requires_explicit_utf8_text_and_size_limit(self):
        source = self.workspace.parent / "novel.md"
        source.write_text("小说文本", encoding="utf-8")
        self.assertEqual(self.value(self.call("read_novel", source_path=str(source)))["novel_text"], "小说文本")
        source.write_bytes(b"\xff\xfe")
        self.assertTrue(self.call("read_novel", source_path=str(source))["isError"])
        source.write_bytes(b"x" * (mcp_server.MAX_NOVEL_BYTES + 1))
        self.assertTrue(self.call("read_novel", source_path=str(source))["isError"])
        self.assertTrue(self.call("read_novel", source_path=str(source.with_suffix(".json")))["isError"])
        self.assertTrue(self.call("read_novel", source_path="novel.md")["isError"])

    def test_local_directory_discovery_and_directory_novel_guidance(self):
        folder = self.workspace.parent / "御剑飞行"
        folder.mkdir()
        nested = folder / "剧本改编"
        nested.mkdir()
        source = nested / "第01集.md"
        source.write_text("合成剧本原文", encoding="utf-8")
        listed = self.value(self.call("list_directory", path=str(folder)))
        self.assertEqual(listed["entries"][0]["path"], str(nested))
        found = self.value(self.call("find_files", path=str(folder), pattern="*.md"))
        self.assertEqual(found["entries"][0]["path"], str(source))
        self.assertEqual(self.value(self.call("read_novel", source_path=found["entries"][0]["path"]))["novel_text"], "合成剧本原文")
        error = self.call("read_novel", source_path=str(folder))
        self.assertTrue(error["isError"])
        self.assertIn("list_directory", error["content"][0]["text"])
        self.assertIn("find_files", error["content"][0]["text"])
        for name in ("list_directory", "find_files"):
            self.assertTrue(mcp_server.TOOL_MAP[name]["annotations"]["readOnlyHint"])
        self.assertEqual(self.server.workers, {})
        self.assertEqual(list(self.server.projects.iterdir()), [])

    def test_novel_does_not_read_credential_text_or_symlinks(self):
        source = self.workspace.parent / "credentials.txt"
        source.write_text("synthetic-private-never-read", encoding="utf-8")
        ordinary = self.workspace.parent / "script.md"
        ordinary.symlink_to(source)
        for path in (source, ordinary):
            response = self.call("read_novel", source_path=str(path))
            self.assertTrue(response["isError"])
            self.assertNotIn("synthetic-private-never-read", str(response))

    def test_explicit_creative_directory_can_create_direct_child_and_resume(self):
        creative = self.workspace.parent / "桌面创作"
        creative.mkdir()
        script = creative / "剧本.txt"
        script.write_text("合成独立新项目剧本", encoding="utf-8")
        selected = mcp_server.MCPServer(self.workspace, workdir=creative)
        self.addCleanup(selected.close)
        listed = selected.call_tool("list_projects", {})
        self.assertEqual(listed["projects_directory"], str(self.server.projects))
        self.assertEqual(listed["creation_directory"], str(creative))
        root = creative / "Nolan-制作"
        state = selected.call_tool("create_project", {"project": str(root), "source_path": str(script)})
        self.assertEqual(state["project"], str(root))
        self.assertFalse(state["budget"]["approved"])
        self.assertEqual(state["plan"]["status"], "pending")
        self.assertEqual(script.read_text(), "合成独立新项目剧本")
        self.assertFalse((creative / "project.json").exists())
        self.assertFalse((self.server.projects / root.name).exists())
        projects = selected.call_tool("list_projects", {})["projects"]
        self.assertEqual([item["project"] for item in projects], [str(root)])
        resumed = mcp_server.MCPServer(self.workspace, workdir=creative)
        self.addCleanup(resumed.close)
        self.assertEqual(resumed.call_tool("project_status", {"project": str(root)})["id"], state["id"])

    def test_explicit_empty_creative_directory_can_be_the_project(self):
        creative = self.workspace.parent / "空创作目录"
        creative.mkdir()
        selected = mcp_server.MCPServer(self.workspace, workdir=creative)
        self.addCleanup(selected.close)
        selected.call_tool("create_project", {"project": str(creative), "novel_text": "合成原文"})
        self.assertEqual((creative / "source/novel.txt").read_text(), "合成原文")
        self.assertEqual([item["project"] for item in selected.call_tool("list_projects", {})["projects"]], [str(creative)])

    def test_creative_directory_rejects_overwrite_deeper_paths_and_symlink_escape(self):
        creative = self.workspace.parent / "创作目录"
        creative.mkdir()
        keep = creative / "剧本.md"
        keep.write_text("合成不可覆盖", encoding="utf-8")
        selected = mcp_server.MCPServer(self.workspace, workdir=creative)
        self.addCleanup(selected.close)
        outside = self.workspace.parent / "outside"
        outside.mkdir()
        (creative / "linked").symlink_to(outside, target_is_directory=True)
        (creative / "managed-link").symlink_to(self.server.projects, target_is_directory=True)
        for root in (creative, outside / "new", creative / "deep/new", creative / "linked/new", creative / "linked", creative / "managed-link/new"):
            with self.subTest(root=root), self.assertRaises(mcp_server.ToolError):
                selected.call_tool("create_project", {"project": str(root), "novel_text": "合成"})
        self.assertEqual(keep.read_text(), "合成不可覆盖")
        self.assertEqual(list(outside.iterdir()), [])
        with self.assertRaises(mcp_server.ToolError):
            mcp_server.MCPServer(self.workspace, workdir=creative / "linked")

    def test_creative_listing_does_not_recursively_scan_or_move_old_projects(self):
        self.create()
        before = (self.root / "project.json").read_bytes()
        creative = self.workspace.parent / "创作目录"
        creative.mkdir()
        nested = creative / "资料/深层项目"
        core.create_project(nested, "合成深层项目")
        unrelated = self.workspace.parent / "其他项目"
        core.create_project(unrelated, "合成其他项目")
        (creative / "linked-project").symlink_to(unrelated, target_is_directory=True)
        selected = mcp_server.MCPServer(self.workspace, project=unrelated, workdir=creative)
        self.addCleanup(selected.close)
        listed = selected.call_tool("list_projects", {})
        self.assertEqual({item["project"] for item in listed["projects"]}, {str(self.root), str(unrelated)})
        self.assertEqual((self.root / "project.json").read_bytes(), before)
        self.assertTrue(self.root.exists())

    def test_source_path_creation_preserves_original(self):
        source = self.workspace.parent / "original.txt"
        source.write_text("不变的原稿", encoding="utf-8")
        self.value(self.call("create_project", project=str(self.root), source_path=str(source)))
        self.assertEqual(source.read_text(), "不变的原稿")
        self.assertEqual((self.root / "source/novel.txt").read_text(), "不变的原稿")

    def test_novel_pagination_preserves_full_source_on_creation(self):
        source = self.workspace.parent / "long.txt"
        text = "短剧故事" * 7000
        source.write_text(text, encoding="utf-8")
        first = self.value(self.call("read_novel", source_path=str(source)))
        self.assertEqual(first["total_chars"], len(text))
        self.assertEqual(first["next_offset"], 12000)
        self.assertTrue(first["truncated"])
        second = self.value(self.call("read_novel", source_path=str(source), offset=12000, max_chars=20000))
        self.assertIsNone(second["next_offset"])
        self.assertFalse(second["truncated"])
        self.assertEqual(first["novel_text"] + second["novel_text"], text)
        for args in ({"offset": True}, {"offset": len(text) + 1}, {"max_chars": 20001}, {"max_chars": 0}):
            self.assertTrue(self.call("read_novel", source_path=str(source), **args)["isError"])
        self.value(self.call("create_project", project=str(self.root), source_path=str(source)))
        self.assertEqual((self.root / "source/novel.txt").read_text(), text)

    def test_error_categories_are_actionable_without_echoing_secret_values(self):
        self.create()
        for message, category in (("未配置 ARK_API_KEY=secret-value", "credentials"),
                                  ("剩余预算不足 secret-value", "budget"),
                                  ("前集尚未人工批准 secret-value", "approval"),
                                  ("当前模型不支持 secret-value", "model_configuration")):
            with core.transaction(self.root) as state:
                state["last_error"] = message
            result = self.value(self.call("project_status", project=str(self.root)))
            self.assertEqual(result["error_detail"]["category"], category)
            self.assertNotIn("secret-value", json.dumps(result))

    def test_status_redacts_settings_jobs_events_and_raw_exceptions(self):
        self.create()
        with core.transaction(self.root) as state:
            state["settings"]["ark"]["api_key"] = "private-test-key"
            state["last_error"] = "private-test-key"
            state["events"].append({"private": "private-test-key"})
        result = self.call("project_status", project=str(self.root))
        self.assertNotIn("private-test-key", json.dumps(result))
        state = self.value(result)
        for key in ("settings", "jobs", "events", "source"):
            self.assertNotIn(key, state)
        with patch.object(core, "configure", side_effect=ValueError("private-test-key")):
            result = self.call("set_creative_brief", project=str(self.root), brief="画面")
        self.assertTrue(result["isError"])
        self.assertNotIn("private-test-key", json.dumps(result))

    def test_creative_brief_only_changes_that_setting(self):
        self.create()
        before = core.load_project(self.root)
        self.value(self.call("set_creative_brief", project=str(self.root), brief="轻喜剧，竖屏"))
        after = core.load_project(self.root)
        expected = copy.deepcopy(before["settings"])
        expected["creative_brief"] = "轻喜剧，竖屏"
        self.assertEqual(after["settings"], expected)
        self.assertEqual(after["budget"], before["budget"])
        self.assertEqual(after["plan"], before["plan"])

    def test_revision_requires_fresh_version_and_does_not_start(self):
        self.create()
        self.save_plan()
        before = (self.root / "project.json").read_bytes()
        result = self.call("request_revision", project=str(self.root), kind="plan", id="plan", version=0, feedback="调整")
        self.assertTrue(result["isError"])
        self.assertEqual((self.root / "project.json").read_bytes(), before)
        state = self.value(self.call("request_revision", project=str(self.root), kind="plan", id="plan", version=1, feedback="增加悬念"))
        self.assertEqual(state["plan"]["status"], "rejected")
        self.assertFalse(state["background"]["running"])
        self.assertEqual(core.load_project(self.root)["jobs"], {})

    def test_caption_request_passes_explicit_version_and_does_not_start(self):
        self.create()
        self.save_plan()
        with patch.object(core, "request_caption_fix", return_value=None) as request:
            self.value(self.call("correct_captions", project=str(self.root), id="E001", version=7, correction="改一个错字"))
        request.assert_called_once_with(self.root, "E001", 7, "改一个错字")
        self.assertEqual(self.server.workers, {})

    def _synthetic_segment_proposal(self, root, episode_id, groups):
        with core.transaction(root) as state:
            episode = core._find(state, "episode", episode_id)
            episode["segment_proposal"] = {
                "id": "synthetic-proposal", "status": "candidate",
                "source_ids": [identity for group in groups for identity in group],
                "created_at": core.now(), "shots": [], "private": "SYNTHETIC-PRIVATE-PROPOSAL",
            }

    def test_segment_proposal_opens_review_without_installing_plan_or_generating(self):
        self.create()
        self.save_plan()
        core.set_paused(self.root, True)
        before = core.load_project(self.root)
        groups = [["E001-S001", "E001-S002"]]

        def open_saved(root):
            self.assertEqual(core.load_project(root)["episodes"][0]["segment_proposal"]["status"], "candidate")

        with patch("norlan.regroup.propose_segments", side_effect=self._synthetic_segment_proposal) as propose, \
             patch.object(self.server, "_open_review", side_effect=open_saved) as review, \
             patch.object(core, "install_plan") as install, patch.object(pipeline, "run") as run:
            result = self.value(self.call("propose_segments", project=str(self.root), episode_id="E001", groups=groups))
        propose.assert_called_once_with(self.root, "E001", groups)
        review.assert_called_once_with(self.root)
        install.assert_not_called()
        run.assert_not_called()
        proposal = result["episodes"][0]["segment_proposal"]
        self.assertEqual(proposal["status"], "candidate")
        self.assertNotIn("SYNTHETIC-PRIVATE-PROPOSAL", json.dumps(result))
        self.assertIn("用户本人", result["message"])
        after = core.load_project(self.root)
        for key in ("assets", "jobs", "budget", "plan"):
            self.assertEqual(after[key], before[key])
        self.assertEqual(after["episodes"][0]["shots"], before["episodes"][0]["shots"])
        self.assertTrue(after["paused"])
        self.assertEqual(self.server.workers, {})

    def test_segment_proposal_review_failure_preserves_saved_candidate_and_allows_retry(self):
        self.create()
        self.save_plan()
        with patch("norlan.regroup.propose_segments", side_effect=self._synthetic_segment_proposal) as propose, \
             patch.object(self.server, "_open_review", side_effect=RuntimeError("SYNTHETIC-PRIVATE-ERROR")):
            result = self.value(self.call("propose_segments", project=str(self.root), episode_id="E001",
                                          groups=[["E001-S001"], ["E001-S002"]]))
        self.assertEqual(propose.call_count, 1)
        self.assertEqual(result["episodes"][0]["segment_proposal"]["status"], "candidate")
        self.assertEqual(result["review_error"], mcp_server.REVIEW_ERROR_MESSAGE)
        self.assertIn("open_review", result["operator_actions"]["review_retry"])
        self.assertIsNone(result["background"]["error"])
        self.assertNotIn("SYNTHETIC-PRIVATE-ERROR", json.dumps(result))

    def test_invalid_segment_proposal_does_not_open_browser_or_bypass_core_gate(self):
        self.create()
        before = (self.root / "project.json").read_bytes()
        with patch("norlan.regroup.propose_segments", side_effect=core.GateError("尚有调用在执行")) as propose, \
             patch.object(self.server, "_open_review") as review:
            result = self.call("propose_segments", project=str(self.root), episode_id="E001", groups=[["S1", "S2"]])
            self.assertTrue(result["isError"])
            self.assertEqual(propose.call_count, 1)
            for groups in ([], [[]], [["S1", "S2", "S3", "S4"]], ["S1"], [["../secret"]]):
                self.assertTrue(self.call("propose_segments", project=str(self.root), episode_id="E001", groups=groups)["isError"])
            self.assertEqual(propose.call_count, 1)
            review.assert_not_called()
        self.assertEqual((self.root / "project.json").read_bytes(), before)

    def test_pending_segment_proposal_cannot_be_resumed_or_generated_by_model(self):
        self.create()
        self.save_plan()
        core.set_paused(self.root, True)
        self._synthetic_segment_proposal(self.root, "E001", [["E001-S001", "E001-S002"]])
        before = (self.root / "project.json").read_bytes()
        with patch.object(core, "set_paused") as pause, patch.object(pipeline, "run") as run:
            result = self.call("run_production", project=str(self.root))
            self.assertTrue(result["isError"])
            self.assertIn("open_review", result["content"][0]["text"])
            pause.assert_not_called()
            run.assert_not_called()
        self.assertEqual(self.server.workers, {})
        self.assertEqual((self.root / "project.json").read_bytes(), before)
        self.assertTrue(core.load_project(self.root)["paused"])

    def test_background_run_returns_once_pause_and_close_preserve_project(self):
        self.create()
        started, release = threading.Event(), threading.Event()

        def hold(root):
            with core.run_lock(root):
                started.set()
                release.wait(4)

        with patch.object(pipeline, "run", side_effect=hold) as run:
            try:
                self.assertTrue(self.value(self.call("run_production", project=str(self.root)))["started"])
                self.assertTrue(started.wait(2))
                self.assertFalse(self.value(self.call("run_production", project=str(self.root)))["started"])
                self.assertTrue(self.value(self.call("project_status", project=str(self.root)))["background"]["running"])
                self.assertTrue(self.value(self.call("pause_production", project=str(self.root)))["paused"])
                self.server.close()
                self.assertTrue(core.load_project(self.root)["paused"])
                self.assertTrue((self.root / "source/novel.txt").is_file())
                self.assertEqual(run.call_count, 1)
            finally:
                release.set()
                self.server.workers[self.root].join(2)

    def test_background_run_respects_plan_approval_and_makes_no_online_calls(self):
        self.create()
        self.save_plan()
        with patch.object(pipeline, "_brain", side_effect=AssertionError("online forbidden")), patch.object(pipeline, "_assets", side_effect=AssertionError("approval forbidden")):
            self.value(self.call("run_production", project=str(self.root)))
            self.server.workers[self.root].join(2)
        state = core.load_project(self.root)
        self.assertEqual(state["status"], "awaiting_plan")
        self.assertEqual(state["plan"]["status"], "candidate")
        self.assertEqual(state["jobs"], {})

    def test_cross_process_running_lock_does_not_resume_other_worker(self):
        self.create()
        core.set_paused(self.root, True)
        with core.run_lock(self.root):
            self.assertTrue(self.call("run_production", project=str(self.root))["isError"])
            self.assertTrue(self.call("save_plan", project=str(self.root), plan=pipeline.demo_plan())["isError"])
            self.assertTrue(self.call("set_creative_brief", project=str(self.root), brief="变更")["isError"])
        state = core.load_project(self.root)
        self.assertTrue(state["paused"])
        self.assertEqual(state["settings"]["creative_brief"], "")

    def test_background_production_cannot_reserve_beyond_user_budget(self):
        self.create()
        self.save_plan()
        core.approve(self.root, "plan", "plan", 1)
        core.configure(self.root, limit=.5)

        def attempt_paid_job(root):
            core.reserve(root, "test-not-submitted", "image", "hero", 1)
            self.fail("budget must stop this before any paid submission")

        with patch.object(pipeline, "_assets", side_effect=attempt_paid_job):
            self.value(self.call("run_production", project=str(self.root)))
            self.server.workers[self.root].join(2)
        state = core.load_project(self.root)
        self.assertEqual(state["status"], "blocked")
        self.assertEqual(state["budget"]["limit"], .5)
        self.assertEqual(state["budget"]["spent"], 0)
        self.assertEqual(state["jobs"], {})

    def test_open_review_reuses_local_server_and_close_stops_it(self):
        self.create()
        with patch.object(mcp_server.webbrowser, "open", return_value=False):
            first = self.value(self.call("open_review", project=str(self.root)))
            second = self.value(self.call("open_review", project=str(self.root)))
        self.assertEqual(first["url"], second["url"])
        self.assertTrue(first["url"].startswith("http://127.0.0.1:"))
        self.assertNotIn("token", first)
        server, thread = self.server.reviews[self.root]
        self.assertTrue(thread.is_alive())
        self.server.close()
        self.assertFalse(thread.is_alive())
        self.assertEqual(server.fileno(), -1)


if __name__ == "__main__":
    unittest.main()
