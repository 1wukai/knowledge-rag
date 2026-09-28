from contextlib import asynccontextmanager
from datetime import datetime, timezone
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

import httpx

from interview_rag.api.app import create_app
from interview_rag.application.cleanup import cleanup_generations
from interview_rag.application.snapshot import compute_manifest_hash
from interview_rag.config import Settings
from interview_rag.manifest import Manifest, ManifestDocument, file_sha256, load_manifest, save_manifest, scan_candidates

PROJECT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("console_launcher", PROJECT / "scripts/serve.py")
launcher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(launcher)


class ConsoleAPI(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=PROJECT / ".venv", prefix="console-test-")
        self.root = Path(self.temp.name)
        notes = self.root / "corpus/backend"
        notes.mkdir(parents=True)
        self.source = notes / "transactions.md"
        self.source.write_text("# 事务\n\n事务的隔离性。", encoding="utf-8")
        self.manifest = Manifest(corpus_id="agent-interview", documents=[ManifestDocument(
            doc_id="test-doc", path="corpus/backend/transactions.md", title="事务", category="backend", hash=file_sha256(self.source)
        )])
        self.path = self.root / "manifest.json"
        save_manifest(self.path, self.manifest)
        self.settings = Settings(_env_file=None, workspace_root=self.root, manifest_path=self.path,
                                 runtime_settings_path=self.root / "runtime.json", index_state_path=self.root / "state.json", qdrant_mode="local")
        self.app = create_app(self.settings)
        self.client = httpx.AsyncClient(transport=httpx.ASGITransport(app=self.app), base_url="http://test")

    async def asyncTearDown(self):
        await self.client.aclose()
        self.temp.cleanup()

    async def test_local_capabilities_and_invalid_inputs(self):
        response = await self.client.get("/v1/system/status")
        self.assertEqual(response.json()["mode"], "local")
        self.assertFalse(response.json()["features"]["conversations"])
        self.assertEqual((await self.client.get("/v1/ingestions")).status_code, 400)
        self.assertEqual((await self.client.get("/v1/ingestions?limit=0")).status_code, 422)
        self.assertEqual((await self.client.post("/v1/chat", json={"question":"x", "top_k":0})).status_code, 422)
        self.assertEqual((await self.client.post("/v1/index-generations/cleanup?retain=-1")).status_code, 422)

    async def test_edit_category_disable_and_refresh(self):
        before = compute_manifest_hash(self.manifest)
        response = await self.client.patch("/v1/corpus/documents/test-doc", json={"title":"事务笔记", "category":"数据库", "enabled":False})
        self.assertEqual(response.status_code, 200)
        manifest = load_manifest(self.path)
        self.assertEqual(manifest.documents[0].title, "事务笔记")
        self.assertNotEqual(compute_manifest_hash(manifest), before)
        self.assertFalse(manifest.documents[0].enabled)
        self.source.write_text("# 事务\n\n内容更新。", encoding="utf-8")
        updated = await self.client.patch("/v1/corpus/documents/test-doc", json={"refresh":True})
        self.assertEqual(updated.json()["hash"], file_sha256(self.source))
        self.assertEqual((await self.client.patch("/v1/corpus/documents/test-doc", json={"title":"  "})).status_code, 400)
        self.assertEqual((await self.client.patch("/v1/corpus/documents/missing", json={"category":"x"})).status_code, 404)

    async def test_scan_and_add_are_scoped(self):
        notes = self.source.parent
        (notes / "new.md").write_text("# 新笔记\n\n缓存。", encoding="utf-8")
        (notes / "README.md").write_text("# 索引", encoding="utf-8")
        outside = self.root / "private.md"
        outside.write_text("private", encoding="utf-8")
        (notes / "escape.md").symlink_to(outside)
        candidates = scan_candidates(self.manifest, self.root)
        self.assertEqual([c["path"] for c in candidates], ["corpus/backend/new.md"])
        added = await self.client.post("/v1/corpus/add", json={"doc_ids":["corpus/backend/new.md"]})
        self.assertEqual(added.status_code, 200)
        self.assertEqual(len(load_manifest(self.path).documents), 2)
        self.assertEqual((await self.client.post("/v1/corpus/add", json={"doc_ids":["private.md"]})).status_code, 400)

    async def test_server_corpus_uses_frozen_members_not_local_file(self):
        self.settings.qdrant_mode = "server"
        self.settings.index_state_path.write_text(json.dumps({"documents":[{"doc_id":"wrong-local"}]}))
        generation = SimpleNamespace(generation_id="g1", collection="gen_g1", created_at=datetime.now(timezone.utc), embedding_model="mock", embedding_dim=4, chunk_count=1, manifest_hash="hash")
        session = AsyncMock()
        session.scalar.side_effect = [SimpleNamespace(active_generation_id="g1"), generation, SimpleNamespace(manifest_json=self.manifest.model_dump_json())]
        session.scalars.return_value = SimpleNamespace(all=lambda:["test-doc"])
        @asynccontextmanager
        async def factory(): yield session
        with patch("interview_rag.api.app.get_database", return_value=SimpleNamespace(session_factory=factory)):
            response = await self.client.get("/v1/corpus")
        self.assertEqual(response.json()["index"]["generation_id"], "g1")
        self.assertEqual(response.json()["index"]["documents"][0]["doc_id"], "test-doc")
        self.assertTrue(response.json()["documents"][0]["indexed"])

    async def test_generation_list_skips_wide_json_and_filesort(self):
        from sqlalchemy.dialects import mysql

        self.settings.qdrant_mode = "server"
        session = AsyncMock()
        session.scalar.return_value = SimpleNamespace(active_generation_id="g1", publication_revision=7)
        row = SimpleNamespace(generation_id="g1", collection="gen_g1", status="active", chunk_count=113,
                              embedding_model="mock", embedding_dim=4,
                              created_at=datetime(2026, 9, 10, tzinfo=timezone.utc), activated_at=None)
        session.execute.return_value = SimpleNamespace(all=lambda: [row])

        @asynccontextmanager
        async def factory():
            yield session

        with patch("interview_rag.api.app.get_database", return_value=SimpleNamespace(session_factory=factory)):
            response = await self.client.get("/v1/index-generations")
        self.assertEqual(response.status_code, 200)
        sql = str(session.execute.await_args.args[0].compile(dialect=mysql.dialect()))
        self.assertNotIn("bm25_stats", sql)
        self.assertIn("ORDER BY index_generations.id DESC", sql)
        payload = response.json()
        self.assertEqual(payload["active"], "g1")
        self.assertEqual(payload["revision"], 7)
        self.assertEqual(payload["generations"][0]["generation_id"], "g1")
        self.assertEqual(payload["generations"][0]["created_at"], "2026-09-10T00:00:00+00:00")

    async def test_evidence_count_is_respected(self):
        from interview_rag.application import query_generation
        from interview_rag.runtime_settings import defaults

        runtime = defaults()
        runtime.rerank = None
        generation = SimpleNamespace(generation_id="g1", collection="gen_g1", embedding_model=runtime.embedding.model, embedding_dim=2, chunk_count=3, bm25_stats=None)
        hits = [{"id": str(i), "text":"内容", "doc_id":f"d{i}", "title":"标题", "start_line":1, "end_line":2, "hash":"hash"} for i in range(3)]
        with patch.object(query_generation, "store_for_settings"), patch.object(query_generation, "embed_texts", new=AsyncMock(return_value=[[1.0, 0.0]])), patch.object(query_generation, "hybrid_search", new=AsyncMock(return_value=SimpleNamespace(hits=hits, dense_top_score=0.9))):
            outcome = await query_generation._retrieve(self.settings, runtime, None, generation, "问题", 1, 0.4)
        self.assertEqual(len(outcome.hits), 1)
        self.assertEqual(len(outcome.citations), 1)

    async def test_cleanup_never_deletes_pointer_with_stale_status(self):
        session = AsyncMock()
        session.scalar.return_value = SimpleNamespace(active_generation_id="active")
        rows = [SimpleNamespace(status="ready", generation_id="active", collection="keep"), SimpleNamespace(status="ready", generation_id="old", collection="old")]
        session.execute.return_value = SimpleNamespace(scalars=lambda:SimpleNamespace(all=lambda:rows))
        store = MagicMock()
        result = await cleanup_generations(session, store, "agent-interview", retain=0)
        self.assertEqual(result["deleted_generations"], ["old"])
        store.delete_collection.assert_called_once_with("old")


class LauncherTests(unittest.TestCase):
    def test_server_preflight_then_migration_then_owned_services(self):
        cwd = Path.cwd()
        try:
            with patch.object(sys, "argv", ["serve.py"]), patch.object(launcher, "check_port"), patch.object(launcher, "check_infra", new=AsyncMock()) as check, patch.object(launcher.subprocess, "run") as migrate, patch.object(launcher, "supervise", return_value=0) as run:
                self.assertEqual(launcher.main(), 0)
                check.assert_awaited_once()
                self.assertEqual(migrate.call_args.args[0][-2:], ["upgrade", "head"])
                commands, env = run.call_args.args
                self.assertEqual(len(commands), 3)
                self.assertEqual(env["IR_QDRANT_MODE"], "server")
        finally:
            os.chdir(cwd)

    def test_failed_preflight_does_not_migrate_or_launch(self):
        cwd = Path.cwd()
        try:
            with patch.object(sys, "argv", ["serve.py"]), patch.object(launcher, "check_port"), patch.object(launcher, "check_infra", new=AsyncMock(side_effect=RuntimeError("Unavailable"))), patch.object(launcher.subprocess, "run") as migrate, patch.object(launcher, "supervise") as run:
                with self.assertRaises(RuntimeError): launcher.main()
                migrate.assert_not_called()
                run.assert_not_called()
        finally:
            os.chdir(cwd)

    def test_local_mode_skips_database_migration(self):
        cwd = Path.cwd()
        try:
            with patch.object(sys, "argv", ["serve.py", "--local", "--mock"]), patch.object(launcher, "check_port"), patch.object(launcher, "check_infra", new=AsyncMock()) as check, patch.object(launcher.subprocess, "run") as migrate, patch.object(launcher, "supervise", return_value=0) as run:
                self.assertEqual(launcher.main(), 0)
                check.assert_not_awaited()
                for call in migrate.call_args_list:
                    self.assertNotIn("alembic", call.args[0])
                self.assertEqual(run.call_args.args[1]["IR_QDRANT_MODE"], "local")
        finally:
            os.chdir(cwd)

    def test_supervisor_only_signals_owned_groups(self):
        first, second = MagicMock(pid=111), MagicMock(pid=222)
        first.poll.return_value = 3
        second.poll.return_value = None
        with patch.object(launcher.subprocess, "Popen", side_effect=[first, second]), patch.object(launcher.os, "killpg") as kill:
            result = launcher.supervise([("first", ["cmd"], PROJECT), ("second", ["cmd"], PROJECT)], dict(os.environ))
        self.assertEqual(result, 3)
        self.assertEqual({call.args[0] for call in kill.call_args_list}, {111, 222})
        self.assertEqual(kill.call_args_list[0].args[1], signal.SIGTERM)


if __name__ == "__main__":
    unittest.main()
