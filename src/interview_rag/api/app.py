"""FastAPI 应用：健康检查、provider 设置、入库、问答，并托管 React 构建产物。"""

from __future__ import annotations

import json
import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from sqlalchemy import select

from interview_rag.application import query_generation
from interview_rag.application.ingest import IngestError, IngestResult, ingest_documents
from interview_rag.application.ingestion_service import get_job, submit_ingestion
from interview_rag.application.index_state import load_state
from interview_rag.application.publish import PublishConflict, publish_generation, rollback
from interview_rag.application.query import IncompatibleIndexError, NoIndexError, answer_question
from interview_rag.config import Settings, get_settings, resolve_path
from interview_rag.logging import configure_logging
from interview_rag.manifest import load_manifest, resolve_manifest_path
from interview_rag.providers import ProviderError, chat, embed_texts, rerank
from interview_rag.runtime_settings import RuntimeSettings, load as load_runtime, save as save_runtime
from interview_rag.storage.database import get_database
from interview_rag.storage.models import Corpus, IndexGeneration

logger = logging.getLogger("interview_rag.api")

# 项目根 = src/interview_rag/api/app.py 的上三级；React 构建产物在 web/dist。
_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_WEB_DIST = _PROJECT_ROOT / "web" / "dist"


class IngestRequest(BaseModel):
    doc_ids: list[str]


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=16000)
    conversation_id: str | None = None
    top_k: int = Field(default=5, ge=1, le=20)
    min_score: float | None = Field(default=None, ge=0, le=1)
    stream: bool = True


class DocumentUpdate(BaseModel):
    title: str | None = Field(default=None, min_length=1, max_length=256)
    category: str | None = Field(default=None, max_length=64)
    enabled: bool | None = None
    refresh: bool = False


class ActivateRequest(BaseModel):
    expected_generation_id: str | None = None
    expected_revision: int


def _is_server_mode(settings: Settings) -> bool:
    return settings.qdrant_mode == "server"


def _sse_format(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


@asynccontextmanager
async def lifespan(app: FastAPI):
    configure_logging(app.state.settings.log_level, app.state.settings.log_json)
    logger.info("api starting", extra={"environment": app.state.settings.environment})
    yield
    logger.info("api stopped")


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    app = FastAPI(title=settings.app_name, version="0.1.0", lifespan=lifespan)
    app.state.settings = settings
    app.state.runtime = load_runtime(resolve_path(settings.runtime_settings_path))

    @app.get("/health/live")
    async def live() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/health/ready")
    async def ready() -> dict[str, str | bool]:
        return {"status": "ok", "dependencies_checked": False}

    @app.get("/v1/settings")
    async def get_settings_api() -> RuntimeSettings:
        return app.state.runtime

    @app.put("/v1/settings")
    async def put_settings_api(update: RuntimeSettings) -> RuntimeSettings:
        if not update.embedding.base_url or not update.generation.base_url:
            raise HTTPException(status_code=400, detail="base_url 不能为空")
        save_runtime(resolve_path(settings.runtime_settings_path), update)
        app.state.runtime = update
        return update

    @app.post("/v1/providers/test")
    async def test_providers() -> dict:
        result: dict = {}
        try:
            vectors = await embed_texts(app.state.runtime.embedding, ["连通性测试"])
            result["embedding"] = {"ok": True, "message": f"成功，维度 {len(vectors[0])}"}
        except Exception as exc:  # noqa: BLE001
            result["embedding"] = {"ok": False, "message": str(exc)}
        try:
            text = await chat(
                app.state.runtime.generation,
                [{"role": "user", "content": "你好，请回复 ok"}],
            )
            result["generation"] = {"ok": True, "message": text[:80]}
        except Exception as exc:  # noqa: BLE001
            result["generation"] = {"ok": False, "message": str(exc)}
        if app.state.runtime.rerank and app.state.runtime.rerank.base_url:
            try:
                rerank_results = await rerank(
                    app.state.runtime.rerank, "测试", ["连通性测试", "重排测试"]
                )
                result["rerank"] = {"ok": True, "message": f"成功，{len(rerank_results)} 条"}
            except Exception as exc:  # noqa: BLE001
                result["rerank"] = {"ok": False, "message": str(exc)}
        return result

    async def current_index() -> dict | None:
        if not _is_server_mode(settings):
            return load_state(resolve_path(settings.index_state_path))
        from interview_rag.storage.models import CorpusSnapshot, GenerationChunk, Chunk

        db = get_database(settings.database_url)
        async with db.session_factory() as session:
            corpus = await session.scalar(select(Corpus).where(Corpus.corpus_id == "agent-interview"))
            if not corpus or not corpus.active_generation_id:
                return None
            generation = await session.scalar(select(IndexGeneration).where(
                IndexGeneration.generation_id == corpus.active_generation_id
            ))
            if not generation:
                raise HTTPException(status_code=503, detail="已发布索引记录缺失")
            doc_ids = set((await session.scalars(
                select(Chunk.doc_id).join(GenerationChunk, GenerationChunk.chunk_id == Chunk.chunk_id)
                .where(GenerationChunk.generation_id == generation.generation_id).distinct()
            )).all())
            snapshot = await session.scalar(select(CorpusSnapshot).where(
                CorpusSnapshot.manifest_hash == generation.manifest_hash
            ))
            documents = json.loads(snapshot.manifest_json)["documents"] if snapshot else []
            return {
                "generation_id": generation.generation_id,
                "collection": generation.collection,
                "generated_at": generation.created_at.isoformat(),
                "embedding_model": generation.embedding_model,
                "embedding_dim": generation.embedding_dim,
                "chunk_count": generation.chunk_count,
                "documents": [d for d in documents if d["doc_id"] in doc_ids],
            }

    @app.get("/v1/system/status")
    async def system_status() -> dict:
        server = _is_server_mode(settings)
        state = await current_index()
        return {
            "mode": "server" if server else "local",
            "features": {name: server for name in (
                "conversations", "async_ingestion", "answer_history", "index_management"
            )},
            "active_generation_id": state.get("generation_id") if state else None,
        }

    @app.get("/v1/corpus")
    async def corpus() -> dict:
        manifest = load_manifest(resolve_manifest_path(settings))
        state = await current_index()
        indexed = {d["doc_id"] for d in state.get("documents", [])} if state else set()
        documents = [
            {
                "doc_id": doc.doc_id,
                "path": doc.path,
                "title": doc.title,
                "hash": doc.hash,
                "category": doc.category,
                "enabled": doc.enabled,
                "indexed": doc.doc_id in indexed,
            }
            for doc in manifest.documents
        ]
        return {"documents": documents, "index": state}

    @app.get("/v1/corpus/candidates")
    async def corpus_candidates() -> dict:
        """扫描允许目录下未纳入的 MD、TXT、PDF 与 Word 文件。"""
        from interview_rag.manifest import scan_candidates

        manifest = load_manifest(resolve_manifest_path(settings))
        candidates = scan_candidates(manifest, settings.workspace_root)
        return {"candidates": candidates}

    @app.post("/v1/corpus/add")
    async def corpus_add(req: IngestRequest) -> dict:
        """把候选文件（按 path）纳入 manifest，返回更新后的文档列表。"""
        from interview_rag.manifest import (
            add_documents,
            scan_candidates,
            save_manifest,
        )

        manifest_path = resolve_manifest_path(settings)
        manifest = load_manifest(manifest_path)
        candidates = scan_candidates(manifest, settings.workspace_root)
        by_path = {c["path"]: c for c in candidates}
        entries = [by_path[p] for p in req.doc_ids if p in by_path]
        if not entries:
            raise HTTPException(status_code=400, detail="没有可纳入的候选文件")
        try:
            manifest = add_documents(manifest, entries)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        save_manifest(manifest_path, manifest)
        return {
            "added": [e["path"] for e in entries],
            "documents": [
                {"doc_id": d.doc_id, "path": d.path, "title": d.title, "category": d.category}
                for d in manifest.documents
            ],
        }

    @app.post("/v1/ingest")
    async def ingest(req: IngestRequest) -> IngestResult:
        try:
            return await ingest_documents(settings, app.state.runtime, req.doc_ids)
        except IngestError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except ProviderError as exc:
            raise HTTPException(status_code=502, detail=f"Embedding 服务异常：{exc}") from exc
        except Exception as exc:  # noqa: BLE001
            logger.exception("ingest 未预期异常")
            raise HTTPException(status_code=500, detail=f"入库失败：{exc}") from exc

    @app.post("/v1/ingestions")
    async def submit_ingestions(req: IngestRequest) -> dict:
        """提交异步入库任务。server 模式走 job+outbox；local 模式降级为同步执行。"""
        if not _is_server_mode(settings):
            result = await ingest(req)
            return {
                "job_id": f"local-{int(time.time())}",
                "status": "succeeded",
                "chunk_count": result.chunk_count,
                "embedding_model": result.embedding_model,
                "embedding_dim": result.embedding_dim,
                "reused": result.reused,
                "removed": result.removed,
            }
        db = get_database(settings.database_url)
        try:
            async with db.session_factory() as session:
                return await submit_ingestion(session, settings, req.doc_ids)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:  # noqa: BLE001
            logger.exception("submit ingestion 未预期异常")
            raise HTTPException(status_code=500, detail=f"提交失败：{exc}") from exc

    @app.patch("/v1/corpus/documents/{doc_id}")
    async def update_document(doc_id: str, req: DocumentUpdate) -> dict:
        from interview_rag.manifest import save_manifest, refresh_document

        path = resolve_manifest_path(settings)
        manifest = load_manifest(path)
        doc = next((d for d in manifest.documents if d.doc_id == doc_id), None)
        if doc is None:
            raise HTTPException(status_code=404, detail="文档不存在")
        if req.title is not None:
            if not req.title.strip():
                raise HTTPException(status_code=400, detail="标题不能为空")
            doc.title = req.title.strip()
        if req.category is not None:
            doc.category = req.category.strip()
        if req.enabled is not None:
            doc.enabled = req.enabled
        if req.refresh:
            try:
                refresh_document(doc, settings.workspace_root)
            except (ValueError, OSError) as exc:
                raise HTTPException(status_code=400, detail="无法刷新文档，请检查源文件路径、编码和大小") from exc
        save_manifest(path, manifest)
        return doc.model_dump()

    @app.get("/v1/ingestions")
    async def list_ingestions(limit: int = Query(default=50, ge=1, le=200)) -> dict:
        from interview_rag.application.ingestion_service import list_jobs

        if not _is_server_mode(settings):
            raise HTTPException(status_code=400, detail="local 模式无异步任务")
        async with get_database(settings.database_url).session_factory() as session:
            return {"jobs": await list_jobs(session, limit)}

    @app.get("/v1/ingestions/{job_id}")
    async def get_ingestion(job_id: str) -> dict:
        if not _is_server_mode(settings):
            raise HTTPException(status_code=404, detail="local 模式无异步任务")
        db = get_database(settings.database_url)
        async with db.session_factory() as session:
            job = await get_job(session, job_id)
            if not job:
                raise HTTPException(status_code=404, detail="任务不存在")
            return job

    @app.get("/v1/index-generations")
    async def list_generations() -> dict:
        if not _is_server_mode(settings):
            return {"generations": [], "active": None, "revision": None}
        from sqlalchemy import select

        db = get_database(settings.database_url)
        async with db.session_factory() as session:
            corpus = await session.scalar(
                select(Corpus).where(Corpus.corpus_id == "agent-interview")
            )
            # 不取 bm25_stats（单行 JSON 可达数 MB），并按主键倒序走 (corpus_id, id)
            # 索引反向扫描；否则宽行 filesort 会触发 MySQL 1038 out of sort memory。
            rows = (
                await session.execute(
                    select(
                        IndexGeneration.generation_id,
                        IndexGeneration.collection,
                        IndexGeneration.status,
                        IndexGeneration.chunk_count,
                        IndexGeneration.embedding_model,
                        IndexGeneration.embedding_dim,
                        IndexGeneration.created_at,
                        IndexGeneration.activated_at,
                    )
                    .where(IndexGeneration.corpus_id == "agent-interview")
                    .order_by(IndexGeneration.id.desc())
                )
            ).all()
            return {
                "generations": [
                    {
                        "generation_id": g.generation_id,
                        "collection": g.collection,
                        "status": g.status,
                        "chunk_count": g.chunk_count,
                        "embedding_model": g.embedding_model,
                        "embedding_dim": g.embedding_dim,
                        "created_at": g.created_at.isoformat() if g.created_at else None,
                        "activated_at": g.activated_at.isoformat() if g.activated_at else None,
                    }
                    for g in rows
                ],
                "active": corpus.active_generation_id if corpus else None,
                "revision": corpus.publication_revision if corpus else 0,
            }

    @app.post("/v1/index-generations/{generation_id}/activate")
    async def activate_generation(generation_id: str, req: ActivateRequest) -> dict:
        if not _is_server_mode(settings):
            raise HTTPException(status_code=400, detail="local 模式无发布指针")
        db = get_database(settings.database_url)
        try:
            async with db.session_factory() as session:
                return await publish_generation(
                    session,
                    "agent-interview",
                    generation_id,
                    req.expected_generation_id,
                    req.expected_revision,
                )
        except PublishConflict as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/v1/index-generations/{generation_id}/rollback")
    async def rollback_generation(generation_id: str, req: ActivateRequest) -> dict:
        if not _is_server_mode(settings):
            raise HTTPException(status_code=400, detail="local 模式无发布指针")
        db = get_database(settings.database_url)
        try:
            async with db.session_factory() as session:
                return await rollback(
                    session, "agent-interview", generation_id, req.expected_revision
                )
        except PublishConflict as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/v1/index-generations/cleanup")
    async def cleanup_generations_api(retain: int = Query(default=2, ge=0, le=100)) -> dict:
        if not _is_server_mode(settings):
            raise HTTPException(status_code=400, detail="local 模式无 collection 清理")
        from interview_rag.application.cleanup import cleanup_generations
        from interview_rag.storage.vectorstore import store_for_settings

        db = get_database(settings.database_url)
        store = store_for_settings(settings)
        async with db.session_factory() as session:
            return await cleanup_generations(session, store, "agent-interview", retain=retain)

    @app.post("/v1/chat")
    async def chat_api(req: ChatRequest):
        # local 模式：仅支持非流式（P0 回退路径）。
        if not _is_server_mode(settings):
            try:
                return await answer_question(
                    settings, app.state.runtime, req.question, req.top_k, req.min_score
                )
            except (IncompatibleIndexError, NoIndexError) as exc:
                raise HTTPException(status_code=409, detail=str(exc)) from exc
            except ProviderError as exc:
                raise HTTPException(status_code=502, detail=f"模型服务异常：{exc}") from exc
            except Exception as exc:  # noqa: BLE001
                logger.exception("chat 未预期异常")
                raise HTTPException(status_code=500, detail=f"服务内部错误：{exc}") from exc

        if req.stream:
            async def _sse_chat():
                async for item in query_generation.answer_question_stream(
                    settings,
                    app.state.runtime,
                    req.question,
                    req.conversation_id,
                    req.top_k,
                    req.min_score,
                ):
                    yield _sse_format(item["event"], item["data"])

            return StreamingResponse(
                _sse_chat(),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
            )
        try:
            return await query_generation.answer_question(
                settings, app.state.runtime, req.question, req.top_k, req.min_score, req.conversation_id
            )
        except (IncompatibleIndexError, query_generation.NoActiveGenerationError) as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except ProviderError as exc:
            raise HTTPException(status_code=502, detail=f"模型服务异常：{exc}") from exc
        except Exception as exc:  # noqa: BLE001
            logger.exception("chat 未预期异常")
            raise HTTPException(status_code=500, detail=f"服务内部错误：{exc}") from exc

    @app.get("/v1/conversations")
    async def list_conversations_api() -> dict:
        if not _is_server_mode(settings):
            raise HTTPException(status_code=400, detail="local 模式无会话")
        from interview_rag.application.conversation_service import list_conversations

        db = get_database(settings.database_url)
        async with db.session_factory() as session:
            return {"conversations": await list_conversations(session)}

    @app.post("/v1/conversations")
    async def create_conversation_api() -> dict:
        if not _is_server_mode(settings):
            raise HTTPException(status_code=400, detail="local 模式无会话")
        from interview_rag.application.conversation_service import create_conversation

        db = get_database(settings.database_url)
        async with db.session_factory() as session:
            return await create_conversation(session)

    @app.get("/v1/conversations/{conversation_id}/messages")
    async def list_messages_api(conversation_id: str) -> dict:
        if not _is_server_mode(settings):
            raise HTTPException(status_code=400, detail="local 模式无会话")
        from interview_rag.application.conversation_service import list_messages

        db = get_database(settings.database_url)
        async with db.session_factory() as session:
            return {"messages": await list_messages(session, conversation_id)}

    @app.get("/v1/answers")
    async def list_answers_api(conversation_id: str | None = None) -> dict:
        if not _is_server_mode(settings):
            raise HTTPException(status_code=400, detail="local 模式无答案记录")
        from interview_rag.application.conversation_service import list_answers

        db = get_database(settings.database_url)
        async with db.session_factory() as session:
            return {"answers": await list_answers(session, conversation_id)}

    @app.get("/v1/answers/{answer_id}")
    async def get_answer_api(answer_id: str) -> dict:
        if not _is_server_mode(settings):
            raise HTTPException(status_code=400, detail="local 模式无答案记录")
        from interview_rag.application.conversation_service import get_answer

        db = get_database(settings.database_url)
        async with db.session_factory() as session:
            answer = await get_answer(session, answer_id)
            if not answer:
                raise HTTPException(status_code=404, detail="答案不存在")
            return answer

    @app.get("/v1/answers/{answer_id}/citations/{citation_index}")
    async def get_citation_api(answer_id: str, citation_index: int) -> dict:
        if not _is_server_mode(settings):
            raise HTTPException(status_code=400, detail="local 模式无答案记录")
        from interview_rag.application.conversation_service import get_answer

        db = get_database(settings.database_url)
        async with db.session_factory() as session:
            answer = await get_answer(session, answer_id)
            if not answer:
                raise HTTPException(status_code=404, detail="答案不存在")
            citation = next((c for c in answer["citations"] if c["index"] == citation_index), None)
            if not citation:
                raise HTTPException(status_code=404, detail="引用不存在")
            return citation

    # 托管 React 构建产物；/v1 与 /health 路由优先于该挂载点。
    if _WEB_DIST.is_dir():
        app.mount("/", StaticFiles(directory=_WEB_DIST, html=True), name="web")
    else:

        @app.get("/", include_in_schema=False)
        async def index() -> dict:
            return {
                "message": "前端未构建：请在 web/ 目录执行 npm install && npm run build，"
                "或使用 npm run dev 启动 Vite 开发服务器。"
            }

    return app


def main() -> None:
    import uvicorn

    settings = get_settings()
    configure_logging(settings.log_level, settings.log_json)
    uvicorn.run(
        "interview_rag.api.app:create_app",
        factory=True,
        host="127.0.0.1",
        port=8000,
        log_config=None,
    )


if __name__ == "__main__":
    main()
