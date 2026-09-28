"""首启种子：local 模式下索引为空时，把 manifest 全量启用文档同步入库。

目标是让 clone 用户一条命令起跑：无索引时自动构建内置语料索引；embedding 指向
默认 mock 端点且不可达时，临时拉起 mock 服务完成入库后关闭。真实 provider 配置下
直接调用云端；失败只提示不阻断启动，用户仍可在 UI 配置后手动入库。
"""

from __future__ import annotations

import asyncio
import socket
import subprocess
import sys
import time
from pathlib import Path

from interview_rag.config import Settings, resolve_path
from interview_rag.application.index_state import load_state
from interview_rag.application.ingest import ingest_documents
from interview_rag.manifest import load_manifest, resolve_manifest_path
from interview_rag.providers import ProviderError
from interview_rag.runtime_settings import load as load_runtime

_MOCK_PORT = 8001


def _port_open(port: int) -> bool:
    with socket.socket() as sock:
        sock.settimeout(1)
        return sock.connect_ex(("127.0.0.1", port)) == 0


async def seed_if_empty(settings: Settings) -> bool:
    """索引已存在或入库不可用时返回 False；成功构建返回 True。"""
    if load_state(resolve_path(settings.index_state_path)):
        return False

    manifest = load_manifest(resolve_manifest_path(settings))
    doc_ids = [doc.doc_id for doc in manifest.documents if doc.enabled]
    if not doc_ids:
        print("首启种子：manifest 无启用文档，跳过入库。", flush=True)
        return False

    runtime = load_runtime(resolve_path(settings.runtime_settings_path))
    needs_mock = not runtime.embedding.base_url or f":{_MOCK_PORT}" in runtime.embedding.base_url

    proc = None
    try:
        if needs_mock and not _port_open(_MOCK_PORT):
            root = Path(__file__).resolve().parents[3]
            proc = subprocess.Popen(
                [sys.executable, "-m", "uvicorn", "interview_rag.mock_model.app:app",
                 "--host", "127.0.0.1", "--port", str(_MOCK_PORT), "--log-level", "warning"],
                cwd=root, start_new_session=True,
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
            deadline = time.monotonic() + 30
            while not _port_open(_MOCK_PORT) and time.monotonic() < deadline:
                await asyncio.sleep(0.2)
        result = await ingest_documents(settings, runtime, doc_ids)
    except ProviderError as exc:
        print(f"首启种子：Embedding 服务不可用（{exc}），跳过自动入库；可在 UI 配置模型后手动入库。", flush=True)
        return False
    finally:
        if proc is not None:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()

    print(f"首启种子完成：{len(result.doc_ids)} 篇文档、{result.chunk_count} 个 chunk 已入库。", flush=True)
    return True
