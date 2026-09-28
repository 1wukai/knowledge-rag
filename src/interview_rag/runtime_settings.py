"""运行时 provider 配置：网页可编辑，持久化到本地 JSON（已 gitignore）。"""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel

from interview_rag.providers import ProviderConfig


class RuntimeSettings(BaseModel):
    embedding: ProviderConfig
    generation: ProviderConfig
    rerank: ProviderConfig | None = None


def defaults() -> RuntimeSettings:
    return RuntimeSettings(
        embedding=ProviderConfig(
            base_url="http://127.0.0.1:8001/v1", model="mock-embedding"
        ),
        generation=ProviderConfig(
            base_url="http://127.0.0.1:8001/v1", model="mock-chat"
        ),
        rerank=ProviderConfig(
            base_url="http://127.0.0.1:8001/v1", model="mock-rerank"
        ),
    )


def load(path: Path) -> RuntimeSettings:
    try:
        if path.exists():
            return RuntimeSettings.model_validate(json.loads(path.read_text(encoding="utf-8")))
    except Exception:
        pass
    return defaults()


def save(path: Path, settings: RuntimeSettings) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(settings.model_dump_json(indent=2), encoding="utf-8")
