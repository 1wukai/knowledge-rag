"""应用配置：从环境变量 / .env 读取，供 API、worker 与 CLI 共用。"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# 源码树中的仓库根：<repo>/src/interview_rag/config.py
_SOURCE_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="IR_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_name: str = "interview-rag"
    environment: str = "dev"
    log_level: str = "INFO"
    log_json: bool = True

    # 语料清单中每条文档的 path 相对于仓库根；容器内通过 IR_WORKSPACE_ROOT 覆盖。
    workspace_root: Path = Field(default_factory=lambda: _SOURCE_ROOT)
    # manifest.json 自身路径相对运行目录（本地为项目根，容器内为 /app）。
    manifest_path: Path = Field(default_factory=lambda: Path("data/manifest.json"))

    database_url: str = "mysql+aiomysql://rag:rag@localhost:3306/rag"
    redis_url: str = "redis://localhost:6379/0"
    qdrant_url: str = "http://localhost:6333"

    # 本地无 Docker 调试：Qdrant 内嵌存储目录与本地状态文件，均相对运行目录。
    qdrant_local_path: Path = Field(default_factory=lambda: Path("data/qdrant_local"))
    # qdrant 使用模式：server（服务端 URL）或 local（内嵌目录）；P1 多进程时用 server。
    qdrant_mode: str = "local"
    runtime_settings_path: Path = Field(default_factory=lambda: Path("data/runtime_settings.json"))
    index_state_path: Path = Field(default_factory=lambda: Path("data/index_state.json"))

    # 检索相关性阈值：top-1 余弦相似度低于该值时直接拒答（0~1，需按 embedding 模型标定）。
    retrieval_min_score: float = 0.4

    # 证据上下文 token 预算（近似，字符数估算）；进入生成模型的证据总量上限。
    context_max_tokens: int = 6000

    # embedding 单次批大小；千问 text-embedding-v4 等部分服务单次上限为 10。
    embedding_batch_size: int = 10

    # embedding 并发批数上限；受云端 RPM 限制，需按实际配额标定。
    embedding_concurrency: int = 4

    # 模型走云端 API；P0 默认启用 mock，无需真实 Key。
    models_mock: bool = True

    generation_provider: str = "openai-compatible"
    generation_base_url: str = ""
    generation_model: str = ""
    generation_api_key: str = ""

    embedding_provider: str = "openai-compatible"
    embedding_base_url: str = ""
    embedding_model: str = ""
    embedding_api_key: str = ""

    rerank_provider: str = "openai-compatible"
    rerank_base_url: str = ""
    rerank_model: str = ""
    rerank_api_key: str = ""


@lru_cache
def get_settings() -> Settings:
    return Settings()


def resolve_path(path: Path) -> Path:
    """把配置中的相对路径解析为相对当前运行目录的绝对路径。"""
    return path if path.is_absolute() else Path.cwd() / path
