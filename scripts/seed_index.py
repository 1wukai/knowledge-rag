"""首启种子入口：local 模式且索引为空时，把 manifest 全量启用文档同步入库。

以独立进程运行：内嵌 Qdrant 为单进程锁，种子进程退出后 API 进程才能打开同一索引目录。
"""

from __future__ import annotations

import asyncio

from interview_rag.application.first_run import seed_if_empty
from interview_rag.config import get_settings


def main() -> int:
    asyncio.run(seed_if_empty(get_settings()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
