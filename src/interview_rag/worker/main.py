"""worker 与 dispatcher 入口（CLI）。"""

from __future__ import annotations

import asyncio

from arq.connections import RedisSettings


def worker_main() -> None:
    from arq import run_worker

    from interview_rag.config import get_settings
    from interview_rag.worker.jobs import QUEUE, WorkerSettings, on_shutdown, on_startup

    settings = get_settings()
    run_worker(
        WorkerSettings,
        redis_settings=RedisSettings.from_dsn(settings.redis_url),
        queue_name=QUEUE,
        on_startup=on_startup,
        on_shutdown=on_shutdown,
    )


def dispatcher_main() -> None:
    import argparse

    from interview_rag.application.dispatcher import run_dispatcher
    from interview_rag.config import get_settings

    parser = argparse.ArgumentParser(prog="interview-rag-dispatcher")
    parser.add_argument("--interval", type=float, default=1.0)
    args = parser.parse_args()

    settings = get_settings()
    asyncio.run(run_dispatcher(settings, poll_interval=args.interval))
