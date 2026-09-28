"""启动本地控制台并管理本次启动的子进程。"""

from __future__ import annotations

import argparse
import asyncio
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def _port_holder(port: int) -> str:
    """尽力查出占用端口的进程（只读 lsof），失败返回空串。"""
    try:
        out = subprocess.run(
            ["lsof", "-nP", f"-iTCP:{port}", "-sTCP:LISTEN", "-t"],
            capture_output=True, text=True, timeout=5,
        ).stdout.split()
        if not out:
            return ""
        pid = out[0]
        cmd = subprocess.run(
            ["ps", "-p", pid, "-o", "command="],
            capture_output=True, text=True, timeout=5,
        ).stdout.strip()
        return f"PID {pid}" + (f"：{cmd[:120]}" if cmd else "")
    except Exception:  # noqa: BLE001
        return ""


def check_port(port: int) -> None:
    """检测端口是否有进程在监听。

    用 connect 而非 bind：bind 会把 TIME_WAIT 状态的端口误判为占用，导致 Ctrl+C
    后立即重启报「端口占用」。connect 只在有真实 listener 时才成功。
    """
    with socket.socket() as sock:
        sock.settimeout(1)
        result = sock.connect_ex(("127.0.0.1", port))
        if result == 0:
            holder = _port_holder(port)
            hint = f"（占用进程 {holder}）" if holder else ""
            raise RuntimeError(
                f"端口 {port} 已占用{hint}。请先停止该进程，或调整 IR_API_PORT。"
            ) from None


async def check_infra(settings) -> None:
    import httpx
    from redis.asyncio import Redis
    from sqlalchemy import text
    from interview_rag.storage.database import Database

    db = Database(settings.database_url)
    redis = Redis.from_url(settings.redis_url, socket_connect_timeout=3, socket_timeout=3)
    try:
        try:
            async with asyncio.timeout(5):
                async with db.session_factory() as session:
                    await session.execute(text("SELECT 1"))
        except Exception:
            raise RuntimeError("MySQL 不可用，请检查现有容器和 IR_DATABASE_URL（库需预先存在）。") from None
        try:
            await redis.ping()
        except Exception:
            raise RuntimeError("Redis 不可用，请检查现有容器和 IR_REDIS_URL。") from None
        try:
            async with httpx.AsyncClient(timeout=5) as client:
                response = await client.get(settings.qdrant_url.rstrip("/") + "/healthz")
                response.raise_for_status()
        except Exception:
            raise RuntimeError("Qdrant 不可用，请检查现有容器和 IR_QDRANT_URL。") from None
    finally:
        await redis.aclose()
        await db.dispose()


def supervise(commands: list[tuple[str, list[str], Path]], env: dict[str, str]) -> int:
    children: list[tuple[str, subprocess.Popen]] = []
    stopped = False

    def stop(_signum, _frame):
        nonlocal stopped
        stopped = True

    previous = {s: signal.signal(s, stop) for s in (signal.SIGINT, signal.SIGTERM)}
    try:
        for name, command, cwd in commands:
            if stopped:
                return 0
            print(f"启动 {name}", flush=True)
            children.append((name, subprocess.Popen(command, cwd=cwd, env=env, start_new_session=True)))
        print("Ctrl+C 停止本次启动的服务，现有 Docker 容器保持运行。", flush=True)
        while not stopped:
            for name, child in children:
                code = child.poll()
                if code is not None:
                    print(f"{name} 已退出（{code}），停止本组其他服务。", file=sys.stderr)
                    return code if code > 0 else 1
            time.sleep(0.2)
        return 0
    finally:
        # 每个子进程拥有独立进程组，退出时只回收这一组及其后代。
        for _, child in children:
            try:
                os.killpg(child.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        deadline = time.monotonic() + 8
        for _, child in children:
            try:
                child.wait(timeout=max(0, deadline - time.monotonic()))
            except subprocess.TimeoutExpired:
                pass
        for _, child in children:
            try:
                os.killpg(child.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            child.wait()
        for s, handler in previous.items():
            signal.signal(s, handler)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local", action="store_true", help="内嵌索引与单轮调试模式")
    parser.add_argument("--dev", action="store_true", help="启动 Vite 热更新")
    parser.add_argument("--mock", action="store_true", help="额外启动 mock 模型，不修改已保存模型配置")
    args = parser.parse_args()
    os.chdir(ROOT)
    from interview_rag.config import Settings

    env = dict(os.environ, IR_QDRANT_MODE="local" if args.local else "server")
    settings = Settings(qdrant_mode=env["IR_QDRANT_MODE"])
    port = int(os.environ.get("IR_API_PORT", "8000"))
    if args.dev and port != 8000:
        raise RuntimeError("Vite 代理固定指向 8000；热更新模式请使用默认 IR_API_PORT。")
    for required in [port] + ([8001] if args.mock else []) + ([5173] if args.dev else []):
        check_port(required)
    if args.local:
        # 种子以独立子进程运行：内嵌 Qdrant 单进程锁，种子退出后 API 才能打开索引目录。
        subprocess.run([sys.executable, "scripts/seed_index.py"], env=env, check=False)
    else:
        asyncio.run(check_infra(settings))
        print("基础设施连接正常；迁移当前配置指向的业务库到最新版本。", flush=True)
        subprocess.run([sys.executable, "-m", "alembic", "upgrade", "head"], env=env, check=True)
    commands = []
    if args.mock:
        commands.append(("mock 模型 :8001", [sys.executable, "-m", "uvicorn", "interview_rag.mock_model.app:app", "--host", "127.0.0.1", "--port", "8001"], ROOT))
    if not args.local:
        for role in ("worker", "dispatcher"):
            commands.append((role, [sys.executable, "-c", f"from interview_rag.worker.main import {role}_main; {role}_main()"], ROOT))
    commands.append((f"API :{port}", [sys.executable, "-m", "uvicorn", "interview_rag.api.app:create_app", "--factory", "--host", "127.0.0.1", "--port", str(port)], ROOT))
    if args.dev:
        commands.append(("Vite :5173", ["npm", "run", "dev", "--", "--host", "127.0.0.1", "--strictPort"], ROOT / "web"))
    print(f"控制台：http://127.0.0.1:{5173 if args.dev else port}（服务就绪后访问）", flush=True)
    return supervise(commands, env)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (RuntimeError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(1)
    except subprocess.CalledProcessError:
        print("数据库迁移失败，服务未启动。请检查上方迁移日志。", file=sys.stderr)
        raise SystemExit(1)
