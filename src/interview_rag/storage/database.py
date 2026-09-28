"""MySQL 数据库基础设施：异步引擎与会话工厂。"""

from __future__ import annotations

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    pass


class Database:
    def __init__(self, url: str):
        self._engine = create_async_engine(url, pool_pre_ping=True)
        self._session_factory = async_sessionmaker(
            self._engine, class_=AsyncSession, expire_on_commit=False
        )

    @property
    def session_factory(self) -> async_sessionmaker[AsyncSession]:
        return self._session_factory

    async def dispose(self) -> None:
        await self._engine.dispose()


# 按 URL 复用 Database 实例。
_databases: dict[str, Database] = {}


def get_database(url: str) -> Database:
    db = _databases.get(url)
    if db is None:
        db = Database(url)
        _databases[url] = db
    return db
