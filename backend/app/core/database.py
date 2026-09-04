#D:\FYP\main\backend\app\core\database.py

import logging

from sqlalchemy import make_url
from sqlalchemy.engine import URL
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from app.core.config import settings


logger = logging.getLogger(__name__)


def _build_async_url_and_connect_args(raw_url: str) -> tuple[URL, dict]:
    """Build an asyncpg URL + connect args from a libpq-style URL.

    asyncpg does NOT accept libpq SSL query parameters (sslmode,
    channel_binding) as connect() kwargs. They are extracted from the
    URL query and mapped to the asyncpg ``ssl`` connect argument.
    """
    url = make_url(raw_url)

    query = dict(url.query)
    ssl_mode = query.pop("sslmode", None)
    query.pop("channel_binding", None)

    connect_args = {}
    if ssl_mode in ("require", "verify-ca", "verify-full", "prefer"):
        connect_args["ssl"] = "require"

    async_url = URL.create(
        drivername="postgresql+asyncpg",
        username=url.username,
        password=url.password,
        host=url.host,
        port=url.port,
        database=url.database,
        query=query,
    )

    return async_url, connect_args


if settings.active_database_url:
    _raw = settings.active_database_url.replace("postgresql://", "postgresql+asyncpg://", 1)
    _ASYNC_URL, _CONNECT_ARGS = _build_async_url_and_connect_args(_raw)
    DATABASE_URL = _ASYNC_URL
else:
    _CONNECT_ARGS = {}
    DATABASE_URL = ""
    logger.warning(
        "DATABASE_URL is not set. "
        "The application will fail to start without a database."
    )

logger.info(
    "Using database backend: DB_ENV=%s (%s)",
    settings.DB_ENV,
    "production/cloud" if settings.DB_ENV == "production" else "local",
)

engine = create_async_engine(
    DATABASE_URL,
    echo=settings.APP_ENV == "development",
    pool_size=5,
    max_overflow=10,
    pool_pre_ping=True,
    connect_args=_CONNECT_ARGS,
)

AsyncSessionlocal = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
)


async def get_db():
    async with AsyncSessionlocal() as session:
        yield session
