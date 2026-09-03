#D:\FYP\main\backend\app\core\database.py

import logging

from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from app.core.config import settings


logger = logging.getLogger(__name__)

DATABASE_URL = settings.DATABASE_URL.replace(
    "postgresql://",
    "postgresql+asyncpg://",
    1,
) if settings.DATABASE_URL else ""

if not DATABASE_URL:
    logger.warning(
        "DATABASE_URL is not set. "
        "The application will fail to start without a database."
    )

engine = create_async_engine(
    DATABASE_URL,
    echo=settings.APP_ENV == "development",
    pool_size=5,
    max_overflow=10,
    pool_pre_ping=True,
)

AsyncSessionlocal = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
)


async def get_db():
    async with AsyncSessionlocal() as session:
        yield session
