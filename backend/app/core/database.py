#D:\FYP\main\backend\app\core\database.py
from sqlalchemy.ext.asyncio import AsyncSession
#from app.core.database import AsyncSessionlocal

from app.core.config import settings


from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
DATABASE_URL = settings.DATABASE_URL.replace(
    "postgresql://",
    "postgresql+asyncpg://",
    1
)


engine = create_async_engine(
    DATABASE_URL,
    echo=True,
)

AsyncSessionlocal = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
)


async def get_db():
    async with AsyncSessionlocal() as session:
        yield session