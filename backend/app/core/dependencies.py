from app.core.database import AsyncSessionlocal


async def get_db():
    async with AsyncSessionlocal() as session:
        yield session