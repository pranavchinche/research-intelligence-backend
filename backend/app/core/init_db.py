#D:\FYP\main\backend\app\core\init_db.py

from app.core.database import engine
from app.models import Base


async def init_db():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)