#D:\FYP\main\backend\run.py

import os

import uvicorn

from app.core.config import settings


if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=int(os.environ.get("PORT", settings.PORT)),
    )
