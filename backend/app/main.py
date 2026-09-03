# D:\FYP\main\backend\app\main.py

from contextlib import asynccontextmanager
import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.router import router
from app.core.config import settings
from app.core.init_db import init_db
from app.core.security.rate_limit import RateLimitMiddleware


logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting %s v%s", settings.APP_NAME, settings.APP_VERSION)
    await init_db()

    try:
        from app.services.connectivity.connectivity_guard import (
            connectivity_guard,
        )
        mode = await connectivity_guard.evaluate(force=True)
        logger.info("System mode at startup: %s", mode)
    except Exception as exc:
        logger.warning("Connectivity check failed at startup: %s", exc)

    yield

    try:
        from app.services.cache.redis_cache import redis_cache
        await redis_cache.close()
    except Exception:
        pass
    logger.info("Shutdown complete")


app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    lifespan=lifespan,
)

# Middleware (order matters: CORSMiddleware outer, RateLimit inner)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(RateLimitMiddleware)

app.include_router(router)


@app.get("/")
async def root():
    return {
        "message": "Research Intelligence Platform Backend is Running"
    }
