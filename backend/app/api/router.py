from fastapi import APIRouter

from app.api.papers import router as papers_router
from app.api.search import router as search_router
from app.api.upload import router as upload_router
from app.api.novelty import router as novelty_router
from app.api.chat import router as chat_router
from app.api.gaps import router as gaps_router
from app.api.compare import router as compare_router
from app.api.summary import router as summary_router
from app.api.strengths import router as strengths_router
from app.api.validation import router as validation_router
from app.api.system import router as system_router



router = APIRouter()



router.include_router(papers_router)
router.include_router(search_router)
router.include_router(upload_router)
router.include_router(novelty_router)
router.include_router(gaps_router)
router.include_router(compare_router)
router.include_router(chat_router)
router.include_router(summary_router)
router.include_router(strengths_router)
router.include_router(validation_router)
router.include_router(system_router)