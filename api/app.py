"""
Author  :Alex Aug
Date    :2026/11/12 10:47
describe:
"""
import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from api.exceptions import AppError, ErrorCode, register_exception_handlers
import chat, health, history
from backend.config import BASE_DIR, config
from backend.core.logger import logger

logger = logging.getLogger(f'{BASE_DIR.name}.{__name__}')


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期：初始化问答系统；失败时降级启动"""
    app.state.start_time = time.time()
    app.state.qa_system = None

    try:
        from main import IntegratedQASystem

        app.state.qa_system = IntegratedQASystem()
        logger.info('IntegratedQASystem 初始化成功')
    except Exception as e:
        logger.error(f'IntegratedQASystem 初始化失败，服务以降级模式启动: {e}', exc_info=True)

    yield

    app.state.qa_system = None


app = FastAPI(
    title=config.APP_NAME,
    version=config.APP_VERSION,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=config.CORS_ORIGINS,
    allow_credentials=False,
    allow_methods=['*'],
    allow_headers=['*'],
)

register_exception_handlers(app)

app.include_router(health.router, prefix='/api')
app.include_router(chat.router, prefix=config.API_PREFIX)
app.include_router(history.router, prefix=config.API_PREFIX)

app.mount(
    '/static',
    StaticFiles(directory=str(config.STATIC_DIR), check_dir=False),
    name='static',
)


@app.get('/', include_in_schema=False)
def index() -> FileResponse:
    index_file = config.STATIC_DIR / 'index.html'
    if not index_file.exists():
        raise AppError(ErrorCode.NOT_FOUND, '前端页面尚未部署', 404)
    return FileResponse(index_file)
