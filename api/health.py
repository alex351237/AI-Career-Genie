"""
Author  :Alex Aug
Date    :2026/11/12 10:50
describe:
"""
import asyncio
import logging
import time
from collections.abc import Callable
from typing import Any, Literal

import redis
from fastapi import APIRouter, Request

from api.schemas import ApiResponse, HealthData
from backend.config import BASE_DIR, config

logger = logging.getLogger(f'{BASE_DIR.name}.{__name__}')

router = APIRouter(tags=['health'])


def _probe_mysql() -> None:
    from backend.qa_sys.qa_db import MySQLClient

    MySQLClient().ping()


def _probe_redis() -> None:
    client = redis.Redis(
        host=config.REDIS_HOST,
        port=config.REDIS_PORT,
        password=config.REDIS_PASSWORD or None,
        db=config.REDIS_DB,
        decode_responses=True,
        socket_connect_timeout=config.HEALTH_PROBE_TIMEOUT,
        socket_timeout=config.HEALTH_PROBE_TIMEOUT,
    )
    try:
        client.ping()
    finally:
        client.close()


def _probe_milvus() -> None:
    from pymilvus import MilvusClient

    client = MilvusClient(
        uri=f'http://{config.MILVUS_HOST}:{config.MILVUS_PORT}',
        db_name=config.MILVUS_DATABASE_NAME,
    )
    try:
        client.list_collections()
    finally:
        close = getattr(client, 'close', None)
        if callable(close):
            close()


async def _run_probe(name: str, probe: Callable[[], Any]) -> Literal['up', 'down']:
    try:
        await asyncio.wait_for(
            asyncio.to_thread(probe),
            timeout=config.HEALTH_PROBE_TIMEOUT,
        )
        return 'up'
    except Exception as e:
        logger.warning(f'依赖探测失败 [{name}]: {e}')
        return 'down'


@router.get('/health', response_model=ApiResponse[HealthData], summary='健康检查')
async def health(request: Request) -> ApiResponse[HealthData]:
    dependencies: dict[str, Literal['up', 'down']] = {}
    # 串行探测，避免在依赖故障时放大连接压力
    for name, probe in (
            ('mysql', _probe_mysql),
            ('redis', _probe_redis),
            ('milvus', _probe_milvus),
    ):
        dependencies[name] = await _run_probe(name, probe)

    status: Literal['up', 'degraded'] = (
        'up' if all(value == 'up' for value in dependencies.values())
        else 'degraded'
    )
    start_time = getattr(request.app.state, 'start_time', None) or time.time()
    data = HealthData(
        status=status,
        version=config.APP_VERSION,
        uptime_seconds=round(time.time() - start_time, 2),
        dependencies=dependencies,
    )
    return ApiResponse(data=data)
