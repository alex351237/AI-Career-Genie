"""
Author  :Alex Aug
Date    :2026/11/12 10:48
describe:
"""
import logging
from enum import IntEnum
from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from backend.config import BASE_DIR

logger = logging.getLogger(f'{BASE_DIR.name}.{__name__}')


class ErrorCode(IntEnum):
    """业务错误码"""
    OK = 0
    INVALID_PARAMS = 1001
    NOT_FOUND = 1002
    DEPENDENCY_UNAVAILABLE = 2001
    TOO_MANY_REQUESTS = 3001
    UNAUTHORIZED = 4001
    INTERNAL_ERROR = 5000


class AppError(Exception):
    """业务异常"""

    def __init__(
            self,
            code: int,
            message: str,
            http_status: int = 500,
            data: Any = None
    ):
        super().__init__(message)
        self.code = int(code)
        self.message = message
        self.http_status = http_status
        self.data = data


def error_response(
        code: int,
        message: str,
        http_status: int,
        data: Any = None
) -> JSONResponse:
    return JSONResponse(
        status_code=http_status,
        content={'code': int(code), 'message': message, 'data': data},
    )


_HTTP_ERROR_MAP: dict[int, tuple[ErrorCode, str]] = {
    400: (ErrorCode.INVALID_PARAMS, '请求参数不合法'),
    401: (ErrorCode.UNAUTHORIZED, '未认证'),
    403: (ErrorCode.UNAUTHORIZED, '未认证'),
    404: (ErrorCode.NOT_FOUND, '资源或路由不存在'),
    405: (ErrorCode.NOT_FOUND, '资源或路由不存在'),
    422: (ErrorCode.INVALID_PARAMS, '请求参数不合法'),
    429: (ErrorCode.TOO_MANY_REQUESTS, '请求过于频繁'),
    503: (ErrorCode.DEPENDENCY_UNAVAILABLE, '依赖不可用'),
}


def register_exception_handlers(app: FastAPI) -> None:
    """注册全局异常处理器"""

    @app.exception_handler(AppError)
    async def _handle_app_error(request: Request, exc: AppError):
        logger.warning(f'业务异常 [{exc.code}] {request.method} {request.url.path}: {exc.message}')
        return error_response(exc.code, exc.message, exc.http_status, exc.data)

    @app.exception_handler(RequestValidationError)
    async def _handle_validation_error(request: Request, exc: RequestValidationError):
        details = [
            {
                'loc': list(error.get('loc', [])),
                'msg': error.get('msg', ''),
                'type': error.get('type', ''),
            }
            for error in exc.errors()
        ]
        logger.info(f'参数校验失败 {request.method} {request.url.path}: {details}')
        return error_response(
            ErrorCode.INVALID_PARAMS,
            '请求参数不合法',
            400,
            details,
        )

    @app.exception_handler(StarletteHTTPException)
    async def _handle_http_exception(request: Request, exc: StarletteHTTPException):
        code, default_message = _HTTP_ERROR_MAP.get(
            exc.status_code,
            (ErrorCode.INTERNAL_ERROR if exc.status_code >= 500 else ErrorCode.INVALID_PARAMS,
             '请求处理失败' if exc.status_code < 500 else '服务内部错误'),
        )
        message = default_message
        if (
                isinstance(exc.detail, str)
                and exc.detail
                and exc.detail not in {'Not Found', 'Method Not Allowed', 'Bad Request'}
        ):
            message = exc.detail
        return error_response(code, message, exc.status_code)

    @app.exception_handler(Exception)
    async def _handle_unexpected_error(request: Request, exc: Exception):
        # 只记录日志，不向客户端暴露堆栈
        logger.error(
            f'未处理异常 {request.method} {request.url.path}: {exc}',
            exc_info=True,
        )
        return error_response(
            ErrorCode.INTERNAL_ERROR,
            '服务内部错误，请稍后重试',
            500,
        )
