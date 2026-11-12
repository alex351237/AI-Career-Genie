"""
Author  :Alex Aug
Date    :2026/11/12 10:58
describe:
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from fastapi import Request

from api.exceptions import AppError, ErrorCode

if TYPE_CHECKING:
    from main import IntegratedQASystem


def get_qa_system(request: Request) -> "IntegratedQASystem | None":
    """获取应用启动时初始化的问答系统单例（可能未初始化）"""
    return getattr(request.app.state, 'qa_system', None)


def require_qa_system(
        qa_system: "IntegratedQASystem | None",
) -> "IntegratedQASystem":
    """确认问答系统已初始化；放在路由函数内调用，保证参数校验优先返回 400"""
    if qa_system is None:
        raise AppError(
            ErrorCode.DEPENDENCY_UNAVAILABLE,
            '问答系统未初始化，依赖服务不可用',
            503,
        )
    return qa_system