"""
Author  :Alex Aug
Date    :2026/11/12 10:50
describe:API 请求 / 响应模型
"""
import re
from typing import Generic, Literal, TypeVar

from pydantic import BaseModel, Field, field_validator

SESSION_ID_PATTERN = r'^[A-Za-z0-9_-]{1,64}$'
_SESSION_ID_RE = re.compile(SESSION_ID_PATTERN)

T = TypeVar('T')


class ApiResponse(BaseModel, Generic[T]):
    """统一响应结构"""
    code: int = 0
    message: str = 'ok'
    data: T | None = None


class ChatRequest(BaseModel):
    """同步 / 流式问答请求"""
    question: str = Field(..., min_length=1, max_length=2000, description='用户问题')
    session_id: str | None = Field(
        default=None,
        description='会话 ID；不传或传空字符串时由服务端生成',
    )

    @field_validator('question')
    @classmethod
    def _strip_question(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError('question 不能为空')
        return value

    @field_validator('session_id')
    @classmethod
    def _validate_session_id(cls, value: str | None) -> str | None:
        if value is None:
            return None
        value = value.strip()
        if not value:
            return None
        if not _SESSION_ID_RE.fullmatch(value):
            raise ValueError('session_id 仅允许 1~64 位字母、数字、下划线和连字符')
        return value


class ChatData(BaseModel):
    """问答结果"""
    answer: str
    source: Literal['qa', 'rag', 'system']
    session_id: str
    latency_ms: float
    cached: bool = False


class HistoryMessage(BaseModel):
    role: str
    content: str


class SessionHistoryData(BaseModel):
    session_id: str
    messages: list[HistoryMessage]


class ClearHistoryData(BaseModel):
    session_id: str
    deleted: int


class HealthData(BaseModel):
    status: Literal['up', 'degraded']
    version: str
    uptime_seconds: float
    dependencies: dict[str, Literal['up', 'down']]
