"""
Author  :Alex Aug
Date    :2026/11/12 10:52
describe:
"""
from fastapi import APIRouter, Depends, Path
#
# from api.dependencies import get_qa_system, require_qa_system
from api.schemas import (
    ApiResponse,
    ClearHistoryData,
    HistoryMessage,
    SESSION_ID_PATTERN,
    SessionHistoryData,
)
# from main import IntegratedQASystem

router = APIRouter(prefix='/sessions', tags=['sessions'])

SessionId = Path(..., pattern=SESSION_ID_PATTERN, description='会话 ID')


@router.get(
    '/{session_id}/history',
    response_model=ApiResponse[SessionHistoryData],
    summary='获取会话历史',
)
def get_history(
        session_id: str = SessionId,
        qa_system: IntegratedQASystem | None = Depends(get_qa_system),
) -> ApiResponse[SessionHistoryData]:
    qa_system = require_qa_system(qa_system)
    messages = [
        HistoryMessage(role=item['role'], content=item['content'])
        for item in qa_system.get_history(session_id)
    ]
    data = SessionHistoryData(session_id=session_id, messages=messages)
    return ApiResponse(data=data)


@router.delete(
    '/{session_id}/history',
    response_model=ApiResponse[ClearHistoryData],
    summary='清空会话历史',
)
def clear_history(
        session_id: str = SessionId,
        qa_system: IntegratedQASystem | None = Depends(get_qa_system),
) -> ApiResponse[ClearHistoryData]:
    qa_system = require_qa_system(qa_system)
    deleted = qa_system.clear_history(session_id)
    data = ClearHistoryData(session_id=session_id, deleted=deleted)
    return ApiResponse(data=data)
