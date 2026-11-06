"""
Author  :Alex Aug
Date    :2026/11/6 19:41
describe:
"""
"""
retry - 三层兜底重试

Author: 骆昊
Version: 0.0.1
"""
import asyncio
import random
from functools import wraps
from typing import ParamSpec, Callable, TypeVar, Awaitable

from backend.core.exceptions import LLMAPIError, MilvusConnectionError, InvalidInputError, AuthenticationError
from backend.core.llm.config import AgentType
from backend.core.logger import setup_logger, logger

RETRYABLE_ERRORS = LLMAPIError, MilvusConnectionError, TimeoutError, ConnectionError
NON_RETRYABLE_ERRORS = InvalidInputError, AuthenticationError,

MAX_RETRIES = 2
RETRY_DELAYS = [1.0, 3.0]
TIMEOUT_PER_ATTEMPT = 30.0

P = ParamSpec('P')
R = TypeVar('R')


def with_retry(_agent_type: AgentType | str):
    """三层兜底装饰器工厂，给异步函数套上“重试 → 降级 → 系统兜底”三层保护"""
    try:
        agent_type = AgentType(_agent_type.lower()) if isinstance(_agent_type, str) else _agent_type
    except ValueError:
        logger.warning(f'invalid agent_type parameter | {_agent_type}')
        agent_type = AgentType.UNKNOWN

    def decorator(func: Callable[P, Awaitable[R]]) -> Callable[P, Awaitable[R]]:

        @wraps(func)
        async def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            # 第一层：自动重试
            last_error = None

            for attempt in range(MAX_RETRIES + 1):
                try:
                    async with asyncio.timeout(TIMEOUT_PER_ATTEMPT):
                        result = await func(*args, **kwargs)
                    if attempt > 0:
                        logger.debug('retry.succeeded')
                    return result
                except NON_RETRYABLE_ERRORS as e:
                    logger.error(f'retry.non_retryable_error | {agent_type} | {e}')
                    raise
                except RETRYABLE_ERRORS as e:
                    last_error = e
                    if attempt < MAX_RETRIES:
                        delay = RETRY_DELAYS[attempt]
                        logger.warning(f'retry.attempt_failed | {agent_type} | {e}')
                        await asyncio.sleep(delay)
                    else:
                        logger.error(f'retry.all_attempts_failed | {agent_type} | {e}')
                except Exception as e:
                    logger.error(f'retry.other_unknown_error | {agent_type} | {e}')
                    last_error = e

            # 第二层：Agent 级降级
            try:
                fallback_result = await AgentFallbackHandler.handle(agent_type, original_error=last_error)
                logger.info(f'retry.fallback_succeeded | {agent_type}')
                return fallback_result
            except Exception as fallback_error:
                logger.error(f'retry.fallback_failed | {agent_type} | {fallback_error}')

            # 第三层：系统级兜底
            logger.error(f'retry.system_fallback | {agent_type} | {last_error}')
            return _system_fallback_response(agent_type)

        return wrapper

    return decorator


class AgentFallbackHandler:
    """第二层降级（每个 Agent 有专项降级策略）"""

    @classmethod
    async def handle(cls, agent_type: AgentType, original_error: Exception | None):
        """根据 agent_type 选择对应的降级策略。"""
        fallback_map = {
            AgentType.QA: cls._qa_fallback,
            AgentType.EXAM: cls._exam_fallback,
            AgentType.RESUME: cls._resume_fallback,
            AgentType.INTERVIEW: cls._interview_fallback,
        }
        handler = fallback_map.get(agent_type)
        if handler:
            return await handler()
        raise original_error

    @classmethod
    async def _qa_fallback(cls) -> dict:
        """知识问答降级：知识库或模型不可用，返回提示语"""
        logger.debug('fallback.qa_service_unavailable')
        return {
            'fallback_used': True,
            'content': '知识库检索暂时不可用，请稍后重试或直接联系教师提问。',
            'structured_output': None,
        }

    @classmethod
    async def _exam_fallback(cls) -> dict:
        """试卷批阅降级：评分服务不可用，标记需教师人工复核"""
        logger.debug('fallback.exam_service_unavailable')
        return {
            'fallback_used': True,
            'needs_teacher_review': True,
            'fallback_note': '评分服务暂时不可用，已标记为需教师人工复核。',
        }

    @classmethod
    async def _resume_fallback(cls) -> dict:
        """简历审查降级：提示服务不可用 / 检查文件"""
        logger.debug('fallback.resume_service_unavailable')
        return {
            'fallback_used': True,
            'content': '简历审查服务暂时不可用，请稍后重试。如持续失败，请检查上传的文件是否完整。',
            'structured_output': None,
        }

    @classmethod
    async def _interview_fallback(cls) -> dict:
        """模拟面试降级：跳过深度分析，返回基础反馈"""
        logger.debug('fallback.interview_service_unavailable')
        return {
            'fallback_used': True,
            'content': '面试评估服务暂时不可用，已记录本次面试对话，请稍后查看报告。',
            'structured_output': None,
        }


def _system_fallback_response(agent_type: AgentType) -> dict:
    """第三层：系统级兜底。所有降级都失败后返回它，保证用户始终能收到响应"""
    messages = {
        AgentType.QA: '非常抱歉，智能问答服务暂时不可用，请稍后再试或直接联系教师提问。',
        AgentType.EXAM: '非常抱歉，试卷批改服务暂时不可用，您的提交已保存，待服务恢复后将自动处理。',
        AgentType.RESUME: '非常抱歉，简历审查服务暂时不可用，请稍后重新上传简历。',
        AgentType.INTERVIEW: '非常抱歉，模拟面试服务暂时不可用，请稍后重新开始。',
    }
    content = messages.get(agent_type, '服务暂时不可用，请稍后再试。')
    return {
        'messages': [],
        'content': content,
        'fallback_used': True,
        'system_fallback': True,
        'structured_output': None,
    }


if __name__ == '__main__':
    setup_logger()

    # ① 正常成功
    @with_retry('qa')
    async def ok():
        return "success"

    calls = {'n': 0}

    # ② 重试后成功
    @with_retry('qa')
    async def fail_twice_then_ok():
        calls['n'] += 1
        if calls['n'] < 3:
            raise LLMAPIError("网络抖动")
        return 'recovered'

    # ③ 不可重试异常立即抛出
    @with_retry('qa')
    async def non_retryable():
        raise InvalidInputError('输入非法')

    # ④ 三次全败 + 无降级策略 → 第三层系统兜底
    @with_retry('foo')
    async def always_fail():
        if random.random() < 0.9999:
            raise LLMAPIError('一直失败')
        return {}

    # ⑤ 三次全败 + 有降级策略 → 第二层 Agent 降级
    @with_retry('qa')
    async def qa_node_always_fail():
        if random.random() < 0.9999:
            raise LLMAPIError('Milvus 连接超时')
        return {}

    async def main():
        print('① 一次成功:', await ok())
        print('~' * 100)

        print('② 失败2次后第3次成功:', await fail_twice_then_ok(), '| 总调用次数:', calls['n'])
        print('~' * 100)

        try:
            await non_retryable()
        except InvalidInputError as e:
            print('③ 不可重试异常立即抛出:', e)
        print('~' * 100)

        result = await always_fail()
        print('④ 全失败→系统兜底: system_fallback =', result.get('system_fallback'),
              '| content =', result['content'][:14], '...')
        print('~' * 100)

        result = await qa_node_always_fail()
        print('⑤ 第二层降级:')
        print('   fallback_used =', result['fallback_used'])
        print('   content       =', result['content'])
        print('~' * 100)

    asyncio.run(main())
