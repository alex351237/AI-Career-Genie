"""
Author  :Alex Aug
Date    :2026/11/11 20:05
describe:
"""

import hashlib
import json
import logging
from typing import Any

import redis

from backend.config import config, BASE_DIR
from backend.core.logger import logger

logger = logging.getLogger(f'{BASE_DIR.name}.{__name__}')

# 配置 Redis 连接池
_redis_pool = redis.ConnectionPool(
    host=config.REDIS_HOST,
    port=config.REDIS_PORT,
    password=config.REDIS_PASSWORD or None,
    db=config.REDIS_DB,
    decode_responses=True,
    max_connections=32
)


class RedisClient:

    def __init__(self, default_ttl: int | None = None, key_prefix: str = config.CACHE_KEY_PREFIX):
        self.client = redis.Redis(connection_pool=_redis_pool)
        self.default_ttl = default_ttl if default_ttl is not None else config.QAS_CACHE_TTL
        self.key_prefix = key_prefix
        try:
            self.client.ping()
            logger.debug('Redis 连接成功')
        except redis.RedisError as e:
            logger.error(f'Redis 连接失败: {e}', exc_info=True)
            # logger.exception(f'Redis 连接失败: {e}')

    @classmethod
    def for_qas(cls) -> 'RedisClient':
        """本地问答专用客户端（缓存标准答案）"""
        return cls(default_ttl=config.QAS_CACHE_TTL, key_prefix=config.CACHE_KEY_PREFIX)

    @classmethod
    def for_rag(cls) -> 'RedisClient':
        """RAG 专用客户端（缓存大模型生成的答案）"""
        return cls(default_ttl=config.RAG_CACHE_TTL, key_prefix=config.RAG_CACHE_KEY_PREFIX)

    def _generate_query_key(self, query: str) -> str:
        clean_query = query.strip().lower()
        query_hash = hashlib.md5(clean_query.encode('utf-8')).hexdigest()
        return f'{self.key_prefix}:cache:{query_hash}'

    @staticmethod
    def _generate_history_key(session_id: str) -> str:
        return f'{config.HISTORY_KEY_PREFIX}:{session_id}'

    def set_data(self, key: str, value: Any, ttl: int | None = None) -> bool:
        """
        通用数据写入
        :param key: 完整的 Redis Key
        :param value: 支持 JSON 序列化的对象
        :param ttl: 过期时间（秒），若未指定则使用 default_ttl
        :return: 操作成功返回 True，否则返回 False
        """
        expire_time = ttl if ttl is not None else self.default_ttl
        try:
            serialized_val = json.dumps(value, ensure_ascii=False)
            self.client.set(key, serialized_val, ex=expire_time)
            logger.debug(f'数据成功写入 Redis [Key: {key}], TTL: {expire_time}s')
            return True
        except (redis.RedisError, TypeError) as e:
            logger.error(f'Redis 存储失败 [Key: {key}]: {e}', exc_info=True)
            return False

    def get_data(self, key: str) -> Any:
        """
        通用数据读取
        :param key: 完整的 Redis Key
        :return: 支持 JSON 序列化的对象或 None
        """
        try:
            value = self.client.get(key)
            if value:
                logger.debug(f'Redis 缓存命中 [Key: {key}]')
                return json.loads(value)
            logger.debug(f'Redis 缓存未命中 [Key: {key}]')
        except (redis.RedisError, json.JSONDecodeError) as e:
            logger.error(f'Redis 读取/解包失败 [Key: {key}]: {e}', exc_info=True)
        return None

    def set_answer(self, query: str, answer_data: Any, ttl: int | None = None) -> bool:
        """
        将问题的答案写入缓存
        :param query: 用户问题
        :param answer_data: 答案
        :param ttl: 缓存过期时间
        :return: 操作成功返回 True，否则返回 False
        """
        key = self._generate_query_key(query)
        return self.set_data(key, answer_data, ttl=ttl)

    def get_answer(self, query: str) -> Any:
        """
        从缓存中获取问题答案
        :param query: 用户问题
        :return: 缓存的答案或 None
        """
        key = self._generate_query_key(query)
        answer = self.get_data(key)
        if answer:
            logger.info(f'从 Redis 成功获取缓存 [Query: {query[:16]}...]')
        return answer

    def clear_cache(self) -> int:
        """
        按当前业务前缀清理答案缓存
        :return: 实际删除的键数量
        """
        pattern = f'{self.key_prefix}:cache:*'
        deleted = 0
        batch: list[str] = []
        try:
            for key in self.client.scan_iter(match=pattern, count=500):
                batch.append(key)
                if len(batch) >= 500:
                    deleted += self.client.delete(*batch)
                    batch.clear()
            if batch:
                deleted += self.client.delete(*batch)
            logger.info(f'已清理业务缓存 [Pattern: {pattern}], 删除 {deleted} 个键')
        except redis.RedisError as e:
            logger.error(f'清理业务缓存失败 [Pattern: {pattern}]: {e}', exc_info=True)
        return deleted

    def add_chat_history(self, session_id: str, role: str, content: str) -> bool:
        """
        追加一轮会话记录，并仅保留最近 HISTORY_MAX_TURNS 轮
        :param session_id: 会话 ID
        :param role: user / assistant
        :param content: 消息内容
        :return: 操作成功返回 True，否则返回 False
        """
        key = self._generate_history_key(session_id)
        max_items = max(2, config.HISTORY_MAX_TURNS * 2)
        try:
            payload = json.dumps({'role': role, 'content': content}, ensure_ascii=False)
            pipe = self.client.pipeline()
            pipe.rpush(key, payload)
            pipe.ltrim(key, -max_items, -1)
            pipe.expire(key, config.QAS_CACHE_TTL)
            pipe.execute()
            logger.debug(f'会话记录写入成功 [Session: {session_id}, Role: {role}]')
            return True
        except (redis.RedisError, TypeError) as e:
            logger.error(f'会话记录写入失败 [Session: {session_id}]: {e}', exc_info=True)
            return False

    def get_chat_history(self, session_id: str) -> list[dict[str, str]]:
        """
        获取指定会话的聊天记录（按时间正序）
        :param session_id: 会话 ID
        :return: [{'role': ..., 'content': ...}, ...]
        """
        key = self._generate_history_key(session_id)
        try:
            raw_items = self.client.lrange(key, 0, -1)
        except redis.RedisError as e:
            logger.error(f'会话记录读取失败 [Session: {session_id}]: {e}', exc_info=True)
            return []

        messages: list[dict[str, str]] = []
        for raw in raw_items:
            try:
                item = json.loads(raw)
            except (TypeError, json.JSONDecodeError):
                continue
            if isinstance(item, dict) and 'role' in item and 'content' in item:
                messages.append({'role': str(item['role']), 'content': str(item['content'])})
        return messages

    def clear_chat_history(self, session_id: str) -> int:
        """
        清空指定会话的聊天记录
        :param session_id: 会话 ID
        :return: 实际删除的键数量
        """
        key = self._generate_history_key(session_id)
        try:
            deleted = int(self.client.delete(key))
            logger.info(f'已清空会话历史 [Session: {session_id}], 删除 {deleted} 个键')
            return deleted
        except redis.RedisError as e:
            logger.error(f'清空会话历史失败 [Session: {session_id}]: {e}', exc_info=True)
            return 0

