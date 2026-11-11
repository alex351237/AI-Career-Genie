"""
Author  :Alex Aug
Date    :2026/11/11 20:04
describe:
"""

import numpy as np
from rank_bm25 import BM25Okapi

from backend.config import BASE_DIR
from backend.core.logger import logger
from qa_cache import RedisClient
from qa_db import MySQLClient
from qa_utils import preprocess_text

logger = logger.getLogger(f'{BASE_DIR.name}.{__name__}')


class BM25Searcher:
    """基于 BM25 算法的文本相似度检索器"""

    def __init__(
        self,
        redis_client: RedisClient,
        mysql_client: MySQLClient,
        min_score_threshold: float = 8.0
    ):
        self.redis_client = redis_client
        self.mysql_client = mysql_client
        self.min_score_threshold = min_score_threshold

        self.bm25: BM25Okapi | None = None
        self.questions: list[list[str]] = []
        self.original_questions: list[str] = []
        self._load_data()

    def _load_data(self):
        original_key = f'{self.redis_client.key_prefix}:original_questions'
        tokenized_key = f'{self.redis_client.key_prefix}:tokenized_questions'

        self.original_questions = self.redis_client.get_data(original_key) or []
        tokenized_questions = self.redis_client.get_data(tokenized_key) or []

        if not self.original_questions or not tokenized_questions:
            logger.debug('BM25 语料库缓存未命中，正在从 MySQL 加载...')
            self.original_questions = self.mysql_client.fetch_questions()
            if not self.original_questions:
                logger.warning('MySQL 数据库中未找到任何预设问题，BM25 引擎初始化空置')
                return

            tokenized_questions = [preprocess_text(question) for question in self.original_questions]
            self.redis_client.set_data(original_key, self.original_questions)
            self.redis_client.set_data(tokenized_key, tokenized_questions)

        self.questions = tokenized_questions
        self.bm25 = BM25Okapi(self.questions)
        logger.debug(f'BM25 模型初始化完成，共加载 {len(self.original_questions)} 条问答语料')

    def search(self, query: str, score_threshold: float | None = None) -> tuple[str | None, bool]:
        """
        执行用户问题的相似度检索
        :param query: 用户问题
        :param score_threshold: 可选的单次检索得分阈值
        :return: Tuple[答案文本或 None, 是否需要后续处理]
        """
        if not query or not isinstance(query, str) or not query.strip():
            logger.warning('无效的查询参数')
            return None, False

        threshold = score_threshold if score_threshold else self.min_score_threshold
        cached_answer = self.redis_client.get_answer(query)
        if cached_answer:
            logger.debug(f'用户查询直接精准命中 Redis 缓存 [Query: {query[:15]}...]')
            return cached_answer, False

        if not self.bm25 or not self.original_questions:
            logger.error('BM25 模型未正确加载语料，无法进行相似度匹配')
            return None, True

        try:
            query_tokens = preprocess_text(query)
            if not query_tokens:
                logger.warning(f'提问文本预处理后词列表为空 [Query: {query}]')
                return None, True

            scores = self.bm25.get_scores(query_tokens)
            best_idx = int(np.argmax(scores))
            best_score = float(scores[best_idx])

            if best_score >= threshold:
                matched_question = self.original_questions[best_idx]
                logger.debug(f'BM25 命中相似问题 [Score: {best_score:.2f} >= {threshold}]: "{matched_question}"')

                answer = self.redis_client.get_answer(matched_question)
                if not answer:
                    logger.debug('未在 Redis 中命中匹配问题的标准答案')
                    answer = self.mysql_client.fetch_answer(matched_question)

                if answer:
                    logger.debug('从 Redis 或 MySQL 中获取到问题的标准答案')
                    self.redis_client.set_answer(matched_question, answer)
                    return answer, False

            logger.info(f'BM25 未找到高可信答案，最高得分为: {best_score:.2f} (阈值: {threshold})')
        except Exception as e:
            logger.error(f'BM25 检索过程发生异常: {e}', exc_info=True)

        return None, True

# 兼容旧类名
BM25Search = BM25Searcher

