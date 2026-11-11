"""
Author  :Alex Aug
Date    :2026/11/11 19:59
describe:qa_db - MySQL 数据库操作

pip install pymysql dbutils
"""
import logging
import threading
from pathlib import Path

import pandas as pd
import pymysql
from dbutils.pooled_db import PooledDB

from backend.config import config, BASE_DIR
from backend.core.logger import logger

logger = logging.getLogger(f'{BASE_DIR.name}.{__name__}')

# 数据库连接池采用惰性单例：模块导入时不建立连接，MySQL 不可用时应用仍可降级启动
_mysql_pool: PooledDB | None = None
_mysql_pool_lock = threading.Lock()


def _get_pool() -> PooledDB:
    """获取（必要时创建）MySQL 连接池"""
    global _mysql_pool
    if _mysql_pool is None:
        with _mysql_pool_lock:
            if _mysql_pool is None:
                _mysql_pool = PooledDB(
                    creator=pymysql,          # 使用 pymysql 数据库驱动
                    maxconnections=16,        # 连接池允许的最大连接数
                    mincached=1,              # 初始化时创建的空闲连接数
                    maxcached=8,              # 连接池中允许的最大空闲连接数
                    blocking=True,            # 连接池满时是否等待（True 表示阻塞等待，False 表示直接报错）
                    ping=1,                   # 每次从池中获取连接时先 ping 测试（1 = 自动重连）
                    host=config.MYSQL_HOST,
                    port=config.MYSQL_PORT,
                    user=config.MYSQL_USER,
                    password=config.MYSQL_PASSWORD,
                    database=config.MYSQL_DATABASE,
                    charset='utf8mb4',
                    connect_timeout=2,
                )
    return _mysql_pool


class MySQLClient:

    @property
    def pool(self) -> PooledDB:
        return _get_pool()

    def ping(self) -> bool:
        """执行 SELECT 1，用于健康检查"""
        conn = self.pool.connection()
        try:
            with conn.cursor() as cursor:
                cursor.execute('SELECT 1')
                cursor.fetchone()
            return True
        finally:
            conn.close()

    def create_table(self):
        create_table_query = '''
        CREATE TABLE IF NOT EXISTS faq
        (
            id           BIGINT AUTO_INCREMENT PRIMARY KEY,
            subject_name VARCHAR(20) NOT NULL,
            question     VARCHAR(1000) NOT NULL,
            answer       VARCHAR(10000) NOT NULL
        )
        '''
        conn = self.pool.connection()
        try:
            with conn.cursor() as cursor:
                cursor.execute(create_table_query)
            conn.commit()
            logger.debug('二维表创建成功')
        except pymysql.MySQLError as e:
            conn.rollback()
            logger.error(f'二维表创建失败: {e}', exc_info=True)
            raise
        finally:
            conn.close()

    def insert_data(self, csv_path: str | Path) -> bool:
        data = pd.read_csv(csv_path)
        data['问题'] = data['问题'].str.lower()
        data = data.fillna('')
        records = [(row['学科名称'], row['问题'], row['答案'])
                   for _, row in data.iterrows()]

        if not records:
            logger.warning(f'CSV 文件数据为空: {csv_path}')
            return False

        conn = self.pool.connection()
        try:
            with conn.cursor() as cursor:
                insert_query = 'INSERT INTO faq (subject_name, question, answer) VALUES (%s, %s, %s)'
                cursor.executemany(insert_query, records)
            conn.commit()
            logger.info(f'成功批量导入 {len(records)} 条问答数据')
            return True
        except Exception as e:
            conn.rollback()
            logger.error(f'数据批量插入失败: {e}', exc_info=True)
            raise
        finally:
            conn.close()

    def fetch_questions(self) -> list[str]:
        """获取所有问题列表"""
        conn = self.pool.connection()
        try:
            with conn.cursor() as cursor:
                cursor.execute('SELECT question FROM faq')
                rows = cursor.fetchall()
                questions = [row[0] for row in rows if row[0]]
            logger.debug('成功获取问题')
            return questions
        except pymysql.MySQLError as e:
            logger.error(f'查询失败: {e}', exc_info=True)
            return []
        finally:
            conn.close()

    def fetch_answer(self, question: str) -> str | None:
        """根据问题精准获取答案"""
        conn = self.pool.connection()
        try:
            with conn.cursor() as cursor:
                cursor.execute(
                    'SELECT answer FROM faq WHERE question=%s LIMIT 1',
                    (question, )
                )
                row = cursor.fetchone()
                if row:
                    return row[0]
                logger.debug(f'未找到问题对应的答案: {question}')
                return None
        except pymysql.MySQLError as e:
            logger.error(f'答案获取失败: {e}', exc_info=True)
        finally:
            conn.close()

