"""
Author  :Alex Aug
Date    :2026/11/3 19:57
describe:
"""
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from sqlalchemy.engine.url import URL

BASE_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    """项目核心配置"""

    # PostgreSQL 相关配置
    db_host:     str = 'localhost'
    db_port:     int = 5433
    db_name:     str = 'eduagent'
    db_user:     str = Field(..., validation_alias='DB_USER')
    db_password: str = Field(..., validation_alias='DB_PASSWORD')

    @property
    def database_uri(self) -> URL:
        return URL.create(
            drivername='postgresql+asyncpg',
            host=self.db_host,
            port=self.db_port,
            username=self.db_user,
            password=self.db_password,
            database=self.db_name,
        )

    # Redis 相关配置
    redis_host:       str = 'localhost'
    redis_port:       int = 6379
    redis_password:   str = Field(..., validation_alias='REDIS_PASSWORD')
    redis_db:         int = 0
    redis_key_prefix: str = 'eduagent'

    @property
    def redis_uri(self) -> str:
        return f'redis://{settings.redis_host}:{settings.redis_port}/{settings.redis_db}'

    # Milvus 相关配置
    milvus_host:       str = 'localhost'
    milvus_port:       int = 19531
    bge_embedding_dim: int = 1024
    collection_name:   str = 'knowledge_domain'


    @property
    def milvus_uri(self) -> str:
        # noinspection HttpUrlsUsage
        return f'http://{self.milvus_host}:{self.milvus_port}'

    # LLM 相关配置
    deepseek_api_key:        str = Field(..., validation_alias='DEEPSEEK_API_KEY')
    deepseek_base_url:       str = 'https://api.deepseek.com'
    deepseek_model_chat:     str = 'deepseek-flash'
    deepseek_model_reasoner: str = 'deepseek-v4-pro'

    openai_api_key:        str = Field(..., validation_alias='OPENAI_API_KEY')
    openai_base_url:       str = 'https://twapi.openai-hk.com/v1'
    openai_model_chat:     str = 'gpt-6-luna'
    openai_model_reasoner: str = 'gpt-6-sol'

    # 模型路径配置
    bge_m3_model_path:     str = str(BASE_DIR / 'models/BAAI/bge-m3')
    reranker_model_path:   str = str(BASE_DIR / 'models/BAAI/bge-reranker-v2-m3')
    classifier_model_path: str = str(BASE_DIR / 'models/classifier/query-classifier-finetuned')

    # JWT（身份验证） 相关配置
    jwt_secret_key:                  str = Field(..., validation_alias='JWT_SECRET_KEY')
    jwt_algorithm:                   str = 'HS256'
    jwt_access_token_expire_minutes: int = 15
    jwt_refresh_token_expire_days:   int = 7

    # Refresh Token Cookie 相关配置
    refresh_token_cookie_name:     str = 'refresh_token'
    refresh_token_cookie_secure:   bool = False
    refresh_token_cookie_samesite: Literal['lax', 'strict', 'none'] = 'lax'
    refresh_token_cookie_path:     str = '/'
    refresh_token_cookie_domain:   str | None = None

    # API 接口相关配置
    max_pdf_size:          int = 20 * 1024 * 1024
    resume_review_timeout: int = 15 * 60
    default_reviews_count: int = 10

    # 文档切割相关配置
    chunk_size:     int = 512
    chunk_overlap:  int = 64

    # RAG 相关配置
    max_contextual_rag_tasks: int = 4
    max_parallel_sub_queries: int = 3
    precise_recall_top_k: int = 10
    vague_recall_top_k: int = 12
    board_per_query_recall_top_k: int = 5
    rerank_max_input_chars: int = 1200
    rerank_top_k: int = 3

    # MCP 服务配置
    kbs_mcp_server_url: str = 'http://localhost:8000/mcp/kbs'
    web_search_mcp_url: str = 'http://localhost:8000/mcp/web-search'
    tavily_api_key:     str = Field(..., validation_alias='TAVILY_API_KEY')
    anysearch_api_key:  str = Field(..., validation_alias='ANYSEARCH_API_KEY')

    default_tenant_id: str = 'tenant_default'

    # 日志配置
    console_log_level: str = 'WARNING'
    file_log_level:    str = 'DEBUG'
    default_log_file:  str = 'api_service.log'

    # 字符编码候选集
    candidate_encodings: list[str] = ['utf-8', 'utf-8-sig', 'gbk', 'gb2312', 'gb18030', 'big5']

    model_config = SettingsConfigDict(
        env_file=str(BASE_DIR / '.env'),
        env_file_encoding='utf-8',
        case_sensitive=False,
        extra='ignore'
    )


# 将配置创建成单例（Singleton）对象
config = Settings()
