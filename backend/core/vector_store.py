"""
Author  :Alex Aug
Date    :2026/11/9 14:47
describe:
"""
import logging
from typing import Any

from langchain_core.documents import Document
from pymilvus import AnnSearchRequest, DataType, MilvusClient, WeightedRanker
from pymilvus.model.hybrid import BGEM3EmbeddingFunction
from pymilvus.model.reranker import BGERerankFunction

from backend.config import BASE_DIR, config
from logger import logger
from document_processor import process_documents

logger = logging.getLogger(f'{BASE_DIR.name}.{__name__}')


def _doc_from_hit(hit: dict[str, Any]) -> Document:
    return Document(
        page_content=hit['text'],
        metadata={
            'timestamp': hit.get('timestamp', ''),
            'parent_id': hit.get('parent_id', ''),
            'parent_content': hit.get('parent_content', hit['text'])
        }
    )


def _get_unique_parent_docs(child_chunks: list[Document]) -> list[Document]:
    seen_parent_ids = set()
    unique_parent_docs = []

    for chunk in child_chunks:
        parent_id = chunk.metadata.get('parent_id')
        parent_content = chunk.metadata.get('parent_content', chunk.page_content)

        dedup_key = parent_id if parent_id else parent_content

        if dedup_key and dedup_key not in seen_parent_ids:
            metadata = chunk.metadata.copy()
            metadata.pop('parent_content', None)

            unique_parent_docs.append(
                Document(
                    page_content=parent_content,
                    metadata=metadata
                )
            )
            seen_parent_ids.add(dedup_key)

    return unique_parent_docs


def _parse_sparse_row(sparse_matrix, index: int) -> dict[int, float]:
    """
    跨平台通用解析 BGEM3 稀疏矩阵行的工具函数（Windows / macOS / Linux 兼容）
    说明：将 BGEM3EmbeddingFunction 模型计算出的稀疏向量（Sparse Vector）中的某一行，
    转化为 Milvus 数据库能够识别的标准字典格式（dict[token_id, weight]）
    """
    row = sparse_matrix[index]
    # 处理 scipy.sparse.csr_matrix 或 array 格式
    if hasattr(row, 'tocoo'):
        coo = row.tocoo()  # 转换为COO 格式（坐标格式 - 行索引、列索引、数值）
        # coo.col 保存所有非零元素的 Token ID / coo.data 保存这些非零元素的权重值
        return {int(col): float(val) for col, val in zip(coo.col, coo.data)}
    # 处理 PyTorch Tensor 或自定义稀疏类结构
    elif hasattr(row, 'col') and hasattr(row, 'data'):
        return {int(col): float(val) for col, val in zip(row.col, row.data)}
    return {}


class VectorStore:
    """向量存储"""

    def __init__(
            self,
            collection_name: str = config.MILVUS_COLLECTION_NAME,
            host: str = config.MILVUS_HOST,
            port: int = config.MILVUS_PORT,
            database: str = config.MILVUS_DATABASE_NAME
    ):
        self.collection_name = collection_name
        self.host = host
        self.port = port
        self.database = database

        # 双塔模型（Bi-Encoder）- 粗排
        # Query 和 Document 分开独立送入模型进行特征提取各自生成一个独立的向量
        # 通过向量点积或余弦相似度计算匹配得分，速度极快（Milvus 已经提前建了索引）
        bge_m3_model_path = str(config.MODELS_DIR / 'BAAI/bge-m3')
        logger.debug('正在加载 BGE-M3 Embedding 模型...')
        self.embedding_function = BGEM3EmbeddingFunction(
            model_name_or_path=bge_m3_model_path,
            use_f16=False,
            device='cpu'
        )
        self.dense_dim = self.embedding_function.dim['dense']

        # 交叉编码器 - 精排
        # 将 Query 和 Document 拼接成同一个文本序列直接输入模型
        # 利用 Transformer 底层的全注意力机制，让 Query 中的每一个词和 Document 中的每一个词都
        # 进行两两交叉交互计算，最后在 [CLS] 位置输出一个相关性得分（标量，归一化之后为 0~1 之间的值）
        # 精度极高，能捕捉到深层语义匹配、逻辑关联与细微上下文差异，但计算耗时很高（深度神经网络模型）
        bge_reranker_model_path = str(config.MODELS_DIR / 'BAAI/bge-reranker-v2-m3')
        logger.info("正在加载 BGE Reranker 模型...")
        self.reranker = BGERerankFunction(bge_reranker_model_path, device='cpu')

        self.milvus_client = MilvusClient(uri=f'http://{self.host}:{self.port}', db_name=self.database)
        self._create_or_load_collection()

    def _create_or_load_collection(self):
        if not self.milvus_client.has_collection(self.collection_name):
            schema = self.milvus_client.create_schema(auto_id=False, enable_dynamic_field=True)
            schema.add_field(field_name='id', datatype=DataType.VARCHAR, is_primary=True, max_length=64)
            schema.add_field(field_name='text', datatype=DataType.VARCHAR, max_length=65535)
            schema.add_field(field_name='dense_vector', datatype=DataType.FLOAT_VECTOR, dim=self.dense_dim)
            schema.add_field(field_name='sparse_vector', datatype=DataType.SPARSE_FLOAT_VECTOR)
            schema.add_field(field_name='parent_id', datatype=DataType.VARCHAR, max_length=64)
            schema.add_field(field_name='parent_content', datatype=DataType.VARCHAR, max_length=65535)
            schema.add_field(field_name='timestamp', datatype=DataType.VARCHAR, max_length=50)

            index_params = self.milvus_client.prepare_index_params()
            # 稠密向量的索引
            index_params.add_index(
                field_name='dense_vector',
                index_name='dense_index',
                index_type='HNSW',
                metric_type='IP',
                params={'M': 16, 'efConstruction': 128}
            )
            # 稀疏向量的索引（BGE-M3 的稀疏向量比传统的 BM25 更聪明）
            # 高级混合检索推荐 BGE-M3 Sparse + metric_type='IP'
            index_params.add_index(
                field_name='sparse_vector',
                index_name='sparse_index',
                index_type='SPARSE_INVERTED_INDEX',
                # 可以选择 BM25 或 IP，底层倒排索引算法有区别
                metric_type='IP',
                # 构建丢弃率：降噪和提效（提升 QPS）
                params={'drop_ratio_build': 0.2}
            )

            # 创建集合（同时创建索引）
            self.milvus_client.create_collection(self.collection_name, schema=schema, index_params=index_params)
            logger.info(f'成功创建集合 [{self.collection_name}] 并建立索引')
        else:
            logger.info(f'集合 [{self.collection_name}] 已存在')

        self.milvus_client.load_collection(self.collection_name)
        logger.info(f'[{self.collection_name}] 集合加载上线成功')

    def add_documents(self, documents: list[Document], batch_size: int = 32):
        """分批将文档切片进行 Vector Embedding 并写入/更新 Milvus"""
        if not documents:
            logger.warning('传入的文档列表为空，跳过写入')
            return

        total = len(documents)
        logger.debug(f'开始向量化并写入 Milvus，总文档块数: {total}')

        # 分批次将数据写入向量数据库
        for i in range(0, total, batch_size):
            batch_docs = documents[i:i + batch_size]
            texts = [doc.page_content for doc in batch_docs]
            embeddings = self.embedding_function(texts)

            data = []
            for index, doc in enumerate(batch_docs):
                child_id = doc.metadata['id']
                parent_id = doc.metadata.get('parent_id', '')
                sparse_vector = _parse_sparse_row(embeddings['sparse'], index)

                data.append({
                    'id': child_id,
                    'text': doc.page_content,
                    'dense_vector': embeddings['dense'][index],
                    'sparse_vector': sparse_vector,
                    'parent_id': parent_id,
                    'parent_content': doc.metadata.get('parent_content', doc.page_content),
                    'timestamp': doc.metadata.get('timestamp', 'unknown')
                })

            if data:
                self.milvus_client.upsert(collection_name=self.collection_name, data=data)
                logger.debug(f'批次 [{i // batch_size + 1}] 成功写入 {len(data)} 条向量')

        logger.info(f'全部文档写入完成，共计 {total} 条')

    def hybrid_search_with_rerank(
            self,
            query: str,
            k: int = config.RETRIEVAL_K,
            top_n: int = config.CANDIDATE_M
    ) -> list[Document]:
        """双路混合检索 (Dense + Sparse) + Rerank 重新排序"""
        if not query or not query.strip():
            return []

        # 向量化 Query（稠密向量 + 稀疏向量）
        query_embeddings = self.embedding_function([str(query)])
        dense_query_vector = query_embeddings['dense'][0]
        sparse_query_vector = _parse_sparse_row(query_embeddings['sparse'], 0)

        # 构建向量检索请求（近似最近邻检索请求）
        dense_request = AnnSearchRequest(
            data=[dense_query_vector],
            anns_field='dense_vector',
            param={
                'metric_type': 'IP',
                'params': {'ef': 64}
            },
            limit=k
        )
        sparse_request = AnnSearchRequest(
            data=[sparse_query_vector],
            anns_field='sparse_vector',
            param={
                'metric_type': 'IP',
                'params': {'drop_ratio_search': 0.2},
            },
            limit=k
        )

        # 粗排：加权混合检索 (密向量 0.7 + 稀疏向量 0.3)
        ranker = WeightedRanker(0.7, 0.3)
        search_results = self.milvus_client.hybrid_search(
            collection_name=self.collection_name,
            reqs=[dense_request, sparse_request],
            ranker=ranker,
            limit=k,
            output_fields=['text', 'parent_id', 'parent_content', 'timestamp']
        )

        if not search_results or not search_results[0]:
            logger.warning(f'用户查询 [{query}] 未匹配到任何向量结果')
            return []

        # 从检索结果提取子块并映射到去重的父块
        child_chunks = [_doc_from_hit(hit['entity']) for hit in search_results[0]]
        parent_docs = _get_unique_parent_docs(child_chunks)

        if not parent_docs:
            return []

        rerank_results = self.reranker(
            query=query,
            documents=[doc.page_content for doc in parent_docs],
            top_k=top_n,
        )

        for res in rerank_results:
            parent_docs[res.index].metadata['rerank_score'] = float(res.score)

        return [parent_docs[res.index] for res in rerank_results]
