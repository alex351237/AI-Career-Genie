"""
Author  :Alex Aug
Date    :2026/11/9 14:43
describe:
"""
import hashlib
import logging
import os
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import MarkdownTextSplitter, RecursiveCharacterTextSplitter
from datetime import datetime
from base.config import config, BASE_DIR
from base.logger import setup_logging
from rag_system.document_loaders.thy_factory import DocumentLoaderFactory

logger = logging.getLogger(f'{BASE_DIR.name}.{__name__}')


def load_documents_from_directory(directory_path: str) -> list[Document]:
    """从指定文件夹递归加载多种类型文件并添加业务元数据（返回未切割的原始文档的列表）"""
    documents = []

    # 从目录名提取学科类别（如 "ai_data" -> "ai"）
    # source = os.path.basename(directory_path).replace("_data", "")

    # 遍历指定目录及其子目录
    for root, _, files in os.walk(directory_path):
        for file in files:
            file_path = os.path.join(root, file)

            # 判断文件扩展名是否在工厂类支持的类型中
            if not DocumentLoaderFactory.is_supported(file_path):
                logger.warning(f'不支持的文件类型，已跳过: {file_path}')
                continue

            try:
                # 统一通过工厂类实例化适配的 Loader
                loader = DocumentLoaderFactory.get_loader(file_path)
                loaded_docs = loader.load()

                # 添加相应的业务元数据
                current_time = datetime.now().isoformat()
                for doc in loaded_docs:
                    # doc.metadata['source'] = source
                    doc.metadata['file_path'] = file_path
                    doc.metadata['timestamp'] = current_time

                documents.extend(loaded_docs)
                logger.debug(f'成功加载文件 [{file}]: 包含 {len(loaded_docs)} 个片段')

            except Exception as e:
                logger.error(f'加载文件 {file_path} 失败: {e}', exc_info=True)

    return documents


def _generate_doc_hash(filepath: str, content: str) -> str:
    """生成文档的全局唯一确定性 ID（基于路径与内容的 MD5 摘要）"""
    hasher = hashlib.md5()
    hasher.update(filepath.encode('utf-8'))
    hasher.update(content[:200].encode('utf-8'))
    return hasher.hexdigest()[:12]


def process_documents(
        directory_path: str,
        parent_chunk_size: int = config.PARENT_CHUNK_SIZE,
        child_chunk_size: int = config.CHILD_CHUNK_SIZE,
        parent_chunk_overlap: int = config.PARENT_CHUNK_OVERLAP,
        child_chunk_overlap: int = config.CHILD_CHUNK_OVERLAP
) -> list[Document]:
    """对文档进行切割（返回保存子文档的列表）"""
    documents = load_documents_from_directory(directory_path)
    logger.debug(f'加载的文档数量: {len(documents)}')

    if not documents:
        return []

    # 初始化父块和子块分词器（通用）
    splitter_params = {
        'separators': [
            '\n\n',             # 1. 强段落分割
            '\n',               # 2. 换行分割
            r'。|！|？',         # 3. 中文句末标点
            r'\.\s|\!\s|\?\s',  # 4. 英文句末标点（带空格）
            r'；|;\s',          # 5. 分号（次级停顿）
            r'，|,\s',          # 6. 逗号（弱停顿）
            ' ',                # 7. 空格
            '',                 # 8. 字符级硬切分（兜底）
        ],
        'keep_separator': 'end',
        'is_separator_regex': True,
    }

    # 初始化 Splitter 实例
    text_parent_splitter = RecursiveCharacterTextSplitter(
        chunk_size=parent_chunk_size,
        chunk_overlap=parent_chunk_overlap,
        **splitter_params
    )
    text_child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=child_chunk_size,
        chunk_overlap=child_chunk_overlap,
        **splitter_params
    )

    markdown_parent_splitter = MarkdownTextSplitter(
        chunk_size=parent_chunk_size,
        chunk_overlap=parent_chunk_overlap
    )
    markdown_child_splitter = MarkdownTextSplitter(
        chunk_size=child_chunk_size,
        chunk_overlap=child_chunk_overlap
    )

    # 保存所有子块的列表
    all_chunks: list[Document] = []

    for i, doc in enumerate(documents, 1):
        file_path = doc.metadata.get('file_path', 'unknown_file_path')
        file_extension = Path(file_path).suffix.lower()

        is_markdown = (file_extension == '.md')
        parent_splitter = markdown_parent_splitter if is_markdown else text_parent_splitter
        child_splitter = markdown_child_splitter if is_markdown else text_child_splitter

        logger.debug(f'正在切分文档: {file_path} | 类型: {'Markdown' if is_markdown else 'Standard'}')

        # 获取文档对应的哈希前缀
        hash_prefix = _generate_doc_hash(file_path, doc.page_content)

        # 切分父块
        parent_chunks = parent_splitter.split_documents([doc])

        for j, parent_chunk in enumerate(parent_chunks):
            # 生成独一无二的父块 ID
            parent_id = f'doc_{hash_prefix}_parent_{j}'

            # 添加元数据
            parent_chunk.metadata['parent_id'] = parent_id
            parent_chunk.metadata['parent_content'] = parent_chunk.page_content

            # 切分子块
            child_chunks = child_splitter.split_documents([parent_chunk])

            for k, sub_chunk in enumerate(child_chunks):
                # 生成独一无二的子块 ID
                sub_chunk.metadata['id'] = f'{parent_id}_child_{k}'
                all_chunks.append(sub_chunk)

    logger.debug(f'切分完成: 原始文档 {len(documents)} 篇 | 生成子块 {len(all_chunks)} 个')
    return all_chunks

