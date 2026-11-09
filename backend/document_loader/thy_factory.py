"""
Author  :Alex Aug
Date    :2026/11/8 18:48
describe:
"""
import logging
from pathlib import Path
from typing import Type

from langchain_core.document_loaders import BaseLoader
from langchain_core.documents import Document

from backend.config import config, BASE_DIR
from backend.core.logger import logger
from loader_docx import OCRDocxLoader
from loader_image import OCRImageLoader
from loader_pdf import OCRPDFLoader

logger = logging.getLogger(f'{BASE_DIR.name}.{__name__}')


class DocumentLoaderFactory:
    """文档加载器工厂类，提供扩展名匹配、配置回退与批量加载功能"""

    SUPPORTED_LOADERS: dict[str, Type[BaseLoader]] = {
        '.pdf':  OCRPDFLoader,
        '.docx': OCRDocxLoader,
        '.jpg':  OCRImageLoader,
        '.jpeg': OCRImageLoader,
        '.png':  OCRImageLoader,
        '.bmp':  OCRImageLoader,
    }

    @classmethod
    def get_loader(cls, file_path: str, default_loader: BaseLoader) -> BaseLoader:
        """根据路径后缀匹配最佳 Loader"""
        path = Path(file_path)
        ext = path.suffix.lower()

        loader_cls = cls.SUPPORTED_LOADERS.get(ext, default_loader)

        if not loader_cls:
            raise ValueError(f'未找到适配文件 [{ext}] 的 Loader 且未配置默认的 Loader')

        logger.debug(f'文件 [{path.name}] 匹配到 Loader: {loader_cls.__name__}')
        # 使用构造器语法创建 Loader 对象
        return loader_cls(str(path))

    @classmethod
    def load_document(cls, file_path: str) -> list[Document]:
        """加载文档"""
        loader = cls.get_loader(file_path)
        return loader.load()

    @classmethod
    def is_supported(cls, file_path: str) -> bool:
        """判断指定文件格式是否在支持的加载列表内"""
        ext = Path(file_path).suffix.lower()
        return ext in cls.SUPPORTED_LOADERS