"""
Author  :Alex Aug
Date    :2026/11/8 17:03
describe:光学文字识别模块, 单例模式.
    注意：rapidocr 解析图片中的文字，也可以进行表格识别，有以下两种版本：
    ~ 当你有 GPU 且追求推理速度时，使用 rapidocr_paddle
    ~ 当只有 CPU 且需要高效推理时，使用 rapidocr_onnxruntime
"""
from functools import lru_cache
import logging

from backend.config import BASE_DIR


logger = logging.getLogger(f'{BASE_DIR.name}.{__name__}')


@lru_cache(maxsize=1)
def get_ocr_instance(use_cuda: bool = True):
    """全局单例获取 RapidOCR，避免多次加载权重模型导致内存/显存溢出"""
    try:
        from rapidocr_paddle import RapidOCR
        logger.debug('成功加载 rapidocr_paddle (GPU/CPU)')
        return RapidOCR(det_use_cuda=use_cuda, cls_use_cuda=use_cuda, rec_use_cuda=use_cuda)
    except ImportError:
        from rapidocr_onnxruntime import RapidOCR
        logger.debug('降级加载 rapidocr_onnxruntime (CPU)')
        return RapidOCR()