"""
Author  :Alex Aug
Date    :2026/11/11 20:07
describe:
"""

import logging
from pathlib import Path

import jieba

jieba.setLogLevel(logging.WARNING)

BASE_DIR = Path(__file__).resolve().parent.parent
STOP_WORDS_FILE = BASE_DIR / 'qa_system' / 'res' / '哈工大停用词表.txt'

with open(STOP_WORDS_FILE, encoding='utf-8-sig') as file_object:
    STOP_WORDS = set(file_object.read().splitlines())


def preprocess_text(text: str) -> list[str]:
    """
    中文句子分词
    :param text: 待分词的文本内容
    :return: 分词后的列表
    """
    words = jieba.lcut(text)
    return [_normalize_terminology(word) for word in words
            if word.strip() and word not in STOP_WORDS]