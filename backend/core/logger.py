"""
Author  :Alex Aug
Date    :2026/11/3 19:58
describe:
"""
import sys

from loguru import logger

from backend.config import settings, BASE_DIR

LOG_DIR = BASE_DIR / 'logs'


def setup_logger(
    *,
    console_level=settings.console_log_level,
    file_log_level=settings.file_log_level,
    log_file=settings.default_log_file,
    rotation='100 MB',
    retention='30 days',
    compression='zip',
    serialize=False,
    backtrace=True,
):
    """初始化日志系统"""
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    # 先移除所有日志的配置（默认配置）
    logger.remove()

    # 添加终端日志输出配置
    logger.add(
        sys.stdout,
        level=console_level,
        colorize=True,
        backtrace=backtrace,
        diagnose=False,
    )
    # 添加文件日志输出配置
    logger.add(
        LOG_DIR / log_file,
        level=file_log_level,     # 日志级别
        rotation=rotation,        # 日志切割策略（时间、大小）
        retention=retention,      # 日志保留策略
        compression=compression,  # 旧日志压缩格式
        serialize=serialize,      # 是否输出 JSON 格式
        enqueue=True,             # 是否异步写入
        catch=True,               # 捕获日志异常
        encoding='utf-8',
        backtrace=backtrace,      # 是否追踪异常（异常栈）
        diagnose=False,           # 是否显示变量诊断（生产环境）
    )
    return logger
