"""
Author  :Alex Aug
Date    :2026/11/10 19:46
describe:
"""
import json
import logging
import os

import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from transformers import BertTokenizer, BertForSequenceClassification, AutoModelForSequenceClassification
from transformers import Trainer, TrainingArguments

from backend.config import BASE_DIR, config
from logger import logger

logger = logging.getLogger(f'{BASE_DIR.name}.{__name__}')


class IntentDataset(Dataset):
    """自定义数据集"""
    def __init__(self, inputs, labels):
        self.inputs = inputs
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.inputs.items()}
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.int64)
        return item


def _evaluate(model, inputs, labels):
    dataset = IntentDataset(inputs, labels)
    trainer = Trainer(model=model)
    predictions = trainer.predict(dataset)
    pred_labels = np.argmax(predictions.predictions, axis=-1)

    logger.info(f'混淆矩阵:\n{confusion_matrix(labels, pred_labels)}')
    logger.info('分类报告:\n' + classification_report(labels, pred_labels, target_names=['通用知识', '专业咨询']))


class QueryClassifierTrainer:
    """离线模型训练器"""

    def __init__(self, base_model_dir: str, model_output_dir: str):
        self.base_model_dir = base_model_dir
        self.model_output_dir = model_output_dir
        self.tokenizer = BertTokenizer.from_pretrained(base_model_dir)
        self.label_map = {'通用知识': 0, '专业咨询': 1}

    def _load_data(self, data_file: str):
        if not os.path.exists(data_file):
            raise FileNotFoundError(f'数据集文件不存在: {data_file}')

        with open(data_file, encoding='utf-8') as file_obj:
            data = [json.loads(line) for line in file_obj if line.strip()]

        texts = [item['query'] for item in data]
        labels = [self.label_map[item['label']] for item in data]
        return texts, labels

    def train(self, data_file: str):
        """执行训练与评估主流程"""
        texts, labels = self._load_data(data_file)
        train_texts, valid_texts, train_labels, valid_labels = train_test_split(
            texts, labels, test_size=0.2, stratify=labels
        )

        params = {
            'truncation': True,
            'padding': True,
            'max_length': 128,
            'return_tensors': 'pt',
        }
        train_inputs = self.tokenizer(train_texts, **params)
        valid_inputs = self.tokenizer(valid_texts, **params)
        train_dataset = IntentDataset(train_inputs, train_labels)
        valid_dataset = IntentDataset(valid_inputs, valid_labels)

        model = BertForSequenceClassification.from_pretrained(self.base_model_dir, num_labels=2)

        training_args = TrainingArguments(
            output_dir='./bert_ft_train_results',  # 训练过程检查点落盘路径
            num_train_epochs=4,                    # 训练轮次
            per_device_train_batch_size=16,        # 单设备训练批次大小
            per_device_eval_batch_size=16,         # 单设备评估批次大小
            warmup_steps=32,                       # 学习率预热步数
            weight_decay=0.01,                     # 权重衰减（L2 正则化系数）
            logging_steps=8,                       # 日志记录频率
            eval_strategy='epoch',                 # 每个 Epoch 之后运行评估
            save_strategy='epoch',                 # 每个 Epoch 之后保存检查点
            load_best_model_at_end=True,           # 训练结束时加载最佳模型
            save_total_limit=1,                    # 检查点保留上限
            metric_for_best_model='eval_loss',     # 评判最佳模型的指标
            fp16=False,                            # 禁用半精度浮点数训练
            dataloader_pin_memory=False,           # macOS 关闭锁页内存
        )
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=valid_dataset,
            compute_metrics=lambda p: {
                # 参数 p 是一个命名元组，包含 predictions 和 label_ids 两个核心属性
                # p.predictions - 模型的原始输出 ---> 需要找最大值的索引来确定标签
                # p.label_ids   - 验证集真实标签 ---> 可以通过二者比较来计算准确率
                'accuracy': (np.argmax(p.predictions, axis=-1) == p.label_ids).mean()
            }
        )

        logger.debug('开始训练 BERT 意图分类模型...')
        trainer.train()

        # 保存微调后的模型
        os.makedirs(self.model_output_dir, exist_ok=True)
        model.save_pretrained(self.model_output_dir)
        self.tokenizer.save_pretrained(self.model_output_dir)
        logger.debug(f'模型已成功导出至: {self.model_output_dir}')

        # 用验证数据进行模型评估
        _evaluate(model, valid_inputs, valid_labels)

    def evaluate(self, data_file: str):
        texts, labels = self._load_data(data_file)


if __name__ == '__main__':

    # from modelscope import AutoTokenizer, AutoModelForMaskedLM
    #
    # tokenizer = AutoTokenizer.from_pretrained(str(config.DATA_DIR / "bert-intent-ft"))
    #
    # model = AutoModelForMaskedLM.from_pretrained(str(config.DATA_DIR / "bert-intent-ft"))
    #
    # evaluate()

    thy_trainer = QueryClassifierTrainer(
        base_model_dir="D:/code/myCode/RagPro/rag_system/models/BAAI/google-bert/bert-base-chinese",
        # base_model_dir=str(config.MODELS_DIR / 'google-bert/bert-base-chinese'),
        model_output_dir=str(config.MODELS_DIR / 'bert-intent-ft'),
    )
    thy_trainer.train(str(config.DATA_DIR / 'model_generic_1830.json'))

