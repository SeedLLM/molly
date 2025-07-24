import torch
from typing import Dict
import os
import json
import numpy as np
import pkg_resources
from transformers import Trainer, TrainingArguments
from transformers.trainer_utils import EvalPrediction, EvalLoopOutput
from transformers.trainer_callback import TrainerCallback
from transformers import EarlyStoppingCallback
from tqdm import tqdm
import inspect

from utils.tools import print_rank_0


class MultimodalTrainer(Trainer):
    """
    Trainer class specifically designed for multimodal training with Qwen and DNABert.
    Inherits from the transformers Trainer class and adds multimodal-specific functionality.
    """
    def __init__(self, 
                 model=None,
                 args=None,
                 data_collator=None,
                 train_dataset=None,
                 eval_dataset=None,
                 tokenizer=None,
                 model_init=None,
                 compute_metrics=None,
                 optimizers=(None, None),
                 preprocess_logits_for_metrics=None,
                 **kwargs):
        """
        初始化MultimodalTrainer
        
        Args:
            model: 要训练的模型
            args: 训练参数
            data_collator: 数据整理器
            train_dataset: 训练数据集
            eval_dataset: 评估数据集
            tokenizer: 分词器 (已弃用)
            model_init: 模型初始化函数
            compute_metrics: 计算评估指标的函数
            optimizers: 优化器和调度器元组
            preprocess_logits_for_metrics: 预处理logits的函数
        """
        
        # 保存tokenizer作为属性以保持向后兼容
        self.tokenizer = tokenizer
        
        # 只保留TrainingArguments支持的参数
        training_args_params = inspect.signature(TrainingArguments).parameters
        training_args_dict = {k: v for k, v in vars(args).items() if k in training_args_params and v is not None}
        training_args = TrainingArguments(**training_args_dict)
        
        args = training_args

        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=tokenizer,
            model_init=model_init,
            compute_metrics=compute_metrics,
            callbacks=[
                EarlyStoppingCallback(
                    early_stopping_patience=getattr(args, 'early_stopping_patience', 3),
                )
            ],
            optimizers=optimizers,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
            **kwargs
        )
        self.args = args
