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


    def evaluation_loop(self, dataloader, description, prediction_loss_only=None, ignore_keys=None, metric_key_prefix="eval"):
        """
        Override the evaluation loop to disable caching during evaluation
        and avoid OOM errors by processing predictions incrementally
        """
        # Disable caching to avoid DynamicCache in outputs
        # This addresses the error with pad_across_processes for DynamicCache
        if self.model is not None and hasattr(self.model, 'forward'):
            original_forward = self.model.forward
            
            # Create a wrapper function that disables cache
            def forward_no_cache(*args, **kwargs):
                kwargs['use_cache'] = False
                return original_forward(*args, **kwargs)
                
            # Replace the forward method temporarily
            self.model.forward = forward_no_cache
        
        # 执行评估
        try:
            num_examples = 0
            total_loss = 0
            num_batches = 0
            num_data = 0
            
            total_eval_samples = None
            
            read_nums = getattr(self.args, 'eval_read_nums', 100)
            batch_size = getattr(self.args, 'eval_batch_size_per_gpu', 4)
            total_eval_samples = read_nums // batch_size
            
            print_rank_0(f"Running Evaluation with batch size {batch_size}")
            print_rank_0(f"Using estimated evaluation length of {total_eval_samples} examples")
            
            pbar = tqdm(total=total_eval_samples, desc="Evaluation")
            
            # ---------------------------------------------------------
            # 如果需要额外指标(如 accuracy / perplexity)才收集 logits。
            # 对于大规模语言模型, logits 维度巨大, 长时间累积会导致 OOM。
            # 默认关闭 logits 收集, 只记录平均 loss; 如确有需要可将其改为 True。
            # ---------------------------------------------------------
            collect_logits = False  # ← 如需计算复杂指标请手动改为 True。

            if collect_logits:
                all_preds = []
                all_labels = []
            
            # 逐批次处理
            for _, inputs in enumerate(dataloader):
                # 准备输入
                inputs = self._prepare_inputs(inputs)
                batch_size = inputs["input_ids"].size(0)
                num_data += batch_size
                
                # 禁用梯度计算
                with torch.no_grad():
                    # 计算损失和预测
                    outputs = self.model(**inputs)
                    loss = outputs.loss.mean().detach() if hasattr(outputs, "loss") else None
                    
                    if loss is not None:
                        total_loss += loss.item()
                        num_batches += 1
                    
                    # 仅在显式启用时才收集 logits 以避免占用过多内存
                    if collect_logits and "labels" in inputs:
                        labels = inputs["labels"].detach()
                        logits = outputs.logits.detach() if hasattr(outputs, "logits") else outputs[0].detach()
                        
                        all_preds.append(logits.cpu())
                        all_labels.append(labels.cpu())
                        num_examples += labels.size(0)
                pbar.update(1)
                
                # 如果达到最大评估样本数，提前结束
                if hasattr(self.args, 'eval_read_nums'):
                    eval_read_nums = getattr(self.args, 'eval_read_nums')
                    if eval_read_nums is not None and num_data >= eval_read_nums:
                        print_rank_0(f"已达到最大评估样本数 {eval_read_nums}，提前结束评估")
                        break
            
            pbar.close()
            
            # 计算平均损失
            eval_loss = total_loss / max(1, num_batches) / batch_size

            
            # 计算其他指标
            metrics = {}
            metrics["loss"] = eval_loss
            
            if collect_logits and len(all_preds) > 0 and len(all_labels) > 0:
                # 合并所有预测和标签
                all_preds = torch.cat(all_preds, dim=0)
                all_labels = torch.cat(all_labels, dim=0)
                
                # 计算其他指标
                if self.compute_metrics is not None:
                    # 确保转换为float32以避免BFloat16错误
                    all_preds_float = all_preds.to(torch.float32)
                    all_labels_float = all_labels.to(torch.float32)
                    
                    metrics.update(
                        self.compute_metrics(EvalPrediction(predictions=all_preds_float.numpy(), label_ids=all_labels_float.numpy()))
                    )
            
            # 添加前缀
            metrics = {f"{metric_key_prefix}_{k}" if not k.startswith(metric_key_prefix) else k: v for k, v in metrics.items()}
            
            # 创建输出对象
            output = EvalLoopOutput(
                predictions=all_preds.numpy() if collect_logits else None,
                label_ids=all_labels.numpy() if collect_logits else None,
                metrics=metrics,
                num_samples=num_examples
            )
            return output
            
        finally:
            if self.model is not None and hasattr(self.model, 'forward') and 'original_forward' in locals():
                self.model.forward = original_forward
