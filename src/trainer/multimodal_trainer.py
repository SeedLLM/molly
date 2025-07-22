import torch
import torch.distributed as dist
from typing import Dict
import os
import json
import numpy as np
import pkg_resources
from transformers import Trainer, TrainingArguments
from transformers.trainer_utils import EvalPrediction
from transformers.trainer_callback import TrainerCallback
from torch.utils.data import DataLoader, IterableDataset
from transformers.trainer_pt_utils import IterableDatasetShard
from tqdm import tqdm

from ..utils.tools import print_rank_0

class EarlyStoppingCallback(TrainerCallback):
    """
    早停回调，用于在验证指标不再改善时提前结束训练
    """
    def __init__(self, patience=3, metric_for_best_model="eval_loss", greater_is_better=False):
        """
        初始化早停回调
        
        Args:
            patience: 容忍多少次评估而不改善
            metric_for_best_model: 用于判断最佳模型的指标名称
            greater_is_better: 对于指标，值越大是否越好
        """
        self.patience = patience
        self.metric_name = metric_for_best_model
        self.greater_is_better = greater_is_better
        self.best_metric = float('-inf') if greater_is_better else float('inf')
        self.patience_counter = 0
        
    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        """
        每次评估后调用
        """
        if metrics is None or self.metric_name not in metrics:
            return
            
        metric_value = metrics[self.metric_name]
        
        # 检查指标是否有改善
        improved = False
        if self.greater_is_better and metric_value > self.best_metric:
            improved = True
        elif not self.greater_is_better and metric_value < self.best_metric:
            improved = True

        if improved:
            self.best_metric = metric_value
            self.patience_counter = 0
            print_rank_0(f"Best {self.metric_name} improved to {metric_value:.4f}")
        else:
            self.patience_counter += 1
            print_rank_0(f"{self.metric_name} did not improve for {self.patience_counter} evaluations")
            
            if self.patience_counter >= self.patience:
                print_rank_0(f"Early stopping triggered after {self.patience_counter} evaluations without improvement")
                control.should_training_stop = True

class EvalOutput:
    """Helper class to store evaluation outputs."""
    def __init__(self, metrics, num_samples):
        self.metrics = metrics
        self.num_samples = num_samples
        self.predictions = None
        self.label_ids = None

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
        # 提取自定义参数
        self.custom_args = kwargs.pop('custom_args', args)
        
        # 保存tokenizer作为属性以保持向后兼容
        self.tokenizer = tokenizer
        
        # 如果需要，将args转换为TrainingArguments
        if not isinstance(args, TrainingArguments) and args is not None:
            # 构建transformers的训练参数
            training_args_dict = {
                "output_dir": getattr(args, 'output_path', './output'),
                "learning_rate": float(getattr(args, 'lr', 1e-5)),
                "per_device_train_batch_size": getattr(args, 'batch_size_per_gpu', 4),
                "per_device_eval_batch_size": getattr(args, 'eval_batch_size_per_gpu', 4),
                "num_train_epochs": float(getattr(args, 'epochs', 3)),
                "weight_decay": getattr(args, 'weight_decay', 0.01),
                "save_strategy": "steps",
                "eval_strategy": "steps",  # 部分版本使用 eval_strategy
                "logging_strategy": "steps",
                "save_steps": getattr(args, 'save_interval', 10000),
                "eval_steps": getattr(args, 'eval_interval', 500),
                "logging_steps": getattr(args, 'show_avg_loss_step', 100),
                "bf16": getattr(args, 'bf16', False),
                "fp16": getattr(args, 'fp16', False),
                "local_rank": getattr(args, 'local_rank', -1),
                "save_total_limit": getattr(args, 'save_total_limit', 2),
                "load_best_model_at_end": True,
                "metric_for_best_model": getattr(args, 'metric_for_best_model', "eval_loss"),
                "greater_is_better": getattr(args, 'greater_is_better', False),
                "report_to": getattr(args, 'report_to', "swanlab"),
                "logging_first_step": True,
                "seed": getattr(args, 'seed', 42),
                "dataloader_drop_last": False,
                "dataloader_num_workers": getattr(args, 'dataloader_num_workers', 0),
                "gradient_accumulation_steps": getattr(args, 'gradient_accumulation_steps', 1),
                "warmup_steps": getattr(args, 'warmup_steps', 0),
                "warmup_ratio": getattr(args, 'warmup_ratio', 0.1),
                "deepspeed": getattr(args, 'ds_config_path', None)
            }

            
            # 检查参数是否被支持
            def check_supported_param(param_name):
                try:
                    test_dict = {"output_dir": "./test", param_name: training_args_dict[param_name]}
                    TrainingArguments(**test_dict)
                    return True
                except Exception as e:
                    print_rank_0(f"参数 {param_name} 不被当前版本支持: {str(e)}")
                    return False
            
            # 修复参数兼容性问题
            supported_params = {}
            for param_name, param_value in training_args_dict.items():
                if check_supported_param(param_name):
                    supported_params[param_name] = param_value
                elif param_name == "eval_strategy" and check_supported_param("evaluation_strategy"):
                    # 如果 eval_strategy 不支持但 evaluation_strategy 支持
                    supported_params["evaluation_strategy"] = param_value
            
            # 使用经过筛选的参数创建 TrainingArguments
            try:
                training_args = TrainingArguments(**supported_params)
                print_rank_0(f"成功创建 TrainingArguments，使用以下参数: {list(supported_params.keys())}")
            except Exception as e:
                print_rank_0(f"创建 TrainingArguments 时出错: {str(e)}")
                # 使用最小化参数创建
                training_args = TrainingArguments(output_dir=getattr(args, 'output_path', './output'))
            
            args = training_args

        # 初始化父类，使用processing_class替代tokenizer
        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=tokenizer,  # 使用新参数名
            model_init=model_init,
            compute_metrics=compute_metrics or self._compute_metrics,
            callbacks=[EarlyStoppingCallback(
                patience=getattr(self.custom_args, 'early_stopping_patience', 3),
                metric_for_best_model=getattr(args, 'metric_for_best_model', 'eval_loss'),
                greater_is_better=getattr(args, 'greater_is_better', False)
            )
            ],
            optimizers=optimizers,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
            **kwargs  # 传递剩余参数
        )

    def _compute_metrics(self, eval_pred: EvalPrediction) -> Dict[str, float]:
        """
        Compute evaluation metrics.
        
        Args:
            eval_pred: Evaluation predictions and labels
            
        Returns:
            metrics: Dictionary containing computed metrics
        """
        predictions, labels = eval_pred
        metrics = {}
        
        # Calculate loss (if needed, depends on your implementation)
        # This is a simplified example - you'd need to adapt to your model's output format
        if isinstance(predictions, tuple):
            logits = predictions[0]
        else:
            logits = predictions
            
        # Only consider positions where labels != -100
        active_positions = labels != -100
        active_logits = logits[active_positions]
        active_labels = labels[active_positions]
        
        # Calculate perplexity
        if active_labels.size > 0:
            loss_fn = torch.nn.CrossEntropyLoss()
            loss = loss_fn(torch.tensor(active_logits), torch.tensor(active_labels, dtype=torch.long))
            metrics['perplexity'] = torch.exp(loss).item()
            
            # Calculate accuracy
            preds = np.argmax(active_logits, axis=-1)
            metrics['accuracy'] = (preds == active_labels).mean().item()
            
        return metrics

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """
        Compute the loss for multimodal training.
        
        Args:
            model: The model
            inputs: Model inputs
            return_outputs: Whether to return model outputs alongside the loss
            **kwargs: Additional arguments for compatibility with parent class
            
        Returns:
            loss: Computed loss value
            outputs: Model outputs (if return_outputs=True)
        """
        outputs = model(**inputs)
        
        if self.label_smoother is not None and "labels" in inputs:
            loss = self.label_smoother(outputs, inputs["labels"])
        else:
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs.loss
            
        if "attention_mask" in inputs:
            # You can implement custom loss calculation based on attention mask if needed
            pass
            
        return (loss, outputs) if return_outputs else loss

    def save_model(self, output_dir=None, _internal_call=False):
        """
        Save the model and training state.
        
        Args:
            output_dir: Directory to save the model to
            _internal_call: Whether this is an internal call
        """
        if output_dir is None:
            output_dir = self.args.output_dir
            
        os.makedirs(output_dir, exist_ok=True)
        
        # 获取实际模型（移除DeepSpeed或DataParallel等包装）
        unwrapped_model = self.model
        while hasattr(unwrapped_model, 'module'):
            # 如果是DataParallel或DistributedDataParallel包装的模型
            unwrapped_model = unwrapped_model.module
        
        # 只在主进程上保存模型
        if self.is_world_process_zero():
            print_rank_0(f"保存模型到 {output_dir}")
            
            # 1. 检查模型是否包含LoRA权重并适当保存
            if hasattr(self, 'custom_args') and getattr(self.custom_args, 'use_lora', False):
                # 对于带有LoRA的模型，需要特殊处理
                print_rank_0("检测到PEFT/LoRA模型，使用专用方法保存适配器...")
                # 保存LoRA适配器权重
                lora_output_dir = os.path.join(output_dir, "lora_weights")
                os.makedirs(lora_output_dir, exist_ok=True)
                unwrapped_model.model.save_pretrained(lora_output_dir)
                print_rank_0(f"LoRA适配器权重已保存到 {lora_output_dir}")
                
                # 也保存完整的模型状态以便兼容
                state_dict = unwrapped_model.state_dict()
                torch.save(state_dict, os.path.join(output_dir, "pytorch_model.bin"))
                print_rank_0(f"完整模型权重已保存到 {os.path.join(output_dir, 'pytorch_model.bin')}")
            else:
                # 常规模型的保存逻辑
                state_dict = unwrapped_model.state_dict()
                torch.save(state_dict, os.path.join(output_dir, "pytorch_model.bin"))
                print_rank_0(f"模型权重已保存到 {os.path.join(output_dir, 'pytorch_model.bin')}")
            
            # 2. 保存配置
            if hasattr(unwrapped_model, "config"):
                unwrapped_model.config.save_pretrained(output_dir)
                print_rank_0(f"模型配置已保存")
            elif hasattr(unwrapped_model, "text_config") and hasattr(unwrapped_model, "bio_config"):
                # 保存多模态配置
                text_config_dir = os.path.join(output_dir, "text_config")
                bio_config_dir = os.path.join(output_dir, "bio_config")
                os.makedirs(text_config_dir, exist_ok=True)
                os.makedirs(bio_config_dir, exist_ok=True)
                
                if hasattr(unwrapped_model, "text_config"):
                    unwrapped_model.text_config.save_pretrained(text_config_dir)
                if hasattr(unwrapped_model, "bio_config"):
                    unwrapped_model.bio_config.save_pretrained(bio_config_dir)
                print_rank_0(f"多模态配置已保存")
            
            # 3. 保存tokenizer
            if self.tokenizer is not None:
                self.tokenizer.save_pretrained(output_dir)
                print_rank_0(f"Tokenizer已保存")
        
        # 4. 如果使用DeepSpeed，也使用DeepSpeed的保存方法
        if self.deepspeed:
            # 避免重复保存，指定一个不同的子目录
            ds_output_dir = os.path.join(output_dir, "deepspeed_checkpoint")
            os.makedirs(ds_output_dir, exist_ok=True)
            print_rank_0(f"使用DeepSpeed保存模型状态到 {ds_output_dir}")
            self.deepspeed.save_checkpoint(ds_output_dir)
        
        # 5. 保存训练参数
        if self.args.should_save and hasattr(self, 'custom_args') and self.is_world_process_zero():
            # 保存训练参数
            training_args_path = os.path.join(output_dir, "training_args.bin")
            torch.save(self.args, training_args_path)
            print_rank_0(f"训练参数已保存到 {training_args_path}")
            
            # 保存multimodal特定配置
            config = {
                'best_metric': getattr(self.custom_args, 'best_metric', float('-inf')),
                'metric_for_best_model': self.args.greater_is_better,
                'greater_is_better': self.args.greater_is_better,
                'early_stopping_patience': getattr(self.custom_args, 'early_stopping_patience', 3),
            }
            
            # 保存LoRA配置信息
            if hasattr(self, 'custom_args'):
                config['multimodal_config'] = {
                    'dna_max_length': getattr(self.custom_args, 'multimodal_k_tokens', 64),
                    'text_max_length': getattr(self.custom_args, 'max_len', 1024),
                }
            
            multimodal_config_path = os.path.join(output_dir, 'multimodal_config.json')
            with open(multimodal_config_path, 'w') as f:
                json.dump(config, f, indent=2)
            print_rank_0(f"多模态配置已保存到 {multimodal_config_path}")

    def get_train_dataloader(self):
        """
        覆盖原有方法以支持IterableDataset
        """
        if self.train_dataset is None:
            return None
            
        if isinstance(self.train_dataset, IterableDataset):
            # 处理IterableDataset
            world_size = 1
            rank = 0
            if dist.is_initialized():
                world_size = dist.get_world_size()
                rank = dist.get_rank()
                
            # 分片IterableDataset
            sharded_dataset = IterableDatasetShard(
                self.train_dataset, 
                self.args.per_device_train_batch_size, 
                world_size, 
                rank
            )
            
            # 创建DataLoader
            return DataLoader(
                sharded_dataset,
                batch_size=self.args.per_device_train_batch_size,
                collate_fn=self.data_collator,
                drop_last=self.args.dataloader_drop_last,
                num_workers=self.args.dataloader_num_workers,
                pin_memory=self.args.dataloader_pin_memory,
            )
        else:
            # 使用默认方法处理常规Dataset
            return super().get_train_dataloader()
    
    def get_eval_dataloader(self, eval_dataset=None):
        """
        覆盖原有方法以支持IterableDataset
        """
        if eval_dataset is None and self.eval_dataset is None:
            return None
            
        eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
        
        if isinstance(eval_dataset, IterableDataset):
            # 处理IterableDataset
            world_size = 1
            rank = 0
            if dist.is_initialized():
                world_size = dist.get_world_size()
                rank = dist.get_rank()
                
            # 分片IterableDataset
            sharded_dataset = IterableDatasetShard(
                eval_dataset, 
                self.args.per_device_eval_batch_size, 
                world_size, 
                rank
            )
            
            # 创建DataLoader
            return DataLoader(
                sharded_dataset,
                batch_size=self.args.per_device_eval_batch_size,
                collate_fn=self.data_collator,
                drop_last=False,
                num_workers=self.args.dataloader_num_workers,
                pin_memory=self.args.dataloader_pin_memory,
            )
        else:
            # 使用默认方法处理常规Dataset
            return super().get_eval_dataloader(eval_dataset)

    def training_step(self, model, inputs, num_items_in_batch=None):
        """
        Perform a training step on a batch of inputs.
        
        Args:
            model: The model to train
            inputs: The inputs and targets of the model
            num_items_in_batch: Number of items in the batch
            
        Returns:
            The training loss
        """
        model.train()
        inputs = self._prepare_inputs(inputs)
        
        # 使用父类的training_step方法，确保传递num_items_in_batch参数
        loss = super().training_step(model, inputs, num_items_in_batch=num_items_in_batch)
                
        return loss

    def _prepare_inputs(self, inputs):
        """
        准备输入数据，确保它们在正确的设备上
        """
        # 处理可能从IterableDataset来的字典数据结构
        if isinstance(inputs, dict):
            return {k: v.to(self.args.device) if isinstance(v, torch.Tensor) else v 
                    for k, v in inputs.items()}
        
        # 处理从标准Dataset来的(inputs, labels)元组
        return super()._prepare_inputs(inputs)
        
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
            
            read_nums = getattr(self.custom_args, 'eval_read_nums', 100)
            batch_size = getattr(self.custom_args, 'eval_batch_size_per_gpu', 4)
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
                if hasattr(self.custom_args, 'eval_read_nums'):
                    eval_read_nums = getattr(self.custom_args, 'eval_read_nums')
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
            output = EvalOutput(metrics=metrics, num_samples=num_examples)
            
            return output
            
        finally:
            if self.model is not None and hasattr(self.model, 'forward') and 'original_forward' in locals():
                self.model.forward = original_forward
