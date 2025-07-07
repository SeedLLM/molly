"""
Train Qwen + DNABERT multimodal model
"""
import os
import gc
from datetime import datetime
from argparse import ArgumentParser
from typing import Optional, Union, List
import json
import logging
import traceback
import sys
import math

import swanlab
import torch
import torch.distributed as dist
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel, DataCollatorForLanguageModeling
import deepspeed

from src.trainer import MultimodalTrainer
from src.model import QwenWithBert, get_qwen_bert_config
from src.dataset import load_dataloder
from src.utils import print_rank_0, refresh_config, set_up_trainable_param, init_swanlab_rank_0, swanlab_log_rank_0


class MultimodalDataCollator:
    """
    简洁多模态 Collator：适配固定长度的 dna_ids_list
    """
    def __init__(self):
        pass

    def __call__(self, features):
        batch = {}

        # 标准张量字段直接堆叠
        for key in ['input_ids', 'attention_mask', 'labels']:
            batch[key] = torch.stack([f[key] for f in features])

        # 处理 dna_ids_list: List[List[Tensor]] → [B, N_dna, L_dna]
        if features[0]["dna_ids_list"] is not None:
            dna_ids_lists = [f["dna_ids_list"] for f in features]
            # 每个样本的 DNA 数量可能不同，需要 pad 数量（不是 pad 每条序列）
            max_dna_count = max(len(seq_list) for seq_list in dna_ids_lists)

            padded_dna_batch = []
            for seq_list in dna_ids_lists:
                padded_list = seq_list.copy()
                while len(padded_list) < max_dna_count:
                    # 假设每条 DNA 序列长度相同，可直接创建空序列作为 pad
                    pad_tensor = torch.zeros_like(seq_list[0])
                    padded_list.append(pad_tensor)
                padded_dna_batch.append(torch.stack(padded_list))  # [N_dna, L_dna]
            batch["dna_ids_list"] = torch.stack(padded_dna_batch)  # [B, N_dna, L_dna]
        else:
            batch["dna_ids_list"] = None

        # cal_metric_pos 是 int/None 列表，按原样收集
        if "cal_metric_pos" in features[0]:
            batch["cal_metric_pos"] = [f["cal_metric_pos"] for f in features]

        return batch


def setup_tokenizers(args):
    """
    Setup tokenizers for both text and DNA models.
    """
    # Load text tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.text_model_path, trust_remote_code=True)
    
    # Add special tokens for DNA sequences
    new_tokens = ["<|dna_start|>", "<|dna_pad|>", "<|dna_end|>"]
    tokenizer.add_special_tokens({"additional_special_tokens": new_tokens})
    
    # Load DNA tokenizer
    dna_tokenizer = AutoTokenizer.from_pretrained(args.bio_model_path, trust_remote_code=True)
    
    return tokenizer, dna_tokenizer

def setup_model_and_optimizer(args, tokenizer):
    """
    Setup model, optimizer and learning rate scheduler.
    """
    print_rank_0("-------------------init model-------------------------")
    
    # Get model configuration
    model_config = get_qwen_bert_config(args.text_model_path, args.bio_model_path)
    model_config.project_token_num = args.multimodal_k_tokens
    
    # Initialize model
    model = QwenWithBert(model_config)
    model.set_special_tokens(tokenizer)
    
    # Load pretrained model parameters if requested
    if args.load_pretrained:
        # Load Qwen model parameters
        print_rank_0(f"Loading Qwen model from {args.text_model_path}")
        qwen_model = AutoModelForCausalLM.from_pretrained(
            args.text_model_path, 
            trust_remote_code=True, 
            torch_dtype=torch.bfloat16
        )
        model.model.load_state_dict(qwen_model.state_dict())
        del qwen_model  # Free memory
        
        # Load DNABert model parameters
        print_rank_0(f"Loading DNA-BERT model from {args.bio_model_path}")
        dna_model = AutoModel.from_pretrained(
            args.bio_model_path, 
            trust_remote_code=True,
            config=model_config.bio_config
        )
        model.bio_model.load_state_dict(dna_model.state_dict(), strict=False)
        del dna_model  # Free memory
    else:
        print_rank_0("Initializing model with random weights (not loading pretrained parameters)")
    
    # Freeze DNA-BERT parameters if requested
    if args.freeze_dna_bert:
        print_rank_0("Freezing DNA-BERT parameters")
        for name, param in model.bio_model.named_parameters():
            param.requires_grad = False
    
    # Print total parameter count
    all_params = sum(p.numel() for p in model.parameters())
    print_rank_0(f"Total model parameters: {all_params:,}")
    
    # Convert model to bfloat16 for efficiency
    torch.cuda.empty_cache()
    gc.collect()
    model = model.to(torch.bfloat16).to(args.device)
    
    return model, model_config

def setup_dataloaders(args, tokenizer, dna_tokenizer):
    """
    Setup training and evaluation dataloaders.
    """
    print_rank_0("-------------------init dataset-----------------------")
    
    # Get distributed training info
    if dist.is_initialized():
        dp_rank = dist.get_rank()
        num_dp_ranks = dist.get_world_size()
    else:
        dp_rank = 0
        num_dp_ranks = 1
    
    print_rank_0(f"Rank: {dp_rank}, World Size: {num_dp_ranks}")
    
    # 直接使用IterableDataset，不转换为map-style Dataset
    dataset_kwargs = {"multimodal_k_tokens": args.multimodal_k_tokens, "multimodal_tokenizer": dna_tokenizer}
    train_dataset = load_dataloder(args, tokenizer, dp_rank, num_dp_ranks, dataset_kwargs, True, return_transformers_dataset=False)
    eval_dataset = None if args.skip_eval else load_dataloder(args, tokenizer, dp_rank, num_dp_ranks, dataset_kwargs, False, return_transformers_dataset=False)
    
    # 计算并添加一些训练相关的参数
    if train_dataset is not None:
        # 计算训练步数
        dataset_size = getattr(train_dataset, 'estimated_len', getattr(args, 'read_nums', None))
        if not dataset_size:
            if hasattr(train_dataset, '__len__'):
                dataset_size = len(train_dataset)
            else:
                # 对于IterableDataset，需要手动设置估计大小
                dataset_size = 1000000  # 1M条数据的估计值，或者从参数中读取
                print_rank_0(f"Warning: Using estimated dataset size {dataset_size} for IterableDataset")
        
        steps_per_epoch = dataset_size // (args.batch_size_per_gpu * num_dp_ranks)
        args.num_micro_update_steps = int(steps_per_epoch * args.epochs)
        args.num_global_update_steps = args.num_micro_update_steps // args.gradient_accumulation_steps
        args.warmup_steps = int(args.num_micro_update_steps * args.warmup)
        
        print_rank_0(f"--->NUMBER OF MICRO UPDATE STEPS: {args.num_micro_update_steps}")
        print_rank_0(f"--->NUMBER OF GLOBAL UPDATE STEPS: {args.num_global_update_steps}")
        print_rank_0(f"--->NUMBER OF WARMUP STEPS: {args.warmup_steps}")
        print_rank_0(f"--->Base learning rate is {args.lr}")
    
    return train_dataset, eval_dataset

def main():
    parser = ArgumentParser()
    # Logo info
    parser.add_argument('--experiment-name', type=str, default='Qwen_DNABERT_sft_exp_',
                       help='Experiment name for logging and checkpoints')
    parser.add_argument('--tensorboard', action='store_true',
                       help='Enable tensorboard logging')
    parser.add_argument('--tb-log-dir', type=str, default=None,
                       help='Tensorboard log directory')
    parser.add_argument('--swanlab', action='store_true',
                       help='Enable swanlab logging')
    parser.add_argument('--swanlab-team', type=str, default=None,
                       help='Swanlab team name')
    parser.add_argument('--swanlab-project', type=str, default=None,
                       help='Swanlab project name')
    parser.add_argument('--test-code', action='store_true',
                       help='Test mode flag')
    parser.add_argument('--profile-log-dir', type=str, default=None,
                       help='Profile log directory')
    parser.add_argument('--global-rank', default=-1, type=int,
                       help='Global rank for distributed training')
    parser.add_argument('--output-path', default="", type=str,
                       help='Output path for saving models and logs')
    
    # Model info
    parser.add_argument('--text-model-path', type=str, required=True,
                       help='Path to the Qwen model')
    parser.add_argument('--bio-model-path', type=str, required=True,
                       help='Path to the DNA-BERT model')
    parser.add_argument('--multimodal-k-tokens', type=int, default=64,
                       help='Number of tokens for DNA sequence projection')
    parser.add_argument('--device', type=str, default="cuda")
    parser.add_argument('--load-pretrained', action='store_true', default=True,
                       help='Load pretrained parameters for both models')
    parser.add_argument('--freeze-dna-bert', action='store_true', default=True,
                       help='Freeze DNA-BERT parameters')
    
    # Dataset info
    parser.add_argument('--train-dataset-path', type=str, required=True,
                       help='Path to training dataset')
    parser.add_argument('--eval-dataset-path', type=str, default=None,
                       help='Path to evaluation dataset')
    parser.add_argument('--max-len', type=int, default=1024,
                       help='Maximum sequence length')
    parser.add_argument('--max-src-len', type=int, default=1024,
                       help='Maximum source sequence length')
    parser.add_argument('--eval-max-len', type=int, default=1024)
    parser.add_argument('--eval-max-src-len', type=int, default=1024)
    parser.add_argument('--skip-eval', action='store_true')
    parser.add_argument('--read-nums', type=int, default=None,
                       help='Number of samples to read')
    parser.add_argument('--eval-read-nums', type=int, default=None,
                       help='Number of evaluation samples to read')
    
    # Dataset compatibility parameters
    parser.add_argument('--prefix', type=str, default=None,
                       help='Prefix added to the input')
    parser.add_argument('--postfix', type=str, default=None,
                       help='Postfix added to the input')
    parser.add_argument('--meta-prompt', type=str, default=None,
                       help='Systematic prompt for input')
    parser.add_argument('--batching-stretegy', type=str, default='padding', 
                       choices=['padding', 'packing'],
                       help='Strategy for batching dataset')
    parser.add_argument('--all-reduce-loss', action='store_true',
                       help='Reduce loss across GPUs')
    
    # Training configuration
    parser.add_argument('--mode', type=str, default='sft',
                       choices=['pretrain', 'sft'],
                       help='Training mode')
    parser.add_argument('--batch-size-per-gpu', type=int, default=4,
                       help='Batch size per GPU')
    parser.add_argument('--eval-batch-size-per-gpu', type=int, default=4,
                       help='Evaluation batch size per GPU')
    parser.add_argument('--epochs', type=int, default=1,
                       help='Number of training epochs')
    parser.add_argument('--train-iters', type=int, default=None,
                       help='Total number of training iterations (alternative to epochs)')
    parser.add_argument('--save-interval', type=int, default=10000,
                       help='Steps between model saves')
    parser.add_argument('--eval-interval', type=int, default=10000,
                       help='Steps between evaluations')
    parser.add_argument('--show_avg_loss_step', type=int, default=10000,
                       help='Steps between loss logging')
    parser.add_argument('--enable-list', nargs='+', type=str, default=None,
                       help='List of enabled parameters')
    parser.add_argument('--save_trainable', type=bool, default=True,
                       help='Save trainable parameters only')
    
    # Optimizer configuration
    parser.add_argument('--gradient-accumulation-steps', type=int, default=1,
                       help='Gradient accumulation steps')
    parser.add_argument('--warmup-min-lr', type=float, default=1.0e-5,
                       help='Minimum learning rate for warmup')
    parser.add_argument('--warmup-max-lr', type=float, default=2.0e-4,
                       help='Maximum learning rate for warmup')
    parser.add_argument('--warmup', type=float, default=0.01,
                       help='Warmup ratio')
    parser.add_argument('--lr', type=float, default=1.0e-4,
                       help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=5e-4,
                       help='Weight decay')
    parser.add_argument('--eps', type=float, default=1e-8,
                       help='Epsilon for optimizer')
    parser.add_argument('--betas', nargs='+', type=float, default=[0.9, 0.999],
                       help='Beta parameters for optimizer')
    
    # Early stopping
    parser.add_argument('--early-stopping-patience', type=int, default=3,
                       help='Patience for early stopping')
    parser.add_argument('--metric-for-best-model', type=str, default='eval_loss',
                       help='Metric to track for model selection')
    parser.add_argument('--greater-is-better', action='store_true',
                       help='Whether higher metric is better')
    
    # DeepSpeed
    parser.add_argument('--ds-config-path', type=str, required=True,
                       help='Path to DeepSpeed configuration')
    parser.add_argument('--local_rank', type=int, default=-1,
                       help='Local rank for distributed training')
    
    # Add DeepSpeed arguments
    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()
    
    # Calculate GPU count for DeepSpeed
    if dist.is_initialized():
        args.gpu_count = dist.get_world_size()
    else:
        args.gpu_count = 1
    
    # Add clip_grad_max_norm if not present
    if not hasattr(args, 'clip_grad_max_norm'):
        args.clip_grad_max_norm = 1.0
    
    try:
        # Initialize distributed training
        deepspeed.init_distributed()
        
        # 正确设置global_rank为当前进程的排名
        args.global_rank = deepspeed.comm.get_rank()
        
        # Setup logging
        writer = None
        if args.global_rank == 0:  # 只在主进程执行
            current_time = datetime.now().strftime('%y-%m-%d_%H-%M')
            if args.swanlab:
                # 使用专门的函数初始化SwanLab，确保只在rank 0执行
                init_swanlab_rank_0(args, experiment_suffix=current_time)
            elif args.tensorboard:
                from torch.utils.tensorboard import SummaryWriter
                log_dir = os.path.join(args.tb_log_dir, args.experiment_name + current_time)
                writer = SummaryWriter(log_dir=log_dir)
        
        # Setup tokenizers
        tokenizer, dna_tokenizer = setup_tokenizers(args)
        
        # Setup model and optimizer
        model, model_config = setup_model_and_optimizer(args, tokenizer)
        
        # Get dataloaders and convert to datasets for Transformers Trainer
        train_dataset, eval_dataset = setup_dataloaders(args, tokenizer, dna_tokenizer)
        
        # Apply parameter freezing
        set_up_trainable_param(model, args)
        
        # Create a custom data collator
        data_collator = MultimodalDataCollator()
        
        # 将所有参数打印出来以进行调试
        if args.global_rank == 0:
            print_rank_0("-------- Training Configuration --------")
            print_rank_0(f"Model: {args.text_model_path} + {args.bio_model_path}")
            print_rank_0(f"Multimodal tokens: {args.multimodal_k_tokens}")
            print_rank_0(f"Batch size: {args.batch_size_per_gpu}")
            print_rank_0(f"Learning rate: {args.lr}")
            print_rank_0(f"Dataset: {args.train_dataset_path}")
        
        # Initialize the MultimodalTrainer
        try:
            trainer = MultimodalTrainer(
                model=model,
                args=args,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                tokenizer=tokenizer,
                data_collator=data_collator,
            )
            
            # Start training
            trainer.train()
        except Exception as e:
            print_rank_0(f"Error in trainer initialization or training: {str(e)}")
            print_rank_0(f"Error type: {type(e)}")
            print_rank_0(f"Detailed traceback: {traceback.format_exc()}")
            raise
        
    except Exception as e:
        # Log any errors
        traceback_info = traceback.format_exc()
        if args.global_rank == 0:
            print_rank_0(traceback_info, logging.ERROR)
        else:
            print(traceback_info)
        raise e
    
    finally:
        # Cleanup
        if dist.is_initialized():
            dist.destroy_process_group()

if __name__ == "__main__":
    main()


    