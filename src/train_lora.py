"""
Train Qwen + Nucleotide Transformer multimodal model
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
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForMaskedLM
import deepspeed

from trainer import MultimodalTrainer
from model import QwenWithBert, get_qwen_bert_config
from dataset.dna_rna_dataset import DNARNADataset, qwen_dna_collate_fn, DatasetConfig
from utils import print_rank_0, refresh_config, set_up_trainable_param, init_swanlab_rank_0, swanlab_log_rank_0, pre_train_lora

from model import QwenWithNt, get_qwen_nt_config

from transformers import set_seed

# 全局生效
set_seed(42)


def setup_tokenizers(args):
    """
    Setup tokenizers for both text and DNA models.
    """
    tokenizer = AutoTokenizer.from_pretrained(args.text_model_path, trust_remote_code=True)
    
    new_tokens = ["<|dna_start|>", "<|dna_pad|>", "<|dna_end|>", 
                  "<|rna_start|>", "<|rna_pad|>", "<|rna_end|>"]
    tokenizer.add_special_tokens({"additional_special_tokens": new_tokens})
    
    dna_tokenizer = AutoTokenizer.from_pretrained(args.bio_model_path, trust_remote_code=True)
    
    return tokenizer, dna_tokenizer

def setup_model_and_optimizer(args, tokenizer):
    print_rank_0("-------------------init model-------------------------")

    # Get model configuration
    model_config = get_qwen_nt_config(args.text_model_path, args.bio_model_path)
    model_config.project_token_num = args.multimodal_k_tokens
    
    # Initialize model
    model = QwenWithNt(model_config)
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
        del qwen_model

        print(f"Loading NT model from {args.bio_model_path}")
        bio_model = AutoModelForMaskedLM.from_pretrained(args.bio_model_path, trust_remote_code=True)
        
        # 加载权重时，strict=False以跳过不匹配的部分
        model.bio_model.load_state_dict(bio_model.state_dict(), strict=True)
        del bio_model
        
        print("Base models loaded successfully")

    else:
        print_rank_0("Initializing model with random weights (not loading pretrained parameters)")
    
    # Freeze DNA-BERT parameters if requested
    if args.freeze_nt:
        print_rank_0("Freezing Nucleotide Transformer parameters")
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
    Setup training and evaluation dataloaders using DNARNADataset.
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
    
    # 创建数据集配置
    train_config = DatasetConfig(
        max_len=args.max_len,
        max_src_len=args.max_src_len,
        mode=args.mode,
        padding=True,
        input_field='input',
        output_field='output'
    )
    
    # 创建训练数据集
    print_rank_0(f"Loading training dataset from {args.train_dataset_path}")
    train_dataset = DNARNADataset(
        parquet_file=args.train_dataset_path,
        tokenizer=tokenizer,
        dataset_config=train_config,
        multimodal_tokenizer=dna_tokenizer,
        read_nums=args.read_nums,
        shuffle=True,
        seed=42,
        multimodal_k_tokens=args.multimodal_k_tokens,
        num_workers=args.dataloader_num_workers,
    )
    
    # 创建评估数据集（如果需要）
    eval_dataset = None
    if not args.skip_eval and args.eval_dataset_path:
        print_rank_0(f"Loading evaluation dataset from {args.eval_dataset_path}")
        eval_config = DatasetConfig(
            max_len=args.eval_max_len,
            max_src_len=args.eval_max_src_len,
            mode=args.mode,
            padding=True,
            input_field='input',
            output_field='output'
        )
        
        eval_dataset = DNARNADataset(
            parquet_file=args.eval_dataset_path,
            tokenizer=tokenizer,
            dataset_config=eval_config,
            multimodal_tokenizer=dna_tokenizer,
            read_nums=args.eval_read_nums,
            shuffle=False,
            seed=42,
            multimodal_k_tokens=args.multimodal_k_tokens,
            num_workers=args.dataloader_num_workers,
        )
    
    return train_dataset, eval_dataset

def main():
    parser = ArgumentParser()
    # Log
    parser.add_argument('--experiment-name', type=str, default='Qwen_NT_sft_exp_',
                       help='Experiment name for logging and checkpoints')
    parser.add_argument('--output_dir', type=str, required=True,
                    help='Output path for saving models and logs')
    parser.add_argument('--swanlab', action='store_true',
                       help='Enable swanlab logging')
    parser.add_argument(
    "--report_to",
    type=str,
    nargs='+',
    default=["swanlab"],
    choices=["swanlab", "tensorboard", "wandb", "mlflow", "neptune"],
    help="Reporting tool(s) for logging"
)
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
    
    # Model
    parser.add_argument('--text-model-path', type=str, required=True,
                       help='Path to the Qwen model')
    parser.add_argument('--bio-model-path', type=str, required=True,
                       help='Path to the DNA-BERT model')
    parser.add_argument('--multimodal-k-tokens', type=int, default=64,
                       help='Number of tokens for DNA sequence projection')
    parser.add_argument('--device', type=str, default="cuda")
    parser.add_argument('--load-pretrained', action='store_true', default=True,
                       help='Load pretrained parameters for both models')
    parser.add_argument('--load_best_model_at_end', type=bool, default=True,
                       help='Load the best model at the end of training')
    parser.add_argument('--freeze-dna-bert', action='store_true', default=True,
                       help='Freeze DNA-BERT parameters')
    parser.add_argument('--freeze-nt', action='store_true', default=True,
                       help='Freeze Nucleotide Transformer parameters')
    parser.add_argument('--bf16', action='store_true', default=False,
                       help='Use bfloat16 precision')
    parser.add_argument('--fp16', action='store_true', default=False,
                       help='Use mixed precision training with fp16')
    
    # Dataset
    parser.add_argument('--train-dataset-path', type=str, required=True,
                       help='Path to training dataset (parquet format)')
    parser.add_argument('--eval-dataset-path', type=str, default=None,
                       help='Path to evaluation dataset (parquet format)')
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
    parser.add_argument('--dataloader_num_workers', type=int, default=0,
                       help='Number of workers for data loading')
    
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
    parser.add_argument('--per_device_train_batch_size', type=int, required=True,
                       help='Batch size per GPU')
    parser.add_argument('--per_device_eval_batch_size', type=int, default=4,
                       help='Evaluation batch size per GPU')
    parser.add_argument('--num_train_epochs', type=int, required=True,
                       help='Number of training epochs')
    parser.add_argument('--train-iters', type=int, default=None,
                       help='Total number of training iterations (alternative to epochs)')
    parser.add_argument('--save_strategy', type=str, required=True,
                       choices=['no', 'steps', 'epoch'],
                       help='Strategy for saving checkpoints')
    parser.add_argument('--save_steps', type=int, default=10000,
                       help='Steps between model saves')
    parser.add_argument('--eval_strategy', type=str, required=True,
                       choices=['no', 'steps', 'epoch'],
                       help='Strategy for evaluation')
    parser.add_argument('--eval_steps', type=int, default=10000,
                       help='Steps between evaluations')
    parser.add_argument('--logging_strategy', type=str, required=True,
                          choices=['no', 'steps', 'epoch'],
                          help='Strategy for logging')
    parser.add_argument('--logging_steps', type=int, default=10000,
                       help='Steps between loss logging')
    parser.add_argument('--enable-list', nargs='+', type=str, default=None,
                       help='List of enabled parameters')
    parser.add_argument('--save_trainable', type=bool, default=True,
                       help='Save trainable parameters only')
    
    # Optimizer configuration
    parser.add_argument('--learning_rate', type=float, required=True,
                    help='Learning rate')
    parser.add_argument('--gradient-accumulation-steps', type=int, default=1,
                       help='Gradient accumulation steps')
    parser.add_argument('--warmup_ratio', type=float, default=0.01,
                       help='Warmup ratio')
    parser.add_argument('--weight-decay', type=float, default=5e-4,
                       help='Weight decay')
    parser.add_argument('--eps', type=float, default=1e-8,
                       help='Epsilon for optimizer')
    parser.add_argument('--betas', nargs='+', type=float, default=[0.9, 0.999],
                       help='Beta parameters for optimizer')
    
    # Early stopping
    parser.add_argument('--early-stopping-patience', type=int, default=10,
                       help='Patience for early stopping')
    parser.add_argument('--metric-for-best-model', type=str, default='eval_loss',
                       help='Metric to track for model selection')
    parser.add_argument('--greater_is_better',  type=bool, default=False,
                       help='Whether higher metric is better')
    
    # LoRA training
    parser.add_argument('--use-lora', action='store_true',
                       help='Whether to use LoRA for parameter-efficient training')
    
    parser.add_argument('--save-total-limit', type=int, default=10,
                        help='Number of total checkpoints to keep')
    
    parser.add_argument('--local_rank', type=int, default=-1,
                       help='Local rank for distributed training')
    parser.add_argument('--save_safetensors', type=bool, default=False,
                       help='Save model in safetensors format')
    
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
    writer = None
    try:
        # Initialize distributed training
        deepspeed.init_distributed()
        
        # Set global_rank to current process rank
        args.global_rank = dist.get_rank()
        
        # Setup logging
        writer = None
        if args.global_rank == 0:  # Only execute on main process
            current_time = datetime.now().strftime('%y-%m-%d_%H-%M')
            if args.swanlab:
                # 使用专门的函数初始化SwanLab，确保只在rank 0执行
                init_swanlab_rank_0(args, experiment_suffix=current_time)
        
        # Setup tokenizers
        tokenizer, dna_tokenizer = setup_tokenizers(args)
        
        # Setup model and optimizer
        model, model_config = setup_model_and_optimizer(args, tokenizer)
        
        # Get dataloaders and convert to datasets for Transformers Trainer
        train_dataset, eval_dataset = setup_dataloaders(args, tokenizer, dna_tokenizer)
        
        # Apply parameter freezing or pre-train lora based on args.use-lora
        if args.use_lora:
            print_rank_0("Using LoRA for parameter-efficient training")
            model = pre_train_lora(model, args)
        else:
            print_rank_0("Using full parameter fine-tuning")
            set_up_trainable_param(model, args)
        
        # Create a custom data collator that uses qwen_dna_collate_fn
        data_collator = qwen_dna_collate_fn
        
        # 将所有参数打印出来以进行调试
        if args.global_rank == 0:
            print_rank_0("-------- Training Configuration --------")
            print_rank_0(f"Model: {args.text_model_path} + {args.bio_model_path}")
            print_rank_0(f"Multimodal tokens: {args.multimodal_k_tokens}")
            print_rank_0(f"Batch size: {args.per_device_train_batch_size}")
            print_rank_0(f"Learning rate: {args.learning_rate}")
            print_rank_0(f"Dataset: {args.train_dataset_path}")
            if args.swanlab:
                print_rank_0(f"SwanLab logging: Enabled")
        
        # Initialize the MultimodalTrainer
        try:
            trainer = MultimodalTrainer(
                model=model,
                args=args,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                tokenizer=tokenizer,
                data_collator=data_collator
            )
            
            # Start training
            trainer.train()
        except Exception as e:
            print_rank_0(f"Error in trainer initialization or training: {str(e)}")
            print_rank_0(f"Error type: {type(e)}")
            print_rank_0(f"Detailed traceback: {traceback.format_exc()}")
            raise
        
    except Exception as e:
        traceback_info = traceback.format_exc()
        if args.global_rank == 0:
            print_rank_0(traceback_info, logging.ERROR)
        else:
            print(traceback_info)
        raise e
    
    finally:
        if writer is not None:
            writer.close()
        if dist.is_initialized():
            dist.destroy_process_group()

if __name__ == "__main__":
    main()


    