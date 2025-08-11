import logging
import traceback
from argparse import ArgumentParser
from datetime import datetime

import deepspeed
import torch
import torch.distributed as dist
from transformers import (
    AutoModelForCausalLM,
    AutoModelForMaskedLM,
    AutoTokenizer,
    set_seed,
)

# pylint: disable=no-name-in-module
# pylint: disable=too-many-branches
# pylint: disable=too-many-statements
from dataset.omics_dataset import DatasetConfig, OmicsDataset, qwen_omics_collate_fn
from model import OmicsOne, get_omics_one_config
from trainer import OmicsTrainer
from utils import (
    init_swanlab_rank_0,
    pre_train_lora,
    print_rank_0,
    set_up_trainable_param,
    time_count,
    get_current_device,
)

# 全局生效
set_seed(42)


def setup_tokenizers(args):
    """
    Setup tokenizers for both text and DNA models.
    """
    tokenizer = AutoTokenizer.from_pretrained(
        args.text_model_path, trust_remote_code=True
    )

    new_tokens = [
        "<|dna_start|>",
        "<|dna_pad|>",
        "<|dna_end|>",
        "<|rna_start|>",
        "<|rna_pad|>",
        "<|rna_end|>",
        "<|protein_start|>",
        "<|protein_pad|>",
        "<|protein_end|>",
    ]

    tokenizer.add_special_tokens({"additional_special_tokens": new_tokens})
    dna_rna_tokenizer = AutoTokenizer.from_pretrained(
        args.dna_rna_model_path, trust_remote_code=True
    )
    protein_tokenizer = AutoTokenizer.from_pretrained(
        args.protein_model_path, trust_remote_code=True
    )

    return tokenizer, dna_rna_tokenizer, protein_tokenizer


def setup_model_and_optimizer(args, tokenizer):
    print_rank_0("-------------------init model-------------------------")

    model_config = get_omics_one_config(
        args.text_model_path, args.dna_rna_model_path, args.protein_model_path
    )
    model_config.dna_rna_project_token_num = args.dna_rna_k_tokens
    model_config.protein_project_token_num = args.protein_k_tokens

    omics_one = OmicsOne(config=model_config)
    omics_one.set_special_tokens(tokenizer)

    current_device = get_current_device()
    # if args.load_pretrained:
    if args.no_load_pretrained:
        with time_count("Randomize llm model"):
            omics_one.model = AutoModelForCausalLM.from_config(model_config.text_config)
        with time_count("Randomize dna rna model"):
            omics_one.dna_rna_model = AutoModelForMaskedLM.from_config(
                model_config.dna_rna_config, trust_remote_code=True
            )
        with time_count("Randomize protein model"):
            omics_one.protein_model = AutoModelForMaskedLM.from_config(
                model_config.protein_config, trust_remote_code=True
            )
    else:
        # import pdb
        # pdb.set_trace()
        # https://github.com/huggingface/transformers/issues/38667
        with time_count("Loaded dna rna model"):
            dna_rna_model = AutoModelForMaskedLM.from_pretrained(
                args.dna_rna_model_path,
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
                device_map=current_device,
            )
            omics_one.dna_rna_model = dna_rna_model

        with time_count("Loaded llm model"):
            qwen_model = AutoModelForCausalLM.from_pretrained(
                args.text_model_path,
                torch_dtype="auto",
                trust_remote_code=True,
                device_map=current_device,
                attn_implementation=args.attn_impl,
            )
            omics_one.model = qwen_model


        with time_count("Loaded protein model"):
            protein_model = AutoModelForMaskedLM.from_pretrained(
                args.protein_model_path,
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
                device_map=current_device,
            )
            omics_one.protein_model = protein_model

        print_rank_0("Base models loaded successfully")

    # 4. 冻结多组学Encoder的参数
    if args.freeze_bio:
        print_rank_0("Freezing DNA/RNA and protein model parameters")
        for p in omics_one.dna_rna_model.parameters():
            p.requires_grad = False
        for p in omics_one.protein_model.parameters():
            p.requires_grad = False

    total = sum(p.numel() for p in omics_one.parameters())
    print_rank_0(f"Total parameters: {total:,}")

    return omics_one, model_config


def setup_dataloaders(args, tokenizer, dna_rna_tokenizer, protein_tokenizer):
    """
    Setup training and evaluation dataloaders using OmicsDataset.
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
        dna_rna_k_tokens=args.dna_rna_k_tokens,
        protein_k_tokens=args.protein_k_tokens,
        mode=args.mode,
        padding=True,
        input_field="input",
        output_field="output",
    )

    # 创建训练数据集
    print_rank_0(f"Loading training dataset from {args.train_dataset_path}")
    train_dataset = OmicsDataset(
        parquet_file=args.train_dataset_path,
        tokenizer=tokenizer,
        dataset_config=train_config,
        dna_rna_tokenizer=dna_rna_tokenizer,
        protein_tokenizer=protein_tokenizer,
        read_nums=args.read_nums,
        shuffle=True,
        seed=42,
        type="Train",
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
            dna_rna_k_tokens=args.dna_rna_k_tokens,
            protein_k_tokens=args.protein_k_tokens,
            input_field="input",
            output_field="output",
        )

        eval_dataset = OmicsDataset(
            parquet_file=args.eval_dataset_path,
            tokenizer=tokenizer,
            dataset_config=eval_config,
            dna_rna_tokenizer=dna_rna_tokenizer,
            protein_tokenizer=protein_tokenizer,
            read_nums=args.eval_read_nums,
            shuffle=False,
            seed=42,
            type="Eval",
            num_workers=args.dataloader_num_workers,
        )

    return train_dataset, eval_dataset


def main():
    parser = ArgumentParser()
    # Log
    parser.add_argument(
        "--experiment-name",
        type=str,
        default="Qwen_NT_sft_exp_",
        help="Experiment name for logging and checkpoints",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output path for saving models and logs",
    )
    parser.add_argument("--swanlab", action="store_true", help="Enable swanlab logging")
    parser.add_argument(
        "--report_to",
        type=str,
        nargs="+",
        default=["swanlab"],
        choices=["swanlab", "tensorboard", "wandb", "mlflow", "neptune"],
        help="Reporting tool(s) for logging",
    )
    parser.add_argument(
        "--swanlab-team", type=str, default=None, help="Swanlab team name"
    )
    parser.add_argument(
        "--swanlab-project", type=str, default=None, help="Swanlab project name"
    )
    parser.add_argument("--test-code", action="store_true", help="Test mode flag")
    parser.add_argument(
        "--profile-log-dir", type=str, default=None, help="Profile log directory"
    )
    parser.add_argument(
        "--global-rank",
        default=-1,
        type=int,
        help="Global rank for distributed training",
    )

    # Model
    parser.add_argument(
        "--text-model-path", type=str, required=True, help="Path to the Qwen model"
    )
    parser.add_argument(
        "--dna-rna-model-path",
        type=str,
        required=True,
        help="Path to the DNA-BERT model",
    )
    parser.add_argument(
        "--dna-rna-k-tokens",
        type=int,
        default=64,
        help="Number of tokens for DNA sequence projection",
    )
    parser.add_argument(
        "--protein-model-path",
        type=str,
        default=None,
        help="Path to the protein encoder checkpoint",
    )
    parser.add_argument(
        "--protein-k-tokens",
        type=int,
        default=64,
        help="Number of tokens for protein sequence projection",
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--no-load-pretrained",
        action="store_true",
        default=False,
        help="Do not load pretrained parameters for both models, use random weight instead",
    )
    parser.add_argument(
        "--load_best_model_at_end",
        action="store_true",
        help="Load the best model at the end of training",
    )
    parser.add_argument(
        "--greater_is_better",
        action="store_true",
        help="Load the best model at the end of training",
    )
    parser.add_argument(
        "--freeze-bio",
        action="store_true",
        default=True,
        help="Freeze the parameters of the DNA/RNA and protein models",
    )
    parser.add_argument(
        "--bf16", action="store_true", default=False, help="Use bfloat16 precision"
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        default=False,
        help="Use mixed precision training with fp16",
    )

    # Dataset
    parser.add_argument(
        "--train-dataset-path",
        type=str,
        required=True,
        help="Path to training dataset (parquet format)",
    )
    parser.add_argument(
        "--eval-dataset-path",
        type=str,
        default=None,
        help="Path to evaluation dataset (parquet format)",
    )
    parser.add_argument(
        "--max-len", type=int, default=1024, help="Maximum sequence length"
    )
    parser.add_argument(
        "--max-src-len", type=int, default=1024, help="Maximum source sequence length"
    )
    parser.add_argument("--eval-max-len", type=int, default=1024)
    parser.add_argument("--eval-max-src-len", type=int, default=1024)
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument(
        "--read-nums", type=int, default=None, help="Number of samples to read"
    )
    parser.add_argument(
        "--eval-read-nums",
        type=int,
        default=None,
        help="Number of evaluation samples to read",
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help="Number of workers for data loading",
    )

    # Dataset compatibility parameters
    parser.add_argument(
        "--prefix", type=str, default=None, help="Prefix added to the input"
    )
    parser.add_argument(
        "--postfix", type=str, default=None, help="Postfix added to the input"
    )
    parser.add_argument(
        "--meta-prompt", type=str, default=None, help="Systematic prompt for input"
    )
    parser.add_argument(
        "--batching-stretegy",
        type=str,
        default="padding",
        choices=["padding", "packing"],
        help="Strategy for batching dataset",
    )
    parser.add_argument(
        "--all-reduce-loss", action="store_true", help="Reduce loss across GPUs"
    )

    # Training configuration
    parser.add_argument(
        "--mode",
        type=str,
        default="sft",
        choices=["pretrain", "sft"],
        help="Training mode",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        required=True,
        help="Batch size per GPU",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=4,
        help="Evaluation batch size per GPU",
    )
    parser.add_argument(
        "--num_train_epochs", type=int, required=True, help="Number of training epochs"
    )
    parser.add_argument(
        "--train-iters",
        type=int,
        default=None,
        help="Total number of training iterations (alternative to epochs)",
    )
    parser.add_argument(
        "--save_strategy",
        type=str,
        required=True,
        choices=["no", "steps", "epoch"],
        help="Strategy for saving checkpoints",
    )
    parser.add_argument(
        "--save_steps", type=int, default=10000, help="Steps between model saves"
    )
    parser.add_argument(
        "--eval_strategy",
        type=str,
        required=True,
        choices=["no", "steps", "epoch"],
        help="Strategy for evaluation",
    )
    parser.add_argument(
        "--eval_steps", type=int, default=10000, help="Steps between evaluations"
    )
    parser.add_argument(
        "--logging_strategy",
        type=str,
        required=True,
        choices=["no", "steps", "epoch"],
        help="Strategy for logging",
    )
    parser.add_argument(
        "--logging_steps", type=int, default=10000, help="Steps between loss logging"
    )
    parser.add_argument(
        "--enable-list",
        nargs="+",
        type=str,
        default=None,
        help="List of enabled parameters",
    )
    parser.add_argument(
        "--save_trainable",
        type=bool,
        default=True,
        help="Save trainable parameters only",
    )
    parser.add_argument(
        "--save_only_model", action="store_true", help="Save only model parameters"
    )
    parser.add_argument("--if_train_llm", type=bool, default=True, help="If train llm")

    # Optimizer configuration
    parser.add_argument(
        "--learning_rate", type=float, required=True, help="Learning rate"
    )
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=1,
        help="Gradient accumulation steps",
    )
    parser.add_argument("--warmup_ratio", type=float, default=0.01, help="Warmup ratio")
    parser.add_argument("--weight-decay", type=float, default=1e-2, help="Weight decay")
    parser.add_argument("--eps", type=float, default=1e-8, help="Epsilon for optimizer")
    parser.add_argument(
        "--betas",
        nargs="+",
        type=float,
        default=[0.9, 0.999],
        help="Beta parameters for optimizer",
    )

    # Early stopping
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=10,
        help="Patience for early stopping",
    )
    parser.add_argument(
        "--metric-for-best-model",
        type=str,
        default="eval_loss",
        help="Metric to track for model selection",
    )

    # LoRA training
    parser.add_argument(
        "--use-lora",
        action="store_true",
        help="Whether to use LoRA for parameter-efficient training",
    )

    parser.add_argument(
        "--save-total-limit",
        type=int,
        default=10,
        help="Number of total checkpoints to keep",
    )

    parser.add_argument(
        "--local_rank", type=int, default=-1, help="Local rank for distributed training"
    )
    parser.add_argument(
        "--save_safetensors",
        type=bool,
        default=False,
        help="Save model in safetensors format",
    )
    parser.add_argument("--lora_r", type=int, default=64, help="LoRA rank")

    # Training Optimization
    parser.add_argument("--attn_impl", type=str, default='sdpa', choices=['sdpa', 'flash_attention_2'], help="FlashAttn Implementation, support none or fa2")

    # Add DeepSpeed arguments
    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()

    # Calculate GPU count for DeepSpeed
    if dist.is_initialized():
        args.gpu_count = dist.get_world_size()
    else:
        args.gpu_count = 1

    # Add clip_grad_max_norm if not present
    if not hasattr(args, "clip_grad_max_norm"):
        args.clip_grad_max_norm = 1.0
    writer = None
    try:
        # Initialize distributed training
        deepspeed.init_distributed()

        # Set global_rank to current process rank
        args.global_rank = dist.get_rank()

        # Setup logging
        writer = None
        if args.global_rank == 0:
            current_time = datetime.now().strftime("%y-%m-%d_%H-%M")
            if args.swanlab:
                init_swanlab_rank_0(args, experiment_suffix=current_time)

        # Setup tokenizers
        tokenizer, dna_rna_tokenizer, protein_tokenizer = setup_tokenizers(args)

        # Setup model and optimizer
        model, _ = setup_model_and_optimizer(args, tokenizer)

        # Get dataloaders and convert to datasets for Transformers Trainer
        train_dataset, eval_dataset = setup_dataloaders(
            args, tokenizer, dna_rna_tokenizer, protein_tokenizer
        )

        # Apply parameter freezing or pre-train lora based on args.use-lora
        if args.use_lora:
            print_rank_0("Using LoRA for parameter-efficient training")
            model = pre_train_lora(model, args)
        else:
            print_rank_0("Using full parameter fine-tuning")
            set_up_trainable_param(model, args)

        # 将所有参数打印出来以进行调试
        if args.global_rank == 0:
            print_rank_0("-------- Training Configuration --------")
            print_rank_0(f"Model: {args.text_model_path} + {args.dna_rna_model_path}")
            print_rank_0(f"DNA/RNA model tokens: {args.dna_rna_k_tokens}")
            print_rank_0(f"Protein model tokens: {args.protein_k_tokens}")
            print_rank_0(f"Batch size: {args.per_device_train_batch_size}")
            print_rank_0(f"Learning rate: {args.learning_rate}")
            print_rank_0(f"Dataset: {args.train_dataset_path}")
            if args.swanlab:
                print_rank_0("SwanLab logging: Enabled")

        args.deepspeed = args.deepspeed_config

        try:
            trainer = OmicsTrainer(
                model=model,
                args=args,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                tokenizer=tokenizer,
                data_collator=qwen_omics_collate_fn,
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
