"""
train qwen + dnabert
"""
import os
import gc
from datetime import datetime
from argparse import ArgumentParser
from typing import Optional, Union, List
import json, configparser
import logging
import traceback

import swanlab
import torch
import torch.distributed as dist
from transformers import AutoModelForCausalLM, AutoModel, AutoTokenizer
from transformers import TrainingArguments
import deepspeed

from src.trainer import CustomTrainer, forward_step_deepspeed, eval_step_deepspeed, backward_step_deepspeed, Trainer
from src.model import QwenWithBert, get_qwen_bert_config
from src.dataset import load_dataloder, RepeatingLoader
from src.utils import print_rank_0, refresh_config, get_optimizer, set_up_trainable_param

from torch.optim import AdamW
from transformers import get_scheduler

def get_optimizer_and_scheduler(args, model, num_training_steps):
    """
    使用官方 AdamW 和 scheduler 创建优化器和调度器。
    
    Args:
        args: argparse 参数
        model: 模型（建议已过滤出 requires_grad 的参数）
        num_training_steps: 总训练步数（用于调度器）

    Returns:
        optimizer, lr_scheduler
    """
    # ✅ 优化器参数组
    optimizer = AdamW(
        params=[p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        betas=tuple(args.betas),  # 例如 (0.9, 0.999)
        eps=args.eps,
        weight_decay=args.weight_decay
    )

    # ✅ 学习率调度器
    lr_scheduler = get_scheduler(
        name="linear",  # 也支持 cosine, constant, polynomial 等
        optimizer=optimizer,
        num_warmup_steps=int(args.warmup * num_training_steps),
        num_training_steps=num_training_steps
    )

    return optimizer, lr_scheduler


def get_dp_info():
    """
    获取当前进程的数据并行 rank 和数据并行总数。
    若未初始化分布式环境，则返回默认值。
    """
    if dist.is_available() and dist.is_initialized():
        dp_rank = dist.get_rank()
        num_dp_ranks = dist.get_world_size()
    else:
        dp_rank = 0
        num_dp_ranks = 1
    return dp_rank, num_dp_ranks


def read_config(file_path, encoding='utf-8'):
    """
    Read config file.
    """
    if file_path.endswith('.json'):
        with open(file_path, 'r', encoding=encoding) as f:
            config = json.load(f)
    elif file_path.endswith('.ini'):
        config = configparser.ConfigParser()
        config.read(file_path)
    else:
        if '.' in file_path:
            format = file_path.split('.')[-1]
        else:
            format = 'Unkown'
        raise ValueError(f"Can not read unsupported file format: {format}")
    return config


def load_local_model(args):
    return_dataset_kwargs = {}

    tokenizer = AutoTokenizer.from_pretrained(args.text_model_path)
    # 增加Special Token 🌟
    new_tokens = ["<|dna_start|>", "<|dna_pad|>", "<|dna_end|>"]
    tokenizer.add_special_tokens({"additional_special_tokens": new_tokens})
    dna_token_id = tokenizer.convert_tokens_to_ids("<|dna_pad|>")
    dna_start_id = tokenizer.convert_tokens_to_ids("<|dna_start|>")
    dna_end_id = tokenizer.convert_tokens_to_ids("<|dna_end|>")
    print_rank_0(f"dna_start_id: {dna_start_id}, dna_token_id: {dna_token_id}, dna_end_id: {dna_end_id}")

    model_config = get_qwen_bert_config(args.text_model_path, args.bio_model_path)
    # 增加生物序列Token最大长度 🌟
    model_config.project_token_num = args.multimodal_k_tokens

    # qwen的config中为bfloat16
    torch.set_default_dtype(torch.bfloat16) 
    model = QwenWithBert(model_config)

    # Load checkpoint if checkpoint path is provieded.
    if args.text_model_path is not None:
        model.model.load_state_dict(AutoModelForCausalLM.from_pretrained(args.text_model_path).state_dict())

    if args.bio_model_path is not None:
        return_dataset_kwargs['multimodal_k_tokens'] = args.multimodal_k_tokens
        model.bio_model.load_state_dict(AutoModel.from_pretrained(args.bio_model_path, config=model_config.multimodal_model_config, trust_remote_code=True).state_dict(), strict=False)
        multimodal_tokenizer = AutoTokenizer.from_pretrained(args.bio_model_path, trust_remote_code=True)
        return_dataset_kwargs['multimodal_tokenizer'] = multimodal_tokenizer

    # Convert dtype to avoid inconsistency between default dtype and checkpoint dtype.
    torch.cuda.empty_cache()
    gc.collect()
    model.to(torch.bfloat16).to(args.device)
    return model, tokenizer, model_config, return_dataset_kwargs


def get_writer(args):
    current_time = datetime.now().strftime('%y-%m-%d_%H-%M')
    if not args.test_code and args.global_rank == 0:
        # 🌟修改为SwanLab
        if args.swanlab:
            os.environ['WANDB_CACHE_DIR'] = args.wandb_cache_dir
            os.environ['WANDB_DIR'] = args.wandb_dir
            swanlab.login(api_key='7BZRyWx1ftGxsthmlgZ1Q', save=True)
            swanlab.init(
                project=args.wandb_project,
                experiment_name=args.experiment_name + current_time,
                config=args
            )
        elif args.tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter
            except ImportError:
                from tensorboard import SummaryWriter
            log_dir = os.path.join(args.tb_log_dir, args.experiment_name + current_time)
            return SummaryWriter(log_dir=log_dir)
        return None

if __name__ == "__main__":
    parser = ArgumentParser()
    # logo info
    parser.add_argument('--experiment-name', type=str, default='Qwen_DNABERT_sft_exp_', 
                       help='The name of the experiment for summary and checkpoint.')
    parser.add_argument('--tensorboard', action='store_true',
                        help='Set this to enable tensorboard logging.')
    parser.add_argument('--tb-log-dir', type=str, default=None,
                        help='Path of tensorboard log dir')
    parser.add_argument('--swanlab', action='store_true',
                        help='Set this to enable wandb logging.')
    parser.add_argument('--swanlab-api-key', type=str, default=None,
                        help='API key of wandb.')
    parser.add_argument('--swanlab-team', type=str, default=None,
                        help='Team of wandb.')
    parser.add_argument('--swanlab-project', type=str, default=None,
                        help='Project of wandb.')
    parser.add_argument('--swanlab-cache-dir', type=str, default=None,
                        help='Cache dir of swanlab')
    parser.add_argument('--swanlab-dir', type=str, default=None,
                        help='Dir of swanlab')
    parser.add_argument('--test-code', type=bool, default=True, help='add this argument to avoid creating log file.')
    parser.add_argument('--profile-log-dir', type=str, default=None,   
                        help='Path of profiler log dir')
    parser.add_argument('--global-rank', default=-1, type=int, 
                      help='global rank')
    parser.add_argument('--output-path', default="", type=str, 
                      help='save path')
    # model info
    parser.add_argument('--text-model-path', type=str, default=None,   
                        help='Path of text llm path')
    parser.add_argument('--bio-model-path', type=str, default=None,   
                        help='Path of bio embedding model path')
    parser.add_argument('--multimodal-k-tokens', type=int, default=64,   
                        help='max number of bio squence tokens')
    parser.add_argument('--device', type=str, default="cuda")
    
    # dataset info
    parser.add_argument('--train-dataset-path', type=str, default="",   
                        help='Path of train dataset')
    parser.add_argument('--eval-dataset-path', type=str, default="",   
                        help='Path of eval dataset')
    parser.add_argument('--max-len', type=int, default=1024, help='Maximum length of tokens for a single data sample')
    parser.add_argument('--max-src-len', type=int, default=1024, help='Maximum length of input tokens')
    parser.add_argument('--eval-max-len', type=int, default=1024)
    parser.add_argument('--eval-max-src-len', type=int, default=1024)
    parser.add_argument('--skip-eval', action='store_true')
    parser.add_argument('--read-nums', type=int, default=None,
                       help='The number of data to read. If this value is None, the dataset will read all data')
    parser.add_argument('--eval-read-nums', type=int, default=None,
                       help='The number of evaluation data to read. If this value is None, the dataset will read all data')
    # 应该可以去除
    parser.add_argument('--prefix', type=str, default=None,
                       help='The prefix added to the input')
    parser.add_argument('--postfix', type=str, default=None,
                       help='The postfix added to the input')
    parser.add_argument('--batching-stretegy', type=str, default='padding', choices=['padding', 'packing'],
                       help='The stretegy for batching dataset')
    parser.add_argument('--meta-prompt', type=str, default=None,
                    help='The systematic prompt for the input')
    # parser.add_argument('--dataset-weights', type=int, nargs='+', default=None)
    # parser.add_argument('--read-start-step', type=int, default=None)

    # training
    parser.add_argument('--mode', type=str, default='sft', choices=['pretrain', 'sft', 'dual_rl', 'rlhf'],
                       help='The training mode')
    parser.add_argument('--batch-size-per-gpu', type=int, default=4, 
                       help='Batch size on a single GPU. batch-size * world_size = total batch_size.')
    parser.add_argument('--eval-batch-size-per-gpu', type=int, default=4, 
                       help='Evaluation batch size on a single GPU. batch-size * world_size = total batch_size.')
    parser.add_argument('--epochs', type=int, default=1, help='Train epoch')
    parser.add_argument('--save-interval', type=int, default=10000, help='save')
    parser.add_argument('--eval-interval', type=int, default=10000, help='save')
    parser.add_argument('--show_avg_loss_step', type=int, default=10000, help='save')
    parser.add_argument('--enable-list', nargs='+', type=str, default=None,
                    help='List of enabled parameters')

    # deepspeed
    parser.add_argument('--ds-config-path',type=str, help='path of ds configuration file')
    parser.add_argument('--local_rank', type=int, default=-1, help='Local rank passed from deepspeed launcher')

    # optimizer
    parser.add_argument('--gradient-accumulation-steps', type=int, default=1, 
                       help='Run optimizer after every gradient-accumulation-steps backwards')
    parser.add_argument('--warmup-min-lr', type=float, default=1.0e-5, 
                       help='Minimum learning rate for deepspeed warmup configurations')
    parser.add_argument('--warmup-max-lr', type=float, default=2.0e-4, 
                       help='Maximum learning rate for deepspeed warmup configurations')
    parser.add_argument('--warmup', type=float, default=0.01, 
                       help='Percentage of data to warm up on (.01 = 1% of all training iters). Default 0.01')
    parser.add_argument('--lr', type=float, default=1.0e-4, help='Initial learning rate')
    parser.add_argument('--clip-grad-max-norm', type=float, default=1.0,
                       help='Threshold norm value for gradient')
    # 用途：通常用于 多 GPU 分布式训练中同步 loss（对所有 GPU 的 loss 求平均）。
    parser.add_argument('--all-reduce-loss', action='store_true')
    parser.add_argument('--weight-decay', type=float, default=5e-4, 
                       help='Weight decay coefficient for L2 regularization')
    parser.add_argument('--eps', type=float, default=1e-8, 
                       help='Initial epsilon for the optimizer')
    parser.add_argument('--betas', nargs='+', type=float, default=[0.9,0.999], 
                       help='Initial beta values for the optimizer')
    
    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()

    writer = get_writer(args=args)
    deepspeed.init_distributed() 

    print_rank_0("-------------------init model-------------------------")
    model, tokenizer, model_config, return_dataset_kwargs = load_local_model(args)

    print_rank_0("-------------------init dataset-----------------------")
    dp_rank, num_dp_ranks = get_dp_info()
    print_rank_0(dp_rank, num_dp_ranks)
    train_dataloader = load_dataloder(args, tokenizer, dp_rank, num_dp_ranks, return_dataset_kwargs, True)
    eval_dataloader = load_dataloder(args, tokenizer, dp_rank, num_dp_ranks, return_dataset_kwargs, False)

    ds_config = read_config(args.ds_config_path, encoding=None)
    ds_config = refresh_config(ds_config, args)

    set_up_trainable_param(model, args)

    # 这里全是None
    optimizer_sd, lr_scheduler_sd = getattr(model_config, 'optmizer_sd', None), getattr(model_config, 'lr_scheduler_sd', None)
    # optimizer, lr_scheduler = get_optimizer(ds_config=ds_config, 
    #                                     args=args, 
    #                                     model=model, 
    #                                     optimizer_sd=optimizer_sd, 
    #                                     lr_scheduler_sd=lr_scheduler_sd)

    _, lr_scheduler = get_optimizer_and_scheduler(args, model, 60000)

    print(lr_scheduler, "show, show before")
    from deepspeed.ops.adam import DeepSpeedCPUAdam

    optimizer = DeepSpeedCPUAdam(
        model.parameters(),
        lr=args.lr,
        betas=tuple(args.betas),
        eps=args.eps,
        weight_decay=args.weight_decay
    )
    model, optimizer, _, lr_scheduler = deepspeed.initialize(model=model, 
                                                optimizer=optimizer, 
                                                lr_scheduler=lr_scheduler,
                                                config=ds_config, 
                                                model_parameters=[p for p in model.parameters() if p.requires_grad],
                                                mpu=None)

    forward_step = forward_step_deepspeed
    eval_step = eval_step_deepspeed
    backward_step = backward_step_deepspeed

    print(optimizer, lr_scheduler, "show, show")

    trainer = Trainer(args, writer)

    # training_args = TrainingArguments(
    #     output_dir=args.output_path,  # 可以从 args 中拿
    #     per_device_train_batch_size=args.batch_size_per_gpu,
    #     per_device_eval_batch_size=args.eval_batch_size_per_gpu,
    #     gradient_accumulation_steps=args.gradient_accumulation_steps,
    #     num_train_epochs=args.epochs,
    #     max_steps=1000,
    #     logging_dir=args.tb_log_dir if args.tensorboard else None,
    #     eval_strategy="steps",
    #     save_strategy="steps",
    #     save_steps=args.save_interval if args.save_interval else 1000,
    #     eval_steps=args.eval_interval if hasattr(args, 'eval_interval') else 1000,
    #     logging_steps=args.show_avg_loss_step,
    #     deepspeed=args.ds_config_path,
    #     report_to="none",  # 如果用 swanlab 自定义日志系统
    # )
    # # fp16=True if torch.cuda.is_available() else False,
    # trainer = CustomTrainer(
    #     model=model,
    #     args=training_args,           # 需要你提前定义 TrainingArguments
    #     train_dataset=train_dataloader,  # 这里不是 Dataloader 而是 Dataset
    #     eval_dataset=eval_dataloader,
    #     tokenizer=tokenizer,
    #     args_namespace=args         # 原始 argparse 参数保留
    # )
    def train_with_profiler(profiler):
        trainer.train(
            model=model,
            train_data_loader=RepeatingLoader(train_dataloader),
            eval_data_loader=None if eval_dataloader is None else RepeatingLoader(eval_dataloader),
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            forward_step=forward_step,
            backward_step=backward_step,
            eval_step=eval_step,
            profiler=profiler,
            log_loss=True
        )

    try:
        profiler = None
        train_with_profiler(None)
    except Exception:
        # When any error occurs during the training process, log the error.
        # Note that only the error occured in the rank 0 will be logged into file.
        traceback_info = traceback.format_exc()
        if args.global_rank == 0:
            print_rank_0(traceback_info, args.global_rank, logging.ERROR)
        else:
            print(traceback_info)

    if dist.is_initialized():
        dist.destroy_process_group()


    