from functools import partial

import deepspeed
import deepspeed.ops as ds_optim
import swanlab
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from torch import nn, optim, device as torch_device
import os
import time
from contextlib import contextmanager
from transformers.utils import is_torch_cuda_available
import torch.distributed as dist
import argparse, texttable, shutil, re

def is_main_process():
    return not dist.is_initialized() or dist.get_rank() == 0

def print_rank_0(*args, **kwargs):
    """
    仅在rank为0的进程上打印消息
    无需传入global_rank参数，自动使用deepspeed.comm.get_rank()获取当前进程排名
    如果deepspeed未初始化，则直接打印
    """
    try:
        if not deepspeed.comm.is_initialized():
            # 如果deepspeed未初始化，直接打印
            print(*args, **kwargs)
        elif deepspeed.comm.get_rank() == 0:
            # 如果是rank 0进程，打印
            print(*args, **kwargs)
    except (ImportError, AttributeError, RuntimeError):
        # 如果deepspeed未安装或者发生其他错误，直接打印
        print(*args, **kwargs)


@contextmanager
def time_count(name="block"):
    start = time.perf_counter()
    yield
    elapsed = time.perf_counter() - start
    # print_rank_0(f"[{name}] took {elapsed:.3f} s")
    print(f"[{name}] took {elapsed:.3f} s")


def get_current_device() -> "torch.device":
    r"""Get the current available device."""
    if is_torch_cuda_available():
        device = "cuda:{}".format(os.getenv("LOCAL_RANK", "0"))
    else:
        device = "cpu"

    return torch_device(device)


def swanlab_log_rank_0(metrics, step, args=None):
    """
    只在rank 0进程执行SwanLab日志记录

    Args:
        metrics: 要记录的指标字典
        step: 当前步骤
        args: 参数对象，用于检查swanlab是否启用
    """
    if args is None or not args.swanlab:
        return

    try:
        # 检查是否是rank 0进程
        is_rank_0 = True
        if deepspeed.comm.is_initialized():
            is_rank_0 = deepspeed.comm.get_rank() == 0

        if is_rank_0:
            swanlab.log(metrics, step)
            print_rank_0(f"SwanLab logged metrics at step {step}")
    except NotImplementedError as e:
        print_rank_0("NotImplementedError: " + str(e))
    except ValueError as e:
        print_rank_0("ValueError: " + str(e))


def init_swanlab_rank_0(args, experiment_suffix=""):
    """
    只在rank 0进程初始化SwanLab

    Args:
        args: 参数对象，包含swanlab相关设置
        experiment_suffix: 实验名称后缀

    Returns:
        bool: 是否成功初始化
    """
    if not args.swanlab:
        return False

    try:
        # 检查是否是rank 0进程
        is_rank_0 = True
        if deepspeed.comm.is_initialized():
            is_rank_0 = deepspeed.comm.get_rank() == 0

        if is_rank_0:
            print_rank_0("Setting up SwanLab logging...")
            print_rank_0(
                f"SwanLab project: {args.swanlab_project}, team: {args.swanlab_team}"
            )

            # 登录
            print(args.swanlab_mode, f"Swanlab model {args.swanlab_mode}")
            if args.swanlab_mode == "cloud":
                swanlab.login(api_key="", save=True)
                print_rank_0("SwanLab login successful")

            # 初始化
            swanlab.init(
                project=args.swanlab_project,
                mode=args.swanlab_mode,
                experiment_name=args.experiment_name + experiment_suffix,
                config=vars(args),
            )
            print_rank_0("SwanLab initialized successfully")
            return True
    except ValueError as e:
        print_rank_0(f"Error initializing SwanLab: {str(e)}")
        return False
    return False


def refresh_config(ds_config, args):
    """
    Refresh the deepspeed config from args.
    The deepspeed config originally read from static config file.
    """
    ds_config["gradient_accumulation_steps"] = args.gradient_accumulation_steps
    ds_config["train_micro_batch_size_per_gpu"] = args.batch_size_per_gpu

    # Update optimizer parameters
    if "optimizer" in ds_config:
        ds_config["optimizer"]["params"]["lr"] = args.lr
        if "betas" in args:
            ds_config["optimizer"]["params"]["betas"] = args.betas
        if "eps" in args:
            ds_config["optimizer"]["params"]["eps"] = args.eps
        if "weight_decay" in args:
            ds_config["optimizer"]["params"][
                "weight_decay"] = args.weight_decay

    # Update scheduler parameters (now at top level)
    if "scheduler" in ds_config:
        if hasattr(args, "num_warmup_steps"):
            ds_config["scheduler"]["params"][
                "warmup_num_steps"] = args.num_warmup_steps
        if hasattr(args, "warmup_min_lr"):
            ds_config["scheduler"]["params"][
                "warmup_min_lr"] = args.warmup_min_lr
        if hasattr(args, "warmup_max_lr"):
            ds_config["scheduler"]["params"]["warmup_max_lr"] = args.lr

    # Set gradient clipping if available
    if hasattr(args, "clip_grad_max_norm"):
        ds_config["gradient_clipping"] = args.clip_grad_max_norm

    # Update batch size if needed
    if "train_batch_size" in ds_config and hasattr(args, "gpu_count"):
        ds_config[
            "train_batch_size"] = args.batch_size_per_gpu * args.gpu_count

    # Ensure using bfloat16 precision
    if "fp16" in ds_config:
        ds_config["fp16"]["enabled"] = False
    if "bf16" in ds_config:
        ds_config["bf16"]["enabled"] = True

    return ds_config


def get_optimizer_type(args, ds_config):
    if "optimizer" in ds_config:
        return ds_config["optimizer"].get("type", "adamw").lower()
    return None


def get_optimizer_instance(optim_type, args, model):
    return get_regular_optimizer(optim_type, args, model)


def get_optimizer(ds_config,
                  args,
                  model,
                  optimizer_sd=None,
                  lr_scheduler_sd=None):
    """
    Set up optimizer and learning rate scheduler.

    If `args.diy_optimizer == True` then optimzer will be created according to args.
    else deepseed will create optimizer for you according to ds_config.

    This function provide clear optimizer prepare process and can adjust the parameter groups if needed.
    """
    optim_type = get_optimizer_type(args, ds_config)
    offload_config = ds_config["zero_optimization"].get(
        "offload_optimizer", {})
    offload_device = offload_config.get("device", None)
    if offload_device == "cpu" or args.offload_optimizer:
        optim_type = "cpu" + optim_type
    isSuccess, optimizer = get_optimizer_instance(optim_type, args, model)

    if isSuccess:
        if "optimizer" in ds_config:
            del ds_config["optimizer"]
        print_rank_0("--->Deepspeed optimizer setting has been overwritten")
    else:
        print_rank_0("--->Try to use diy optimizer failed, use the ds setting")
        return None, None

    lr_scheduler = get_learning_rate_scheduler(optimizer, 0, args)

    if all([optimizer, lr_scheduler, optimizer_sd, lr_scheduler_sd]):
        optimizer.load_state_dict(optimizer_sd)
        lr_scheduler.load_state_dict(lr_scheduler_sd)
    elif any([optimizer_sd, lr_scheduler_sd]):
        print_rank_0(
            "--->Optimizer state dict and lr scheduler state dict have not been loaded"
        )

    return optimizer, lr_scheduler


def get_regular_optimizer(optim_type, args, model):
    try:
        params = [{
            "params": [p for p in model.parameters() if p.requires_grad],
            "lr": 1
        }]

        optimizer_class = {
            "adamw": partial(ds_optim.adam.FusedAdam, adam_w_mode=True),
            "adam": partial(ds_optim.adam.FusedAdam, adam_w_mode=False),
            "cpuadamw": partial(ds_optim.adam.DeepSpeedCPUAdam,
                                adamw_mode=True),
            "cpuadam": partial(ds_optim.adam.DeepSpeedCPUAdam,
                               adamw_mode=False),
            "adamax": optim.Adamax,
            "sparseadam": optim.SparseAdam,
            "torchadam": optim.Adam,
            "torchadamw": optim.AdamW,
        }.get(optim_type)

        if optimizer_class is None:
            raise NotImplementedError(
                "only support adam and its variants for now")

        optimizer = optimizer_class(
            params,
            lr=args.lr,
            weight_decay=args.weight_decay,
            eps=args.eps,
            betas=tuple(args.betas),
        )
        isSuccess = True
    except ValueError as e:
        print_rank_0(f"--->Load local optimizer error as e: {e}")
        isSuccess = False
        optimizer = None
    return isSuccess, optimizer


def get_learning_rate_scheduler(optimizer, iteration, args):
    print(optimizer, "show optimizeroptimizeroptimizer")
    if optimizer is not None:
        lr_scheduler = None
    else:
        lr_scheduler = None
    return lr_scheduler


def freeze_subtree(parent: nn.Module, name: str):
    """
    把 parent.<name> 整棵子树里的所有 Parameter 注销并重新注册为 buffer，
    从而让它们不再出现在 parent.parameters()。
    """
    # 1. 先走到目标子模块
    *prefix, last_key = name.split('.')
    mod = parent
    for k in prefix:
        k = int(k) if k.isdigit() else k
        mod = mod[k] if isinstance(mod, (nn.ModuleList, nn.Sequential)) else getattr(mod, k)

    # 2. last_key 可能是数字下标
    last_key = int(last_key) if last_key.isdigit() else last_key
    sub_module = mod[last_key] if isinstance(mod, (nn.ModuleList, nn.Sequential)) else getattr(mod, last_key)

    # 3. 现在开始摘叶子
    for pname, param in list(sub_module.named_parameters(recurse=True)):
        *path, leaf = pname.split('.')
        cur = sub_module
        for k in path:
            k = int(k) if k.isdigit() else k
            cur = cur[k] if isinstance(cur, (nn.ModuleList, nn.Sequential)) else getattr(cur, k)

        leaf = int(leaf) if leaf.isdigit() else leaf
        param_obj = cur[leaf] if isinstance(cur, (nn.ModuleList, nn.Sequential)) else getattr(cur, leaf)

        # 摘掉
        if isinstance(cur, (nn.ModuleList, nn.Sequential)):
            del cur[leaf]
        else:
            delattr(cur, leaf)
        # 注册成 buffer
        cur.register_buffer(str(leaf) if isinstance(leaf, int) else leaf,
                            param_obj.detach())

def set_up_trainable_param(model, args):

    if not args.train_bio:
        freeze_subtree(model, 'dna_rna_model')
        freeze_subtree(model, 'protein_model')

    if not args.train_mlp:
        freeze_subtree(model, 'dna_rna_projector')
        freeze_subtree(model, 'protein_projector')

    if not args.train_llm:
        freeze_subtree(model, 'model')

    for param in model.dna_rna_model.parameters():
        param.requires_grad = args.train_bio

    for param in model.protein_model.parameters():
        param.requires_grad = args.train_bio

    for param in model.dna_rna_projector.parameters():
        param.requires_grad = args.train_mlp
    for param in model.protein_projector.parameters():
        param.requires_grad = args.train_mlp

    for param in model.model.parameters():
        param.requires_grad = args.train_llm


    # masked_model = MaskParams(model, black_list=['dna_rna_model'])
    # print('visible params:', sum(1 for _ in masked_model.parameters()))
    # return masked_model

def pre_train_lora(model, args):
    for param in model.dna_rna_model.parameters():
        param.requires_grad = False

    for param in model.protein_model.parameters():
        param.requires_grad = False

    target_modules = []
    module_names = set()
    for name, module in model.model.named_modules():
        if isinstance(module, nn.Linear):
            names = name.split(".")
            target_name = names[-1]

            if target_name != "lm_head" and target_name not in module_names:
                target_modules.append(target_name)
                module_names.add(target_name)

    # Add attention-specific layers
    attention_patterns = [
        "q_proj",
        "k_proj",
        "v_proj",
        "out_proj",
        "query",
        "key",
        "value",
    ]
    for pattern in attention_patterns:
        if pattern not in module_names:
            target_modules.append(pattern)

    target_modules = list(target_modules)

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=64,
        target_modules=target_modules,
        lora_dropout=0.05,
        init_lora_weights="gaussian",
        bias="none",
        task_type="CAUSAL_LM",
    )
    model.model = prepare_model_for_kbit_training(model.model)
    model.model = get_peft_model(model.model, lora_config)
    model.model.print_trainable_parameters()

    for param in model.dna_rna_projector.parameters():
        param.requires_grad = True
    for param in model.protein_projector.parameters():
        param.requires_grad = True
    return model

def str2bool(v):
    """
    把字符串转成布尔值，用于 argparse 的 type= 参数。
    支持 true/false, yes/no, 1/0, y/n, t/f 及其大小写组合。
    """
    if isinstance(v, bool):          # 如果已经是 bool，直接返回
        return v
    if v.lower() in {"true", "t", "1", "yes", "y"}:
        return True
    elif v.lower() in {"false", "f", "0", "no", "n"}:
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")

# ---------------- 工具：剥色 + 截断 ----------------
def _strip_ansi(text: str) -> str:
    return re.sub(r'\x1b\[[0-9;]*m', '', text)

def _fit(text: str, max_len: int = 48) -> str:
    if len(_strip_ansi(text)) <= max_len:
        return text
    return text[:max_len-3] + '...'

# ---------------- 2. 彩色打印函数 ----------------
def pretty_print_args(args: argparse.Namespace):
    # ---------------- 1. 把参数按关键字分组 ----------------
    GROUP_MAP = {
        'experiment': {'experiment_name', 'output_dir', 'swanlab', 'report_to',
                    'swanlab_mode', 'swanlab_team', 'swanlab_project',
                    'test_code', 'profile_log_dir'},
        'model     ': {'text_model_path', 'dna_rna_model_path', 'dna_rna_k_tokens',
                'protein_model_path', 'protein_k_tokens', 'device',
                'no_load_pretrained', 'load_best_model_at_end', 'greater_is_better',
                'train_bio', 'train_mlp', 'train_llm', 'bf16', 'fp16'},
        'dataset   ': {'train_dataset_path', 'eval_dataset_path', 'max_len',
                    'max_src_len', 'eval_max_len', 'eval_max_src_len', 'skip_eval',
                    'read_nums', 'eval_read_nums', 'prefix', 'postfix',
                    'meta_prompt', 'batching_stretegy', 'all_reduce_loss'},
        'training  ': {'mode', 'per_device_train_batch_size', 'per_device_eval_batch_size',
                    'num_train_epochs', 'train_iters', 'save_strategy', 'save_steps',
                    'eval_strategy', 'eval_steps', 'logging_strategy', 'logging_steps',
                    'enable_list', 'save_trainable', 'save_only_model', 'if_train_llm',
                    'compute_domain_losses'},
        'optimizer ': {'learning_rate', 'gradient_accumulation_steps', 'warmup_ratio',
                    'weight_decay', 'eps', 'betas'},
        'early_stop': {'early_stopping_patience', 'metric_for_best_model'},
        'lora      ': {'use_lora', 'lora_r', 'save_total_limit', 'save_safetensors'},
        'system    ': {'local_rank', 'attn_impl', 'use_liger', 'dataloader_pin_memory',
                'seed'},
    }

    term = shutil.get_terminal_size((100, 20))
    table = texttable.Texttable(max_width=term.columns)
    table.set_deco(texttable.Texttable.HEADER | texttable.Texttable.VLINES)

    # 表头
    table.header(['Module', 'Key', 'Value'])

    # 逐组填充
    for group, keys in GROUP_MAP.items():
        first = True
        for k in sorted(keys):
            if k not in vars(args):
                continue
            v = getattr(args, k)
            # 提前截断超长列表/字符串，防止 texttable 自动折行
            v_str = str(v)
            if len(v_str) > 50:
                v_str = v_str[:47] + '...'

            table.add_row([group if first else '', k, v_str])
            first = False

    print('\n' + table.draw() + '\n')