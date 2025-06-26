from functools import partial

import torch.optim as optim
import deepspeed.ops as ds_optim
import deepspeed

def print_rank_0(*args, **kwargs):
    if deepspeed.comm.get_rank() == 0:
        print(*args, **kwargs)

def refresh_config(ds_config, args):
    """
    Refresh the deepspeed config from args.
    The deepspeed config originally read from static config file.
    """
    ds_config['gradient_accumulation_steps'] = args.gradient_accumulation_steps
    ds_config['train_micro_batch_size_per_gpu'] = args.batch_size_per_gpu
    ds_config['optimizer']['params']['lr'] = args.lr
    ds_config["optimizer"]["scheduler"]["params"]["warmup_num_steps"] = args.num_warmup_steps
    ds_config["gradient_clipping"] = args.clip_grad_max_norm
    if 'train_batch_size' in ds_config:
        ds_config['train_batch_size'] = args.batch_size_per_gpu * args.gpu_count
    
    ds_config["fp16"]["enabled"] = False
    ds_config["bf16"]["enabled"] = True
    return ds_config

def get_optimizer_type(args, ds_config):
    if 'optimizer' in ds_config:
        return ds_config['optimizer'].get('type', 'adamw').lower()
    return None

def get_optimizer_instance(optim_type, args, model):
    return get_regular_optimizer(optim_type, args, model)

def get_optimizer(ds_config, args, model, optimizer_sd = None, lr_scheduler_sd = None):
    """
    Set up optimizer and learning rate scheduler.

    If `args.diy_optimizer == True` then optimzer will be created according to args.
    else deepseed will create optimizer for you according to ds_config.

    This function provide clear optimizer prepare process and can adjust the parameter groups if needed.
    """
    optim_type = get_optimizer_type(args, ds_config)
    offload_config = ds_config["zero_optimization"].get("offload_optimizer", {})
    offload_device = offload_config.get("device", None)
    if offload_device == 'cpu' or args.offload_optimizer:
        optim_type = 'cpu' + optim_type
    isSuccess, optimizer = get_optimizer_instance(optim_type, args, model)

    if isSuccess:
        if 'optimizer' in ds_config:
            del ds_config['optimizer']
        print_rank_0(f'--->Deepspeed optimizer setting has been overwritten', args.global_rank)
    else:
        print_rank_0(f'--->Try to use diy optimizer failed, use the ds setting', args.global_rank)
        return None, None

    lr_scheduler = get_learning_rate_scheduler(optimizer, 0, args)

    if all([optimizer, lr_scheduler, optimizer_sd, lr_scheduler_sd]):
        optimizer.load_state_dict(optimizer_sd)
        lr_scheduler.load_state_dict(lr_scheduler_sd)
    elif any([optimizer_sd, lr_scheduler_sd]):
        print_rank_0(f'--->Optimizer state dict and lr scheduler state dict have not been loaded as optimizer or lr scheduler is None', args.global_rank)

    return optimizer, lr_scheduler

def get_regular_optimizer(optim_type, args, model):
    try:
        params = [{'params':[p for p in model.parameters() if p.requires_grad], 'lr': 1}]

        optimizer_class = {
            'adamw': partial(ds_optim.adam.FusedAdam, adam_w_mode=True),
            'adam': partial(ds_optim.adam.FusedAdam, adam_w_mode=False),
            'cpuadamw':partial(ds_optim.adam.DeepSpeedCPUAdam, adamw_mode=True),
            'cpuadam':partial(ds_optim.adam.DeepSpeedCPUAdam, adamw_mode=False),
            'adamax': optim.Adamax,
            'sparseadam': optim.SparseAdam,
            'torchadam': optim.Adam,
            'torchadamw': optim.AdamW
        }.get(optim_type)
        
        if optimizer_class is None:
            raise NotImplementedError('only support adam and its variants for now')
        
        optimizer = optimizer_class(params,
                                    lr=args.lr,
                                    weight_decay=args.weight_decay,
                                    eps=args.eps,
                                    betas=tuple(args.betas))
        isSuccess = True
    except Exception as e:
        print_rank_0(f'--->Load local optimizer error as e: {e}', args.global_rank)
        isSuccess = False
        optimizer = None
    return isSuccess, optimizer


def get_learning_rate_scheduler(optimizer, iteration, args):
    init_step = max(iteration - args.num_warmup_steps, 0)
    print(optimizer, "show optimizeroptimizeroptimizer")
    if optimizer is not None:
        lr_scheduler = None
    else:
        lr_scheduler = None
    return lr_scheduler
    
def disable_untrainable_params(model,unable_list):
    for n, p in model.named_parameters():
        flag = False
        for e in unable_list:
            if e.lower() in n.lower():
                flag = True
                break
        if not flag:
            p.requires_grad_(True)
        else:
            p.requires_grad_(False)

def enable_trainable_params(model,enable_list):
    for n, p in model.named_parameters():
        flag = False
        for e in enable_list:
            if e.lower() in n.lower():
                flag = True
                break
        if not flag:
            p.requires_grad_(False)
        else:
            p.requires_grad_(True)

def set_up_trainable_param(model, args):
    """
    Set up trainable parameters of the model according to `args.enable_list` and `args_diable_list`
    and print trainable paramters.

    `args.enable_list` will be considered at first.
    if `args.enable_list` is None, then `args.diable_list` will be considered.
    if both enable_list and disable_list are None, then all parameters will be set to be trainable.

    For example:
        if `args.enable_list == ['wq']` then wq will be trainable and other weights are not.

        if `args.enable_list is None` and `args.diable_list == ['tok_embeddings']` then tok_embeddings
        will be disabled and other weights are trainable
    """
    if args.enable_list is not None:
        enable_trainable_params(model, args.enable_list)
    else:
        print_rank_0('--->All parameters will be set to trainable as both `args.enable_list` and `args.diable_list` are None',
                     args.global_rank)
        disable_untrainable_params(model, [])