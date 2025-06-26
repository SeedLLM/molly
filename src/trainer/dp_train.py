import torch
import deepspeed
from argparse import Namespace
from torch.profiler import record_function
from deepspeed.runtime.pipe.engine import DeepSpeedEngine

from contextlib import contextmanager
import torch.distributed as dist

def reduce_tensor(tensor, world_size):
    rt = tensor.detach().clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= world_size
    return rt

def to_device(batch, device):
    """
    Move every pytorch tensor in the batch data to device for training.
    """
    output = {}
    for k, v in batch.items():
        try:
            output[k] = v.to(device)
        except:
            output[k] = v
    return output

@contextmanager
def gather_params_ctx(param, modifier_rank: int = 0, fwd_module: torch.nn.Module = None):
    """Call DeepSpeed GatheredParameters context manager if DeepSpeed is enabled, otherwise do nothing."""

    with deepspeed.zero.GatheredParameters(param, modifier_rank=modifier_rank, fwd_module=fwd_module):
        yield
    return

def forward_step_deepspeed(model: DeepSpeedEngine, data_loader, args: Namespace, step: int):
    with torch.profiler.record_function("get_data"):
        batch = next(data_loader)
        batch = to_device(batch, args.device)

    with torch.profiler.record_function("forward_path"):
        loss, metric = model(**batch)

        if args.all_reduce_loss:
            # Reduce loss for average loss print, not for backpropagation.
            # DeepSpeed uses on-chip loss for backpropagation and all-reduces gradients afterwards.
            loss_reduced = reduce_tensor(loss, args.world_size)
            metric['loss_reduced'] = loss_reduced
            del loss_reduced
            
        return loss, metric
    
# 🌟使用的如下
def backward_step_deepspeed(model: DeepSpeedEngine, optimizer, loss, lr_scheduler, args, step):
    with record_function("backward_path"):
        try:
            model.backward(loss)
            model.step()
        except RuntimeError as e:
            print(f"[Backward ERROR] step={step}", e)
            print(f"loss: dtype={loss.dtype}, shape={tuple(loss.shape)}, grad_fn={loss.grad_fn}")
    return model

def reduce_gradients(model, world_size):
    for param in model.parameters():
        if param.requires_grad:
            param.grad.data = reduce_tensor(param.grad.data, world_size)


def eval_step_deepspeed(model: DeepSpeedEngine, data_loader, args: Namespace, step: int):
    with record_function("eval_path"):
        batch = next(data_loader)
        batch = to_device(batch, args.device)
        with torch.no_grad():
            loss, metric = model(**batch)
            # TODO: all reduce metrics
        return loss.item(), metric
    

def task_print_ntp(all_metric, args):
    return ""

def task_print_bio(all_metric, args):
    # return the on-chip accuracy
    acc_count = sum([sub_dict.get("accuracy", 0) for sub_dict in all_metric])
    mcc_count = sum([sub_dict.get("mcc", 0) for sub_dict in all_metric])
    return f' acc:{(acc_count/args.show_loss_step) * 100}, mcc:{(mcc_count/args.show_loss_step) * 100}'
