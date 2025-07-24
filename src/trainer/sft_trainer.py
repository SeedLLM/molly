from transformers import Trainer
import torch
import swanlab
from typing import Optional, Dict, Any

import os


class CustomTrainer(Trainer):
    def __init__(self, args_namespace=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_swanlab = True

    def log(self, logs: Dict[str, float]) -> None:
        """
        Override the default log method to optionally log with SwanLab.
        """
        super().log(logs)
        if self.use_swanlab and self.args.local_rank in [-1, 0]:
            step = self.state.global_step
            swanlab.log(logs, step=step)

    def save_model(self, output_dir: Optional[str] = None, _internal_call: bool = False):
        """
        Override save_model to better support DeepSpeed model saving.
        """
        if self.args.should_save:
            output_dir = output_dir if output_dir is not None else self.args.output_dir
            self._save(output_dir)
            if self.use_swanlab:
                swanlab.log({"event": "model_saved", "path": output_dir})

    def _save(self, output_dir: str):
        """
        Save the model and tokenizer using Trainer's internal save logic.
        """
        os.makedirs(output_dir, exist_ok=True)
        torch.save(self.model.state_dict(), os.path.join(output_dir, "pytorch_model.bin"))
        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(output_dir)
        # if self.args.should_save:
        #     self.args.save(output_dir)
        #     self.state.save_to_json(os.path.join(output_dir, "trainer_state.json"))


import os
import json
import math
import torch
import logging
import time

import deepspeed
from typing import Callable
from argparse import Namespace
from torch.utils.data import DataLoader

from utils.tools import print_rank_0

import swanlab

class Timer(object):
    def __init__(self, start=None, n_round=2, iterations: Optional[int] = None):
        """
        A timer environment for loop programs.

        Args:
            start (time): Start time for the timer. If None, the current time is used.
            n_round (int): Number of decimal places to keep for time values.
            iterations (Optional[int]): The total number of iterations the loop will perform.
        """
        self.round = n_round  # Number of decimal places for time values
        self.start = round(start if start is not None else time.time(), self.round)  # Start time of the timer
        self.loop_start = None  # Start time of the current loop iteration
        self.iterations_left = iterations  # Number of iterations left

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Context manager exit method.
        Returns:
            bool: True if no exception occurred, False otherwise.
        """
        self.stop = round(time.time(), self.round)  # Stop time of the timer
        self.time_cost = self.format_time(round(self.stop - self.start, self.round))  # Total time cost
        return exc_type is None  # Return True if no exception occurred

    def average_time(self, entry):
        """
        Records the start or end time of a loop iteration.

        Args:
            entry (str): Either 'start' to record the start time or 'end' to record the end time.

        Raises:
            ValueError: If entry is not 'start' or 'end'.
            AssertionError: If 'end' is called before 'start'.
        """
        current_time = round(time.time(), self.round)  # Current time
        if entry == 'start':
            if self.loop_start is None:
                self.loop_start = current_time  # Record the start time of the loop iteration
        elif entry == 'end':
            assert self.loop_start is not None, 'Please ensure average_time("start") is used before average_time("end")'
            if self.iterations_left is not None:
                self.iterations_left -= 1  # Decrement the number of iterations left
            loop_end = current_time
            self.loop_time = round(loop_end - self.loop_start, self.round)  # Calculate the time taken for the loop iteration
            self.loop_start = None  # Reset the loop start time
        else:
            raise ValueError("Invalid entry value. Expected 'start' or 'end'.")

    def calculate_remaining_time(self):
        """
        Calculates the remaining time based on the average time per iteration and the number of iterations left.

        Returns:
            str: Formatted remaining time.
        """
        total_time_seconds = self.iterations_left * self.loop_time  # Total remaining time in seconds
        return self.format_time(total_time_seconds)

    def format_time(self, input_time):
        if input_time < 60:
            return f"{round(input_time, self.round)}s"  # Less than a minute
        elif input_time < 3600:
            minutes = input_time // 60
            seconds = input_time % 60
            return f"{minutes}min {round(seconds, self.round)}s"  # Less than an hour
        elif input_time < 86400:
            hours = input_time // 3600
            minutes = (input_time % 3600) // 60
            seconds = (input_time % 3600) % 60
            return f"{hours}h {minutes}min {round(seconds, self.round)}s"  # Less than a day
        else:
            days = input_time // 86400
            hours = (input_time % 86400) // 3600
            minutes = ((input_time % 86400) % 3600) // 60
            seconds = ((input_time % 86400) % 3600) % 60
            return f"{days}d {hours}h {minutes}min {round(seconds, self.round)}s"  # More than a day


def ensure_directory_exists(directory, global_rank=0):
    if not os.path.exists(directory) and global_rank == 0: # Only create dir when global rank is 0.
        os.makedirs(directory)
        print(f'---> Directory:{directory} is not existed. created a new floder')

# 🌟 待确认
DATA_PARALLEL_GROUP = None

def get_data_parallel_group():
    """Get the data parallel group the caller rank belongs to."""
    assert DATA_PARALLEL_GROUP is not None, 'data parallel group is not initialized'
    return DATA_PARALLEL_GROUP


class Trainer:
    def __init__(self, args, writer=None):
        self.args = args
        self.end = False
        self.writer = writer
        self.all_loss = 0.0
        self.global_step = 0
        self.all_metric = []
        self.eval_loss = None
        self.eval_metric = []
        self.best_eval_index = 0.0
        self.wait = 0
        self.save_folder = os.path.join(args.output_path, args.experiment_name)
        self.save_config = True
        self.lr = args.lr
        ensure_directory_exists(self.save_folder, self.args.global_rank)

    def train(
        self,
        model: deepspeed.DeepSpeedEngine,
        train_data_loader: DataLoader,
        forward_step: Callable,
        optimizer: Callable = None,
        lr_scheduler: Callable = None,
        eval_data_loader: DataLoader = None,
        backward_step: Callable = None,
        eval_step: Callable = None,
        profiler: Callable = None,
        log_loss: bool = False
    ) -> None:
        """
        Training loop for the model.

        Args:
            model (torch.nn.Module): Model to be trained.
            train_data_loader (DataLoader): Training data loader.
            forward_step (Callable): Forward step function.
            optimizer (Callable, optional): Optimizer function. Defaults to None.
            lr_scheduler (Callable, optional): Learning rate scheduler function. Defaults to None.
            eval_data_loader (DataLoader, optional): Evaluation data loader. Defaults to None.
            backward_step (Callable, optional): Backward step function. Defaults to None.
            eval_step (Callable, optional): Evaluation step function. Defaults to None.
            profiler (Callable, optional): Profiler function. Defaults to None.
            log_loss (bool, optional): Flag to log loss values. Defaults to False.
        """
        # Start Training
        start_step = 0
        micro_step = start_step
        global_step = start_step // self.args.gradient_accumulation_steps

        print_rank_0('--->loaded the model, start training')

        # Check if save_interval is specified
        if self.args.save_interval is None:
            print_rank_0(f'--->Checkpoint will only be saved on the last step of training as `args.save_interval` is None')

        total_print_steps = self.args.num_global_update_steps // self.args.show_avg_loss_step

        # 这里先去除
        # if self.args.save_epoch:
        #     updates_per_epoch = math.ceil(len(train_data_loader) / get_data_parallel_group())
        #     self.args.save_interval = (updates_per_epoch / self.args.gradient_accumulation_steps) * self.args.save_epoch

        if self.args.save_interval is None:
            self.args.save_interval = int(1e30)
            print_rank_0(f'--->Checkpoint will only be saved on the last step of training as `args.save_interval` is None')

        with Timer(iterations=total_print_steps) as timer:
            model.train()
            for step in range(1, self.args.num_micro_update_steps+1):
                # Timer start
                timer.average_time(entry='start')
                
                # Forward step
                loss, metric = forward_step(model, train_data_loader, self.args, step)
                self.lr = lr_scheduler.get_lr()
                self.grad_norm = model.get_global_grad_norm()
                if torch.isnan(loss) or torch.isinf(loss):
                    print("NaN/Inf loss, skipping but maintain graph for other ranks")
                    # loss.new_zeros() 会创建一个和 loss 同 device、dtype 的 0-dim Tensor
                    loss = loss.new_zeros(())
                    self.all_loss += 0.0
                    self.all_metric.append({})
                else:
                    val = loss
                    if val.dim() > 0:
                        val = val.mean()
                    self.all_loss += val.item()
                    # 是否需去除
                    self.all_metric.append(metric)

                # Backward step
                if backward_step:
                    backward_step(model, optimizer, loss, lr_scheduler, self.args, step)
                
                if profiler:
                    profiler.step()

                # 也许需要添加 🌟
                # torch.cuda.empty_cache()

                # Evaluation
                if step % self.args.eval_interval == 0 and eval_step is not None and not self.args.skip_eval:
                    with torch.no_grad():
                        assert eval_data_loader is not None, 'evaluation dataset cannot be None'
                        self.eval_loss, eval_metric = eval_step(model, eval_data_loader, self.args, step)
                        self.eval_metric.append(eval_metric)

                # Logging and saving
                self.info_manager(step, timer, log_loss)
                self.save_model(model, optimizer, lr_scheduler, train_data_loader, step)

                if self.end:
                    print_rank_0("Early stopping triggered.")
                    break

        # Final save
        self.end = True
        self.save_model(model, optimizer, lr_scheduler, train_data_loader, step)

    def earily_stop(self):
        index = self.args.earily_stop_index
        if index == 'loss':
            if self.best_eval_index is None:
                self.best_eval_index = self.eval_loss
            elif self.best_eval_index > self.eval_loss:
                self.best_eval_index = self.eval_loss
                self.wait = 0
            else:
                self.wait += 1
        else:
            eval_metric = self.eval_metric[0]
            if index in eval_metric.keys():
                if self.best_eval_index is None:
                    self.best_eval_index = eval_metric[index]
                elif eval_metric[index] > self.best_eval_index:
                    self.best_eval_index = eval_metric[index]
                else:
                    self.wait += 1
        if self.wait > self.args.earily_stop_patience:
            self.end = True

    def info_manager(self, step: int, timer: Timer, log_loss: bool = False) -> None:
        """
        Manage and log training information.

        Args:
            step (int): Current training step.
            timer (Timer): Timer instance to track time.
            log_loss (bool, optional): Flag to determine the logging level for loss. Defaults to False.
        """
        loss_level = logging.INFO if log_loss else logging.DEBUG

        if self.args.global_rank == 0:
            # Log average loss and time at specified intervals.
            if step % self.args.gradient_accumulation_steps == 0:
                self.global_step += 1
                if self.global_step % self.args.show_avg_loss_step == 0:
                    timer.average_time(entry='end')
                    avg_time = timer.loop_time / self.args.show_avg_loss_step
                    avg_loss = self.all_loss / (self.args.show_avg_loss_step * self.args.gradient_accumulation_steps)
                    remaining_time = timer.calculate_remaining_time()
                    print_str = (f"--->global_step={self.global_step}, micro_step={step}, "
                                f"avg_loss={avg_loss if avg_loss >= 1e-4 else f'{avg_loss:.4e}'}, "
                                f"lr={self.lr:.4e}, "
                                f"avg_time={avg_time:.2f}s, remaining_time={remaining_time}, "
                                f"remaining_steps={self.args.num_global_update_steps - self.global_step}")
                    # 🌟防止梯度爆炸
                    if self.writer is not None:
                        self.writer.add_scalar('loss', avg_loss, self.global_step)
                        self.writer.add_scalar('lr', self.lr, self.global_step)
                        if self.grad_norm is not None:
                            self.writer.add_scalar('grad_norm', self.grad_norm, self.global_step)
                        self.writer.add_scalar('avg_time', avg_time, self.global_step)
                    if self.args.wandb and not self.args.test_code:
                        swanlab.log({'loss': avg_loss,
                                   'grad_norm': self.grad_norm,
                                   'lr': self.lr,
                                   'avg_time': avg_time}, 
                                   self.global_step)

                    if self.get_task_print:
                        print_str += self.get_task_print(self.all_metric, self.args)
                    
                    print_rank_0(print_str, self.args.global_rank, loss_level)
                    self.all_loss = 0.0
                    self.all_metric = []

            # Log evaluation loss at specified intervals.
            if step % self.args.eval_interval == 0 and self.eval_loss is not None and not self.args.skip_eval:
                print_str = f"--->micro_step={step}, eval_loss={self.eval_loss:.4f}"
                if self.get_task_print:
                    print_str += self.get_task_print(self.eval_metric, self.args)
                print_rank_0(print_str, self.args.global_rank, loss_level)
                if self.args.global_rank == 0:
                    if self.writer is not None:
                        self.writer.add_scalar('eval_loss', self.eval_loss, self.global_step)
                    if self.args.wandb  and not self.args.test_code:
                        swanlab.log({'eval_loss': self.eval_loss}, self.global_step)
                self.eval_loss = 0.0
                self.eval_metric = []

    def register_task_print(self, print_func):
        self.task_print = print_func

    @property
    def get_task_print(self):
        return getattr(self, "task_print", None)

    def save_model(self, model, optimizer, lr_scheduler, dataloader, step: int) -> None:
        """
        Save model, optimizer, and scheduler state.

        Args:
            model (torch.nn.Module): The model to be saved.
            optimizer (torch.optim.Optimizer): The optimizer to be saved.
            lr_scheduler (torch.optim.lr_scheduler._LRScheduler): The learning rate scheduler to be saved.
            step (int): The current training step.
        """
        config_path = os.path.join(self.save_folder, 'config.json')
        

        # Save the training configuration if required
        if self.save_config and isinstance(self.args, Namespace) and self.args.global_rank == 0:
            with open(config_path, 'w', encoding='utf-8') as f:
                print_rank_0(f'--->Saving training config at step {step+1} in {config_path}.')
                save_dict = {k: v for k, v in self.args.__dict__.items() if k != 'device'}
                json.dump(save_dict, f)
                self.save_config = False

        # Handle saving for pipeline parallel training 🌟修改
        if False:
        # if self.args.num_pp_stages is not None:
            should_save = (not self.end and (step + 1) % self.args.save_interval == 0) or (self.end and (step + 1) % self.args.save_interval != 0)
            if should_save:
                tag = f'step_{step+1}' if not self.end else 'final'
                print_rank_0(f'--->Start saving model at {step+1}th step in {self.save_folder}.')
                model.save_checkpoint(self.save_folder, tag=tag)
                print_rank_0('--->Saved the model.')
        else:
            # Handle saving for other cases
            if not self.end and (step + 1) % self.args.save_interval == 0:
                save_path = os.path.join(self.save_folder, f'step_{step+1}.ckpt')
                print_rank_0(f'--->Start saving model at step {step+1} in {save_path}.')
                self.torch_save(model, optimizer, lr_scheduler, dataloader, save_path)
                print_rank_0('--->Saved the model.')
            elif self.end:
                save_path = os.path.join(self.save_folder, 'final.ckpt')
                print_rank_0(f'--->Start saving model at final step in {save_path}.')
                self.torch_save(model, optimizer, lr_scheduler, dataloader, save_path)
                print_rank_0('--->Saved the model.')

    def torch_save(self, 
                model:torch.nn.Module, 
                optimizer: Callable, 
                lr_scheduler: Callable, 
                dataloader, 
                save_path: str):
        is_zero3 = hasattr(model, 'module') and hasattr(model.module, 'zero_optimization_partition_weights')
        model_state_dict = {}

        if is_zero3:
            print_rank_0('--->Gathering full model weights from all GPUs for ZeRO-3...')
            for name, param in model.module.named_parameters():         
                with deepspeed.zero.GatheredParameters(param):
                    if self.requires_save(name, param):
                        # Avoid OOM when zero3 is utilized.
                        model_state_dict[name] = param.data.clone().detach().cpu()
        else:
            for name, param in model.module.named_parameters():
                if self.requires_save(name, param):
                    model_state_dict[name] = param.data

        if self.args.global_rank == 0:
            if optimizer and lr_scheduler and not self.args.relora_steps:
                ckpt_to_save = {'model_state_dict': model_state_dict,
                                'optimizer_state_dict': optimizer.state_dict(),
                                'lr_scheduler_state_dict': lr_scheduler.state_dict()}
            else:
                ckpt_to_save = model_state_dict
            torch.save(ckpt_to_save, save_path)

    def requires_save(self, param_name, param):
        if self.args.save_trainable:
            if param.requires_grad:
                return True
            if self.args.params_to_save:
                for save_name in self.args.params_to_save:
                    if save_name in param_name:
                        return True
            return False
        else:
            return True
    
if __name__ == '__main__':
    """A quick test for trainer"""
    import os
    import torchvision
    from torchvision import datasets, transforms
    from dataclasses import dataclass
    import traceback

    os.environ['NO_LOG_FILE'] = 'true'
    model = torchvision.models.resnet18()

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    fake_dataset = datasets.FakeData(size=100, num_classes=3, transform=transforms.ToTensor())
    test_dataset = datasets.FakeData(size=200, num_classes=3, transform=transforms.ToTensor())
    data_loader = iter(DataLoader(fake_dataset, batch_size=10, shuffle=True))
    test_data_loader = iter(DataLoader(test_dataset, batch_size=10, shuffle=True))

    # Define forward and backward step functions
    def forward_step(model, data_loader, args, step):
        inputs, labels = next(data_loader)
        outputs = model(inputs)
        loss = torch.nn.CrossEntropyLoss()(outputs, labels)
        metric = {"acc": accuracy(outputs, labels)}
        return loss, metric

    def eval_step(model, data_loader, args, step):
        inputs, labels = next(data_loader)
        outputs = model(inputs)
        loss = torch.nn.CrossEntropyLoss()(outputs, labels)
        return loss.item()


    def backward_step(_, optimizer, loss):
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    def accuracy(outputs, labels):
        _, predicted = torch.max(outputs, 1)
        correct = (predicted == labels).sum().item()
        total = labels.size(0)
        return correct / total
    
    def task_print(all_metric, args):
        acc_count = sum([sub_dict['acc'] for sub_dict in all_metric])
        return f' train_acc:{(acc_count/args.show_loss_step) * 100}%'

    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    @dataclass
    class ARGS:
        num_micro_update_steps = 10000
        show_loss_step = 10
        save_interval = 10000
        eval_interval = 100000
        output_path = '.'
        experiment_name = 'resnet'
        global_rank = 0
        local_rank = 0
        gradient_accumulation_steps = 1
        
    trainer = Trainer(args=ARGS, writer=None)
    trainer.register_task_print(task_print)
    try:
        trainer.train(model=model, 
                      train_data_loader=data_loader, 
                      eval_data_loader=test_data_loader,
                      optimizer=optimizer, 
                      forward_step=forward_step, 
                      eval_step=eval_step,
                      backward_step=backward_step)
    except:
        traceback_info = traceback.format_exc()
        print_rank_0(traceback_info, ARGS.global_rank ,level=logging.ERROR)

