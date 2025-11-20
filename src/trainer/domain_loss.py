import contextlib
import copy
import functools
import glob
import importlib.metadata
import inspect
import json
import math
import os
import random
import re
import shutil
import sys
import tempfile
import time
import warnings
from collections.abc import Iterator, Mapping
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Optional, Union, Dict, List


# Integrations must be imported before ML frameworks:
# ruff: isort: off
from transformers.integrations import (
    get_reporting_integration_callbacks,
)

# ruff: isort: on

import huggingface_hub.utils as hf_hub_utils
import numpy as np
import torch
import torch.distributed as dist
from huggingface_hub import ModelCard, create_repo, upload_folder
from packaging import version
from torch import nn
from torch.utils.data import DataLoader, Dataset, IterableDataset, RandomSampler, SequentialSampler

from transformers.configuration_utils import PretrainedConfig
from transformers.data.data_collator import DataCollator, DataCollatorWithPadding, default_data_collator
from transformers.debug_utils import DebugOption, DebugUnderflowOverflow
from transformers.feature_extraction_sequence_utils import SequenceFeatureExtractor
from transformers.feature_extraction_utils import FeatureExtractionMixin
from transformers.hyperparameter_search import ALL_HYPERPARAMETER_SEARCH_BACKENDS, default_hp_search_backend
from transformers.image_processing_utils import BaseImageProcessor
from transformers.integrations.deepspeed import deepspeed_init, deepspeed_load_checkpoint, is_deepspeed_available
from transformers.integrations.tpu import tpu_spmd_dataloader
from transformers.modelcard import TrainingSummary
from transformers.modeling_utils import PreTrainedModel, load_sharded_checkpoint, unwrap_model
from transformers.models.auto.modeling_auto import (
    MODEL_FOR_CAUSAL_LM_MAPPING_NAMES,
    MODEL_MAPPING_NAMES,
)
from transformers.optimization import Adafactor, get_scheduler
from transformers.processing_utils import ProcessorMixin
from transformers.pytorch_utils import (
    is_torch_greater_or_equal_than_2_3,
)
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.trainer_callback import (
    CallbackHandler,
    DefaultFlowCallback,
    ExportableState,
    PrinterCallback,
    ProgressCallback,
    TrainerCallback,
    TrainerControl,
    TrainerState,
)
from transformers.trainer_pt_utils import (
    DistributedTensorGatherer,
    EvalLoopContainer,
    IterableDatasetShard,
    LabelSmoother,
    LayerWiseDummyOptimizer,
    LengthGroupedSampler,
    SequentialDistributedSampler,
    distributed_broadcast_scalars,
    distributed_concat,
    find_batch_size,
    get_model_param_count,
    get_module_class_from_name,
    get_parameter_names,
    nested_concat,
    nested_detach,
    nested_numpify,
    nested_xla_mesh_reduce,
    reissue_pt_warnings,
    remove_dummy_checkpoint,
    set_rng_state_for_device,
)
from transformers.trainer_utils import (
    PREFIX_CHECKPOINT_DIR,
    BestRun,
    EvalLoopOutput,
    EvalPrediction,
    HPSearchBackend,
    HubStrategy,
    PredictionOutput,
    RemoveColumnsCollator,
    SaveStrategy,
    TrainerMemoryTracker,
    TrainOutput,
    check_target_module_exists,
    default_compute_objective,
    denumpify_detensorize,
    enable_full_determinism,
    find_executable_batch_size,
    get_last_checkpoint,
    has_length,
    neftune_post_forward_hook,
    number_of_arguments,
    seed_worker,
    set_seed,
    speed_metrics,
)
from transformers.training_args import OptimizerNames, ParallelMode, TrainingArguments
from transformers.utils import (
    ADAPTER_CONFIG_NAME,
    ADAPTER_SAFE_WEIGHTS_NAME,
    ADAPTER_WEIGHTS_NAME,
    CONFIG_NAME,
    SAFE_WEIGHTS_INDEX_NAME,
    SAFE_WEIGHTS_NAME,
    WEIGHTS_INDEX_NAME,
    WEIGHTS_NAME,
    XLA_FSDPV2_MIN_VERSION,
    PushInProgress,
    PushToHubMixin,
    can_return_loss,
    check_torch_load_is_safe,
    find_labels,
    is_accelerate_available,
    is_apollo_torch_available,
    is_bitsandbytes_available,
    is_datasets_available,
    is_galore_torch_available,
    is_grokadamw_available,
    is_in_notebook,
    is_liger_kernel_available,
    is_lomo_available,
    is_peft_available,
    is_safetensors_available,
    is_sagemaker_dp_enabled,
    is_sagemaker_mp_enabled,
    is_schedulefree_available,
    is_torch_hpu_available,
    is_torch_mlu_available,
    is_torch_mps_available,
    is_torch_musa_available,
    is_torch_neuroncore_available,
    is_torch_npu_available,
    is_torch_xla_available,
    is_torch_xpu_available,
    is_torchao_available,
    logging,
    strtobool,
)
from transformers.utils.deprecation import deprecate_kwarg
from transformers.utils.import_utils import requires
from transformers.utils.quantization_config import QuantizationMethod


DEFAULT_CALLBACKS = [DefaultFlowCallback]
DEFAULT_PROGRESS_CALLBACK = ProgressCallback

if is_in_notebook():
    from .utils.notebook import NotebookProgressCallback

    DEFAULT_PROGRESS_CALLBACK = NotebookProgressCallback

if is_datasets_available():
    import datasets

if is_torch_xla_available():
    import torch_xla.core.xla_model as xm
    import torch_xla.debug.metrics as met
    import torch_xla.runtime as xr
    from torch_xla import __version__ as XLA_VERSION

    IS_XLA_FSDPV2_POST_2_2 = version.parse(XLA_VERSION) >= version.parse(XLA_FSDPV2_MIN_VERSION)
    if IS_XLA_FSDPV2_POST_2_2:
        import torch_xla.distributed.spmd as xs
else:
    IS_XLA_FSDPV2_POST_2_2 = False


if is_sagemaker_mp_enabled():
    import smdistributed.modelparallel.torch as smp
    from smdistributed.modelparallel import __version__ as SMP_VERSION

    IS_SAGEMAKER_MP_POST_1_10 = version.parse(SMP_VERSION) >= version.parse("1.10")

    from .trainer_pt_utils import smp_forward_backward, smp_forward_only, smp_gather, smp_nested_concat
else:
    IS_SAGEMAKER_MP_POST_1_10 = False


if is_safetensors_available():
    import safetensors.torch

if is_peft_available():
    from peft import PeftModel


if is_accelerate_available():
    from accelerate import Accelerator, skip_first_batches
    from accelerate import __version__ as accelerate_version
    from accelerate.state import AcceleratorState
    from accelerate.utils import (
        AutocastKwargs,
        DistributedDataParallelKwargs,
        DistributedType,
        TorchTensorParallelPlugin,
        load_fsdp_model,
        load_fsdp_optimizer,
        save_fsdp_model,
        save_fsdp_optimizer,
    )

    DATA_SAMPLERS = [RandomSampler]
    if version.parse(accelerate_version) > version.parse("1.3.0"):
        from accelerate.utils import TorchTensorParallelPlugin
    if version.parse(accelerate_version) > version.parse("0.23.0"):
        from accelerate.data_loader import SeedableRandomSampler

        DATA_SAMPLERS += [SeedableRandomSampler]

    if is_deepspeed_available():
        from accelerate.utils import DeepSpeedSchedulerWrapper

if is_accelerate_available("0.28.0"):
    from accelerate.utils import DataLoaderConfiguration


def _is_peft_model(model):
    if is_peft_available():
        classes_to_check = (PeftModel,)
        # Here we also check if the model is an instance of `PeftMixedModel` introduced in peft>=0.7.0: https://github.com/huggingface/transformers/pull/28321
        if version.parse(importlib.metadata.version("peft")) >= version.parse("0.7.0"):
            from peft import PeftMixedModel

            classes_to_check = (*classes_to_check, PeftMixedModel)
        return isinstance(model, classes_to_check)
    return False


def _get_fsdp_ckpt_kwargs():
    # TODO: @AjayP13, @younesbelkada replace this check with version check at the next `accelerate` release
    if is_accelerate_available() and "adapter_only" in list(inspect.signature(save_fsdp_model).parameters):
        return {"adapter_only": True}
    else:
        return {}


def safe_globals():
    # Starting from version 2.4 PyTorch introduces a check for the objects loaded
    # with torch.load(weights_only=True). Starting from 2.6 weights_only=True becomes
    # a default and requires allowlisting of objects being loaded.
    # See: https://github.com/pytorch/pytorch/pull/137602
    # See: https://pytorch.org/docs/stable/notes/serialization.html#torch.serialization.add_safe_globals
    # See: https://github.com/huggingface/accelerate/pull/3036
    if version.parse(torch.__version__).release < version.parse("2.6").release:
        return contextlib.nullcontext()

    np_core = np._core if version.parse(np.__version__) >= version.parse("2.0.0") else np.core
    allowlist = [np_core.multiarray._reconstruct, np.ndarray, np.dtype]
    # numpy >1.25 defines numpy.dtypes.UInt32DType, but below works for
    # all versions of numpy
    allowlist += [type(np.dtype(np.uint32))]

    return torch.serialization.safe_globals(allowlist)


if TYPE_CHECKING:
    import optuna

    if is_datasets_available():
        import datasets

logger = logging.get_logger(__name__)


# Name of the files used for checkpointing
TRAINING_ARGS_NAME = "training_args.bin"
TRAINER_STATE_NAME = "trainer_state.json"
OPTIMIZER_NAME = "optimizer.pt"
SCALER_NAME = "scaler.pt"
OPTIMIZER_NAME_BIN = "optimizer.bin"
SCHEDULER_NAME = "scheduler.pt"
FSDP_MODEL_NAME = "pytorch_model_fsdp"

from liger_kernel.transformers.model.loss_utils import LigerForCausalLMLoss

from dataclasses import dataclass

from transformers.cache_utils import Cache, EncoderDecoderCache
from transformers.utils import ModelOutput
from transformers.trainer import _is_peft_model

@dataclass
class CausalLMOutputWithPast(ModelOutput):
    """
    Base class for causal language model (or autoregressive) outputs.

    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            Language modeling loss (for next-token prediction).
        logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        past_key_values (`Cache`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            It is a [`~cache_utils.Cache`] instance. For more details, see our [kv cache guide](https://huggingface.co/docs/transformers/en/kv_cache).

            Contains pre-computed hidden-states (key and values in the self-attention blocks) that can be used (see
            `past_key_values` input) to speed up sequential decoding.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """
    loss: Optional[torch.FloatTensor] = None
    domain_losses: Optional[List[torch.FloatTensor]] = None
    logits: Optional[torch.FloatTensor] = None
    past_key_values: Optional[Cache] = None
    hidden_states: Optional[tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[tuple[torch.FloatTensor, ...]] = None


def my_inner_training_loop(
        self, batch_size=None, args=None, resume_from_checkpoint=None, trial=None, ignore_keys_for_eval=None
    ):
    self.accelerator.free_memory()
    self._train_batch_size = batch_size
    if self.args.auto_find_batch_size:
        if self.state.train_batch_size != self._train_batch_size:
            from accelerate.utils import release_memory

            (self.model_wrapped,) = release_memory(self.model_wrapped)
            self.model_wrapped = self.model

            # Check for DeepSpeed *after* the initial pass and modify the config
            if self.is_deepspeed_enabled:
                # Temporarily unset `self.args.train_batch_size`
                original_bs = self.args.per_device_train_batch_size
                self.args.per_device_train_batch_size = self._train_batch_size // max(1, self.args.n_gpu)
                self.propagate_args_to_deepspeed(True)
                self.args.per_device_train_batch_size = original_bs
        self.state.train_batch_size = self._train_batch_size
    logger.debug(f"Currently training with a batch size of: {self._train_batch_size}")
    # Data loader and number of training steps
    train_dataloader = self.get_train_dataloader()
    if self.is_fsdp_xla_v2_enabled:
        train_dataloader = tpu_spmd_dataloader(train_dataloader)

    # Setting up training control variables:
    # number of training epochs: num_train_epochs
    # number of training steps per epoch: num_update_steps_per_epoch
    # total number of training steps to execute: max_steps
    total_train_batch_size = self.get_total_train_batch_size(args)

    (
        num_train_epochs,
        num_update_steps_per_epoch,
        num_examples,
        num_train_samples,
        epoch_based,
        len_dataloader,
        max_steps,
    ) = self.set_initial_training_values(args, train_dataloader, total_train_batch_size)

    num_train_tokens = None
    if self.args.include_tokens_per_second:
        num_train_tokens = self.num_tokens(train_dataloader, None if epoch_based else max_steps)
        # If going by epochs, multiply tokens linearly
        if len_dataloader is not None and epoch_based:
            num_train_tokens *= args.num_train_epochs
        # Otherwise since its steps, we just multiply by grad accum
        else:
            num_train_tokens *= args.gradient_accumulation_steps

    if DebugOption.UNDERFLOW_OVERFLOW in self.args.debug:
        if self.args.n_gpu > 1:
            # nn.DataParallel(model) replicates the model, creating new variables and module
            # references registered here no longer work on other gpus, breaking the module
            raise ValueError(
                "Currently --debug underflow_overflow is not supported under DP. Please use DDP"
                " (torchrun or torch.distributed.launch (deprecated))."
            )
        else:
            debug_overflow = DebugUnderflowOverflow(self.model)  # noqa

    delay_optimizer_creation = (
        is_sagemaker_mp_enabled() or self.is_fsdp_xla_enabled or self.is_fsdp_enabled or self.is_tp_enabled
    )

    # Can't delay optimizer creation when using FSDP2: https://github.com/huggingface/accelerate/blob/3f636d626063ffcf9a337c7d3624d61b7d187d59/src/accelerate/accelerator.py#L1404
    is_fsdp2 = self.is_fsdp_enabled and (getattr(self.accelerator.state.fsdp_plugin, "fsdp_version", 1) == 2)
    if is_fsdp2:
        delay_optimizer_creation = False

    # We need to reset the scheduler, as its parameters may be different on subsequent calls
    if self._created_lr_scheduler:
        self.lr_scheduler = None
        self._created_lr_scheduler = False

    if self.is_deepspeed_enabled:
        self.optimizer, self.lr_scheduler = deepspeed_init(self, num_training_steps=max_steps)

    if not delay_optimizer_creation:
        self.create_optimizer_and_scheduler(num_training_steps=max_steps)

    self.state = TrainerState(
        stateful_callbacks=[
            cb for cb in self.callback_handler.callbacks + [self.control] if isinstance(cb, ExportableState)
        ]
    )
    self.state.is_hyper_param_search = trial is not None
    self.state.train_batch_size = self._train_batch_size

    # Compute absolute values for logging, eval, and save if given as ratio
    self.state.compute_steps(args, max_steps)

    # Activate gradient checkpointing if needed
    if args.gradient_checkpointing:
        self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs=args.gradient_checkpointing_kwargs)

    model = self._wrap_model(self.model_wrapped)

    # as the model is wrapped, don't use `accelerator.prepare`
    # this is for unhandled cases such as
    # FSDP-XLA, SageMaker MP/DP, DataParallel, IPEX
    use_accelerator_prepare = True if model is self.model else False

    if use_accelerator_prepare and self.is_fsdp_enabled:
        # In case of auto_find_batch_size=True
        # Remove FSDP wrapping from sub-models.
        self.model = unwrap_model(self.model, recursive=True)

    if delay_optimizer_creation:
        if use_accelerator_prepare:
            # configure fsdp plugin for qlora if any
            self._fsdp_qlora_plugin_updates()
            if self.accelerator.mixed_precision != "fp8":
                self.model = self.accelerator.prepare(self.model)
        self.create_optimizer_and_scheduler(num_training_steps=max_steps)

    # prepare using `accelerator` prepare
    if use_accelerator_prepare:
        self.model.train()
        if hasattr(self.lr_scheduler, "step"):
            if self.use_apex:
                model = self.accelerator.prepare(self.model)
            else:
                if delay_optimizer_creation:
                    self.optimizer = self.accelerator.prepare(self.optimizer)
                else:
                    model, self.optimizer = self.accelerator.prepare(self.model, self.optimizer)
        else:
            # to handle cases wherein we pass "DummyScheduler" such as when it is specified in DeepSpeed config.
            model, self.optimizer, self.lr_scheduler = self.accelerator.prepare(
                self.model, self.optimizer, self.lr_scheduler
            )
    elif self.args.optim in [OptimizerNames.LOMO, OptimizerNames.ADALOMO]:
        # In this case we are in DDP + LOMO, which should be supported
        self.optimizer = self.accelerator.prepare(self.optimizer)

    if self.is_fsdp_enabled:
        self.model = self.model_wrapped = model

    # for the rest of this function `model` is the outside model, whether it was wrapped or not
    if model is not self.model:
        self.model_wrapped = model

    # backward compatibility
    if self.is_deepspeed_enabled:
        self.deepspeed = self.model_wrapped

    # ckpt loading
    if resume_from_checkpoint is not None:
        if self.is_deepspeed_enabled:
            deepspeed_load_checkpoint(
                self.model_wrapped, resume_from_checkpoint, load_module_strict=not _is_peft_model(self.model)
            )
        elif is_sagemaker_mp_enabled() or self.is_fsdp_enabled:
            self._load_from_checkpoint(resume_from_checkpoint, self.model_wrapped)

    # Check if saved optimizer or scheduler states exist
    self._load_optimizer_and_scheduler(resume_from_checkpoint)
    self._load_scaler(resume_from_checkpoint)

    # important: at this point:
    # self.model         is the Transformers Model
    # self.model_wrapped is DDP(Transformers Model), Deepspeed(Transformers Model),
    # FSDP(Transformers Model), Dynamo Optimized Module(Transformers Model) etc.

    # Train!
    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {num_examples:,}")
    logger.info(f"  Num Epochs = {num_train_epochs:,}")
    logger.info(f"  Instantaneous batch size per device = {self.args.per_device_train_batch_size:,}")
    if self.args.per_device_train_batch_size != self._train_batch_size:
        logger.info(f"  Training with DataParallel so batch size has been adjusted to: {self._train_batch_size:,}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_train_batch_size:,}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {max_steps:,}")
    logger.info(f"  Number of trainable parameters = {get_model_param_count(model, trainable_only=True):,}")

    self.state.epoch = 0
    start_time = time.time()
    epochs_trained = 0
    steps_trained_in_current_epoch = 0
    steps_trained_progress_bar = None

    # Check if continuing training from a checkpoint
    if resume_from_checkpoint is not None and os.path.isfile(
        os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME)
    ):
        self.state = TrainerState.load_from_json(os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME))
        self.compare_trainer_and_checkpoint_args(self.args, self.state)
        self._load_callback_state()
        epochs_trained = int(self.state.global_step // num_update_steps_per_epoch)
        if not args.ignore_data_skip:
            steps_trained_in_current_epoch = self.state.global_step % (num_update_steps_per_epoch)
            steps_trained_in_current_epoch *= args.gradient_accumulation_steps
        else:
            steps_trained_in_current_epoch = 0

        logger.info("  Continuing training from checkpoint, will skip to saved global_step")
        logger.info(f"  Continuing training from epoch {epochs_trained}")
        logger.info(f"  Continuing training from global step {self.state.global_step}")
        if not args.ignore_data_skip:
            logger.info(
                f"  Will skip the first {epochs_trained} epochs then the first"
                f" {steps_trained_in_current_epoch} batches in the first epoch."
            )

    # Update the references
    for attr in ("model", "optimizer", "lr_scheduler"):
        setattr(self.callback_handler, attr, getattr(self, attr))
    self.callback_handler.train_dataloader = train_dataloader

    self.state.init_training_references(self, max_steps, num_train_epochs, trial)

    # tr_loss is a tensor to avoid synchronization of TPUs through .item()
    tr_loss = torch.tensor(0.0, device=args.device)
    # _total_loss_scalar is updated everytime .item() has to be called on tr_loss and stores the sum of all losses
    self._total_loss_scalar = 0.0
    self._globalstep_last_logged = self.state.global_step
    model.zero_grad()
    grad_norm: Optional[float] = None
    learning_rate = None
    self.control = self.callback_handler.on_train_begin(args, self.state, self.control)

    if args.eval_on_start:
        self._evaluate(trial, ignore_keys_for_eval, skip_scheduler=True)

    for epoch in range(epochs_trained, num_train_epochs):
        epoch_dataloader = train_dataloader
        if hasattr(epoch_dataloader, "set_epoch"):
            epoch_dataloader.set_epoch(epoch)

        # Reset the past mems state at the beginning of each epoch if necessary.
        if args.past_index >= 0:
            self._past = None

        steps_in_epoch = (
            len(epoch_dataloader)
            if len_dataloader is not None
            else args.max_steps * args.gradient_accumulation_steps
        )
        self.control = self.callback_handler.on_epoch_begin(args, self.state, self.control)

        if epoch == epochs_trained and resume_from_checkpoint is not None and steps_trained_in_current_epoch == 0:
            self._load_rng_state(resume_from_checkpoint)

        rng_to_sync = False
        steps_skipped = 0
        if steps_trained_in_current_epoch > 0:
            epoch_dataloader = skip_first_batches(epoch_dataloader, steps_trained_in_current_epoch)
            steps_skipped = steps_trained_in_current_epoch
            steps_trained_in_current_epoch = 0
            rng_to_sync = True

        step = -1
        epoch_iterator = iter(epoch_dataloader)
        # We chunkify the epoch iterator into gradient accumulation steps `n` batches
        remainder = steps_in_epoch % args.gradient_accumulation_steps
        if remainder == 0:
            remainder = args.gradient_accumulation_steps
        update_step = -1
        total_updates = steps_in_epoch // args.gradient_accumulation_steps + int(
            remainder < args.gradient_accumulation_steps
        )
        for _ in range(total_updates):
            update_step += 1
            num_batches = args.gradient_accumulation_steps if update_step != (total_updates - 1) else remainder
            batch_samples, num_items_in_batch = self.get_batch_samples(epoch_iterator, num_batches, args.device)
            for i, inputs in enumerate(batch_samples):
                step += 1
                do_sync_step = (step + 1) % args.gradient_accumulation_steps == 0 or (step + 1) == steps_in_epoch
                # Since we perform prefetching, we need to manually set sync_gradients
                self.accelerator.gradient_state._set_sync_gradients(do_sync_step)

                if self.args.include_num_input_tokens_seen:
                    main_input_name = getattr(self.model, "main_input_name", "input_ids")
                    if main_input_name not in inputs:
                        logger.warning(
                            "Tried to track the number of tokens seen, however the current model is "
                            "not configured properly to know what item is the input. To fix this, add "
                            "a `main_input_name` attribute to the model class you are using."
                        )
                    else:
                        input_tokens = inputs[main_input_name].numel()
                        input_tokens = torch.tensor(input_tokens, device=self.args.device, dtype=torch.int64)
                        self.state.num_input_tokens_seen += self.accelerator.gather(input_tokens).sum().item()
                if rng_to_sync:
                    self._load_rng_state(resume_from_checkpoint)
                    rng_to_sync = False

                # Skip past any already trained steps if resuming training
                if steps_trained_in_current_epoch > 0:
                    steps_trained_in_current_epoch -= 1
                    if steps_trained_progress_bar is not None:
                        steps_trained_progress_bar.update(1)
                    if steps_trained_in_current_epoch == 0:
                        self._load_rng_state(resume_from_checkpoint)
                    continue
                elif steps_trained_progress_bar is not None:
                    steps_trained_progress_bar.close()
                    steps_trained_progress_bar = None

                if step % args.gradient_accumulation_steps == 0:
                    self.control = self.callback_handler.on_step_begin(args, self.state, self.control)

                # We explicitly want to avoid relying on `accelerator.accumulate` for generation training
                context = (
                    functools.partial(self.accelerator.no_sync, model=model)
                    if i != len(batch_samples) - 1
                    and self.accelerator.distributed_type != DistributedType.DEEPSPEED
                    else contextlib.nullcontext
                )
                with context():
                    # tr_loss_step = self.training_step(model, inputs, num_items_in_batch)
                    tr_loss_step, domain_loss_dict = self.training_step(model, inputs, num_items_in_batch)
                    # tr_loss_step, domain_loss_list = self.training_step(model, inputs, num_items_in_batch)

                if (
                    args.logging_nan_inf_filter
                    and not is_torch_xla_available()
                    and (torch.isnan(tr_loss_step) or torch.isinf(tr_loss_step))
                ):
                    # if loss is nan or inf simply add the average of previous logged losses
                    tr_loss = tr_loss + tr_loss / (1 + self.state.global_step - self._globalstep_last_logged)
                else:
                    if tr_loss.device != tr_loss_step.device:
                        raise ValueError(
                            f"Calculated loss must be on the original device: {tr_loss.device} but device in use is {tr_loss_step.device}"
                        )
                    tr_loss = tr_loss + tr_loss_step

                self.current_flos += float(self.floating_point_ops(inputs))

                if do_sync_step:
                    # Since we perform prefetching, we need to manually set sync_gradients to True
                    self.accelerator.gradient_state._set_sync_gradients(True)

                    # Gradient clipping
                    if args.max_grad_norm is not None and args.max_grad_norm > 0:
                        if is_sagemaker_mp_enabled() and args.fp16:
                            _grad_norm = self.optimizer.clip_master_grads(args.max_grad_norm)
                        elif self.use_apex:
                            from apex import amp

                            # Revert to normal clipping otherwise, handling Apex or full precision
                            _grad_norm = nn.utils.clip_grad_norm_(
                                amp.master_params(self.optimizer),
                                args.max_grad_norm,
                            )
                        else:
                            grad_norm_context = contextlib.nullcontext
                            if self.is_tp_enabled:
                                from torch.distributed._tensor.experimental import implicit_replication

                                grad_norm_context = implicit_replication
                            with grad_norm_context():
                                _grad_norm = self.accelerator.clip_grad_norm_(
                                    model.parameters(),
                                    args.max_grad_norm,
                                )

                        if (
                            is_accelerate_available()
                            and self.accelerator.distributed_type == DistributedType.DEEPSPEED
                        ):
                            grad_norm = model.get_global_grad_norm()
                            # In some cases the grad norm may not return a float
                            if hasattr(grad_norm, "item"):
                                grad_norm = grad_norm.item()
                        else:
                            grad_norm = _grad_norm

                    self.control = self.callback_handler.on_pre_optimizer_step(args, self.state, self.control)

                    self.optimizer.step()

                    self.control = self.callback_handler.on_optimizer_step(args, self.state, self.control)

                    # get leaning rate before update
                    learning_rate = self._get_learning_rate()

                    if not self.accelerator.optimizer_step_was_skipped:
                        # Delay optimizer scheduling until metrics are generated
                        if not isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                            self.lr_scheduler.step()

                    model.zero_grad()
                    self.state.global_step += 1
                    self.state.epoch = epoch + (step + 1 + steps_skipped) / steps_in_epoch
                    self.control = self.callback_handler.on_step_end(args, self.state, self.control)
                    self._maybe_log_save_evaluate(
                        tr_loss,
                        grad_norm,
                        model,
                        trial,
                        epoch,
                        ignore_keys_for_eval,
                        start_time,
                        learning_rate=learning_rate, 
                        extra=domain_loss_dict,
                        # extra=domain_loss_list,
                    )
                else:
                    self.control = self.callback_handler.on_substep_end(args, self.state, self.control)

                # PyTorch/XLA relies on the data loader to insert the mark_step for
                # each step. Since we are breaking the loop early, we need to manually
                # insert the mark_step here.
                if self.control.should_epoch_stop or self.control.should_training_stop:
                    if is_torch_xla_available():
                        xm.mark_step()
                    break
            # We also need to break out of the nested loop
            if self.control.should_epoch_stop or self.control.should_training_stop:
                if is_torch_xla_available():
                    xm.mark_step()
                break
        if step < 0:
            logger.warning(
                "There seems not to be a single sample in your epoch_iterator, stopping training at step"
                f" {self.state.global_step}! This is expected if you're using an IterableDataset and set"
                f" num_steps ({max_steps}) higher than the number of available samples."
            )
            self.control.should_training_stop = True

        self.control = self.callback_handler.on_epoch_end(args, self.state, self.control)
        # self._maybe_log_save_evaluate(
        #     tr_loss, grad_norm, model, trial, epoch, ignore_keys_for_eval, start_time, learning_rate=learning_rate
        # )
        self._maybe_log_save_evaluate(
            tr_loss, grad_norm, model, trial, epoch, ignore_keys_for_eval, start_time, learning_rate=learning_rate, extra=domain_loss_dict
            # tr_loss, grad_norm, model, trial, epoch, ignore_keys_for_eval, start_time, learning_rate=learning_rate, extra=domain_loss_list
        )

        if DebugOption.TPU_METRICS_DEBUG in self.args.debug:
            if is_torch_xla_available():
                # tpu-comment: Logging debug metrics for PyTorch/XLA (compile, execute times, ops, etc.)
                xm.master_print(met.metrics_report())
            else:
                logger.warning(
                    "You enabled PyTorch/XLA debug metrics but you don't have a TPU "
                    "configured. Check your training configuration if this is unexpected."
                )
        if self.control.should_training_stop:
            break

    if args.past_index and hasattr(self, "_past"):
        # Clean the state at the end of training
        delattr(self, "_past")

    logger.info("\n\nTraining completed. Do not forget to share your model on huggingface.co/models =)\n\n")
    if args.load_best_model_at_end and self.state.best_model_checkpoint is not None:
        # Wait for everyone to get here so we are sure the model has been saved by process 0.
        if is_torch_xla_available():
            xm.rendezvous("load_best_model_at_end")
        elif args.parallel_mode == ParallelMode.DISTRIBUTED:
            dist.barrier()
        elif is_sagemaker_mp_enabled():
            smp.barrier()

        self._load_best_model()

    # add remaining tr_loss
    self._total_loss_scalar += tr_loss.item()
    effective_global_step = max(self.state.global_step, 0.001)  # Avoid ZeroDivisionError
    train_loss = self._total_loss_scalar / effective_global_step

    metrics = speed_metrics(
        "train",
        start_time,
        num_samples=num_train_samples,
        num_steps=self.state.max_steps,
        num_tokens=num_train_tokens,
    )
    self.store_flos()
    metrics["total_flos"] = self.state.total_flos
    metrics["train_loss"] = train_loss

    self.is_in_train = False

    self._memory_tracker.stop_and_update_metrics(metrics)

    self.log(metrics)

    run_dir = self._get_output_dir(trial)
    checkpoints_sorted = self._sorted_checkpoints(use_mtime=False, output_dir=run_dir)

    # Delete the last checkpoint when save_total_limit=1 if it's different from the best checkpoint and process allowed to save.
    if self.args.should_save and self.state.best_model_checkpoint is not None and self.args.save_total_limit == 1:
        for checkpoint in checkpoints_sorted:
            if not os.path.samefile(checkpoint, self.state.best_model_checkpoint):
                logger.info(f"Deleting older checkpoint [{checkpoint}] due to args.save_total_limit")
                shutil.rmtree(checkpoint, ignore_errors=True)

    self.control = self.callback_handler.on_train_end(args, self.state, self.control)

    # Wait for the checkpoint to be uploaded.
    self._finish_current_push()

    # After training we make sure to retrieve back the original forward pass method
    # for the embedding layer by removing the forward post hook.
    if self.neftune_noise_alpha is not None:
        self._deactivate_neftune(self.model)

    return TrainOutput(self.state.global_step, train_loss, metrics)

def my_maybe_log_save_evaluate(
          self, tr_loss, grad_norm, model, trial, epoch, ignore_keys_for_eval, start_time, learning_rate=None, extra: Dict={}
        #   self, tr_loss, grad_norm, model, trial, epoch, ignore_keys_for_eval, start_time, learning_rate=None, extra: List=[]
    ):
    if self.control.should_log and self.state.global_step > self._globalstep_last_logged:
        if is_torch_xla_available():
            xm.mark_step()

        # logs: dict[str, float] = {}
        logs: Dict[str, float] = extra

        # all_gather + mean() to get average loss over all processes
        tr_loss_scalar = self._nested_gather(tr_loss).mean().item()

        # reset tr_loss to zero
        tr_loss -= tr_loss

        logs["loss"] = round(tr_loss_scalar / (self.state.global_step - self._globalstep_last_logged), 4)
        if grad_norm is not None:
            logs["grad_norm"] = grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm
        if learning_rate is not None:
            logs["learning_rate"] = learning_rate
        else:
            logs["learning_rate"] = self._get_learning_rate()

        self._total_loss_scalar += tr_loss_scalar
        self._globalstep_last_logged = self.state.global_step
        self.store_flos()

        self.log(logs, start_time)

        # import pdb
        # pdb.set_trace()
        # for item in extra:
        #     logs: dict[str, float] = item
        #     self.log(logs, start_time)

def my_training_step(
        self,
        model: nn.Module,
        inputs: dict[str, Union[torch.Tensor, Any]],
        num_items_in_batch: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Perform a training step on a batch of inputs.

        Subclass and override to inject custom behavior.

        Args:
            model (`nn.Module`):
                The model to train.
            inputs (`dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument `labels`. Check your model's documentation for all accepted arguments.

        Return:
            `torch.Tensor`: The tensor with training loss on this batch.
        """
        model.train()
        if hasattr(self.optimizer, "train") and callable(self.optimizer.train):
            self.optimizer.train()

        inputs = self._prepare_inputs(inputs)
        if is_sagemaker_mp_enabled():
            loss_mb = smp_forward_backward(model, inputs, self.args.gradient_accumulation_steps)
            return loss_mb.reduce_mean().detach().to(self.args.device)

        with self.compute_loss_context_manager():
            # loss = self.compute_loss(model, inputs, num_items_in_batch=num_items_in_batch)
            loss, outputs = self.compute_loss(model, inputs, num_items_in_batch=num_items_in_batch, return_outputs=True)
        
        task_num = inputs.get('task_num')
        def build_domain_loss_dict(domain_losses, source):
            values = [value.item() for value in outputs.domain_losses]
            ret = {}
            log_all = []
            for s, v in zip(source, values):
                log = {}
                source_id = s.item()
                keymap = {
                    0: "antibody_antigen",
                    1: "cpd-prom_core",
                    2: "CRISPROnTarget",
                    3: "emp-H",
                    4: "enhancer_activity",
                    5: "Fluorescence-Fluorescence",
                    6: "FunctionEC-FunctionEC",
                    7: "Isoform-Isoform",
                    8: "MeanRibosomeLoading-MeanRibosomeLoading",
                    9: "Modification-Modification",
                    10: "NoncodingRNAFamily-NoncodingRNAFamily",
                    11: "pd-prom_300",
                    12: "ProgrammableRNASwitches-ProgrammableRNASwitches",
                    13: "promoter_enhancer_interaction",
                    14: "rna_protein_interaction",
                    15: "Solubility-Solubility",
                    16: "Stability-Stability",
                    17: "Thermostability-Thermostability",
                    18: "tf-h",
                    19: "tf-m",
                    100: "Other"
                }
                if source_id in keymap:
                    suffix = keymap[source_id]
                else:
                    suffix = 'bad'
                key = f'loss_{suffix}'
                # 记录所有key对应的loss
                log[key] = v
                log_all.append(log)
                # 相同key记录第一个loss 否则跳过
                if key in ret:
                    pass
                else:
                    ret[key] = v
            return ret, log_all
        domain_loss_dict,log_all = build_domain_loss_dict(outputs.domain_losses, inputs['task_label'])
        # domain_loss_list = build_domain_loss_dict(outputs.domain_losses, inputs['task_label'])
        # import pdb
        # pdb.set_trace()
        for log,num in zip(log_all, task_num):
            for key, value in log.items():
                # logger.info(f"Step {self.state.global_step}: task_num = {num}, domain = {key}, loss = {value}")
                print(f"Step {self.state.global_step}: task_num = {num}, domain = {key}, loss = {value}")

        del inputs
        del outputs.domain_losses
        outputs.domain_losses = None
        if (
            self.args.torch_empty_cache_steps is not None
            and self.state.global_step % self.args.torch_empty_cache_steps == 0
        ):
            if is_torch_xpu_available():
                torch.xpu.empty_cache()
            elif is_torch_mlu_available():
                torch.mlu.empty_cache()
            elif is_torch_musa_available():
                torch.musa.empty_cache()
            elif is_torch_npu_available():
                torch.npu.empty_cache()
            elif is_torch_mps_available():
                torch.mps.empty_cache()
            elif is_torch_hpu_available():
                logger.warning(
                    "`torch_empty_cache_steps` is set but HPU device/backend does not support empty_cache()."
                )
            else:
                torch.cuda.empty_cache()

        kwargs = {}

        # For LOMO optimizers you need to explicitly use the learnign rate
        if self.args.optim in [OptimizerNames.LOMO, OptimizerNames.ADALOMO]:
            kwargs["learning_rate"] = self._get_learning_rate()

        if self.args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training

        if self.use_apex:
            from apex import amp

            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            # Finally we need to normalize the loss for reporting if GA loss bug is not fixed during compute loss
            if (not self.model_accepts_loss_kwargs or num_items_in_batch is None) and self.compute_loss_func is None:
                # loss = loss / self.args.gradient_accumulation_steps
                domain_losses = [value / self.args.gradient_accumulation_steps for value in domain_losses]

            # Turning off loss scaling w.r.t. gradient accumulation when DeepSpeed is enabled
            # https://github.com/huggingface/transformers/pull/35808
            if self.accelerator.distributed_type == DistributedType.DEEPSPEED:
                kwargs["scale_wrt_gas"] = False

            self.accelerator.backward(loss, **kwargs)

            # return loss.detach()
            return loss.detach(), domain_loss_dict
            # return loss.detach(), domain_loss_list

def my_lce_forward(
    self,
    input_ids: Optional[torch.LongTensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[List[torch.FloatTensor]] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    labels: Optional[torch.LongTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    cache_position: Optional[torch.LongTensor] = None,
    logits_to_keep: Union[int, torch.Tensor] = 0,
    skip_logits: Optional[bool] = None,
    **kwargs,
) -> CausalLMOutputWithPast:
    r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        logits_to_keep (`int` or `torch.Tensor`, *optional*):
            If an `int`, compute logits for the last `logits_to_keep` tokens. If `0`, calculate logits for all
            `input_ids` (special case). Only last token logits are needed for generation, and calculating them only for that
            token can save memory, which becomes pretty significant for long sequences or large vocabulary size.
            If a `torch.Tensor`, must be 1D corresponding to the indices to keep in the sequence length dimension.
            This is useful when using packed tensor format (single dimension for batch and sequence length).

    Returns:

    Example:

    ```python
    >>> from transformers import AutoTokenizer, Qwen3ForCausalLM

    >>> model = Qwen3ForCausalLM.from_pretrained("Qwen/Qwen3-8B")
    >>> tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")

    >>> prompt = "Hey, are you conscious? Can you talk to me?"
    >>> inputs = tokenizer(prompt, return_tensors="pt")

    >>> # Generate
    >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
    >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
    ```"""
    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )

    # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
    outputs = self.model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        cache_position=cache_position,
        **kwargs,
    )

    hidden_states = outputs[0]
    # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
    slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
    kept_hidden_states = hidden_states[:, slice_indices, :]

    shift_labels = kwargs.pop("shift_labels", None)
    logits = None
    loss = None
    domain_losses = []

    if skip_logits and labels is None and shift_labels is None:
        raise ValueError("skip_logits is True, but labels and shift_labels are None")

    if skip_logits is None:
        # By default, if in training mode, don't materialize logits
        skip_logits = self.training and (labels is not None or shift_labels is not None)

    # LigerForCausalLMLoss 的底层算子 liger_fused_linear_cross_entropy 只接受 (hidden_states, lm_head_weight, labels, ...) 等固定位置参数,不接受return_dict=True
    kwargs.pop('return_dict', None)
    if skip_logits:
        loss = LigerForCausalLMLoss(
            hidden_states=kept_hidden_states,
            lm_head_weight=self.lm_head.weight,
            labels=labels,
            shift_labels=shift_labels,
            hidden_size=self.config.hidden_size,
            **kwargs,
        )
        temp_logits = self.lm_head(kept_hidden_states)
        batch_size = temp_logits.size(0)
        for i in range(batch_size):
            sample_logits = temp_logits[i].unsqueeze(0)
            sample_labels = labels[i].unsqueeze(0)
            sample_loss = self.loss_function(
                logits=sample_logits,
                labels=sample_labels,
                vocab_size=self.config.vocab_size,
                **kwargs
            )
            domain_losses.append(sample_loss.detach())
    else:
        logits = self.lm_head(kept_hidden_states)
        if labels is not None:
            loss = self.loss_function(
                logits=logits,
                labels=labels,
                vocab_size=self.config.vocab_size,
                **kwargs,
            )
            batch_size = logits.size(0)
            for i in range(batch_size):
                # 提取第 i 个样本的 logits 和 labels
                sample_logits = logits[i]
                sample_labels = labels[i]

                # 调用 ForCausalLMLoss 计算单个样本的损失
                sample_loss = self.loss_function(
                    logits=sample_logits.unsqueeze(0),  # 增加 batch 维度
                    labels=sample_labels.unsqueeze(0),  # 增加 batch 维度
                    vocab_size=self.config.vocab_size,
                    **kwargs
                )
                domain_losses.append(sample_loss.detach())

    return CausalLMOutputWithPast(
        loss=loss,
        domain_losses=domain_losses,
        logits=logits,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
    )
