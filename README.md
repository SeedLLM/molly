# BioMLLM_V2
基于初始版本的多组学大语言模型的修改版本，目标是优化代码，去除冗余

## How to run

1. Hotfix transformers source code

```python
## transformers/modeling_utils.py
## add 4 lines 
    if not model._tp_plan:
        model_tp_plan = {}
    else:
        model_tp_plan = model._tp_plan

## old code
    tp_plan_regex = (
        re.compile("|".join([re.escape(plan) for plan in model_tp_plan]))
        if _torch_distributed_available and torch.distributed.is_initialized()
        else None
    )
```

2. Run training script
```bash
swanlab login

./scripts/train/run_train.sh

# or for test
./scripts/train/run_train_mini.sh
```

3. Debug qwen3_8B + deepspeed training stuck

Open `/usr/local/lib/python3.10/dist-packages/deepspeed/runtime/bf16_optimizer.py`

```python
294         if all_groups_norm <= 0.:
295             if dist.get_rank() == 0:
296                 dist.barrier()
297                 pdb.set_trace()
298             else:
299                 dist.barrier()
300
301         if self.clip_grad > 0.:
302             clip_tensors_by_global_norm(input_tensors=self.get_grads_for_norm(for_clipping=True),
303                                         max_norm=self.clip_grad,
304                                         global_norm=all_groups_norm,
305                                         mpu=self.mpu,
306                                         use_graph=self.graph_harvesting)
```


