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
