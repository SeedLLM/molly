<div align="center">
<img width="400" height="240" alt="Image" src="https://github.com/user-attachments/assets/94b8192d-7f5b-49ed-b2fa-861c643b8b7a" />
</div>

## molly

molly is a Large Language Model composed of multiple encoders, capable of understanding multi-omics data (DNA, RNA, and protein).

molly 是一个集成了多个 encoder 的大语言模型，能够理解 DNA，RNA 和 protein 序列信息。

<img width="994" height="369" alt="Image" src="https://github.com/user-attachments/assets/65b17c06-0506-40a3-bd25-e59172630cff" />

## :star2: Feature
- **Base Model**: Enhanced [Qwen3](https://github.com/QwenLM/Qwen3) with [nucleotide-transformer](https://github.com/instadeepai/nucleotide-transformer) and [ESM-2](https://github.com/facebookresearch/esm) encoders
- **Optimization**: Support [Liger-Kernel](https://github.com/linkedin/Liger-Kernel) and [FlashAttention](https://github.com/Dao-AILab/flash-attention) for 100% training speedup, see [example script](./scripts/train/examples/run_train_1B_v3.sh)

## 🤗 Download trained model

<div>
  <tr>
      <td><a href="https://huggingface.co/tpoisonooo/MOLLM-1.7B">molly-1.7B</a></td>
      <td><a href="https://huggingface.co/tpoisonooo/MOLLM-1.7B">molly-4B</a></td>
      <td><a href="https://huggingface.co/tpoisonooo/MOLLM-1.7B">molly-8B</a></td>
  </tr>
</div>  

## :zap: How to inference

    ```bash
    ./scripts/infer/inference_untrain.sh
    ```

## :fire: How to train

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

4. Fix qwen3_8B + deepspeed training stuck

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

## :pushpin: LICENSE

This project follows [apache license](./LICENSE).
