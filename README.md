<div align="center">
<img width="400" height="240" alt="Image" src="https://github.com/user-attachments/assets/94b8192d-7f5b-49ed-b2fa-861c643b8b7a" />
</div>

## Molly

Molly is a Large Language Model composed of multiple encoders, capable of understanding multi-omics data (DNA, RNA, and protein).

Molly 是一个集成了多个 encoder 的大语言模型，能够理解 DNA，RNA 和 protein 序列信息。
<img width="994" height="369" alt="Image" src="https://github.com/user-attachments/assets/65b17c06-0506-40a3-bd25-e59172630cff" />
Omics-Specific Models（OSMs）指代各自组学赛道中性能领先的专用模型；Enc-Head 则是“组学 Encoder + 分类头”的简洁架构，将预训练编码器与任务相关分类头直接连接。

## :star2: Feature
- **Base Model**: Enhanced [Qwen3](https://github.com/QwenLM/Qwen3) with [nucleotide-transformer](https://github.com/instadeepai/nucleotide-transformer) and [ESM-2](https://github.com/facebookresearch/esm) encoders
- **Optimization**: Support [Liger-Kernel](https://github.com/linkedin/Liger-Kernel) and [FlashAttention](https://github.com/Dao-AILab/flash-attention) for 100% training speedup, see [example script](./scripts/train/examples/run_train_1B_z2_b1.sh)

## 🤗 Download trained model

<div>
  <tr>
      <td><a href="https://huggingface.co/tpoisonooo/MOLLM-1.7B">molly-1.7B</a></td>
      <td><a href="https://huggingface.co/tpoisonooo/MOLLM-4B">molly-4B</a></td>
      <td><a href="https://huggingface.co/tpoisonooo/MOLLM-8B">molly-8B</a></td>
  </tr>
</div>  

## :zap: How to inference

    ```bash
    ./scripts/infer/inference_nt_lora.sh
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
    
    ## model._tp_plan -> model_tp_plan
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

## Eval
* 需要使用训练好的模型对评测集进行推理
    * 脚本中的experiment_name, MODEL_DIR, CHECKPOINT需要修改为训练好模型的路径
    * --text-model-path, --dna-rna-model-path, --protein-model-path为官方预训练权重文件路径
    * --dataset-path为评测集路径
    * --json-file为结果输出路径

```
./scripts/infer/inference_nt_lora.sh
```
* 将推理数据转换为待测评的格式
    * 需要修改src_paths与dst_path，src_paths为推理结果的路径（注意需要是文件夹），dst_path是转换后的输出路径（注意是json文件）
```
python molly/data_tools/convert.py
````
* 使用测评脚本获得模型在各个任务上的性能
    * 此处--input_file_path为转换后的输出的json文件的路径
```
cd molly/eval
.eval.sh
```

## :pushpin: LICENSE

This project follows [apache license](./LICENSE).