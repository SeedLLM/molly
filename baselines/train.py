import torch
from dataclasses import dataclass, field
from transformers import TrainingArguments, HfArgumentParser, set_seed, AutoTokenizer, Trainer, AutoModelForMaskedLM
from dataset import ClassificationDataset, ClassificationCollator
from model import BackboneWithClsHead
from sklearn.metrics import accuracy_score, matthews_corrcoef
import numpy as np
import json

set_seed(42)


@dataclass
class BackboneTrainConfig(TrainingArguments):
    model_type: str = field(
        default="NT",
        metadata={"help": "Model type: NT, ESM, NT+ESM, NT+NT, ESM+ESM"}
    )
    task_name: str = field(
        default=None,
        metadata={"help": "Name of the classification task"}
    )
    dna_rna_model_path: str = field(
        default=None,
        metadata={"help": "Path to the DNA/RNA model (for NT and NT+ESM models)"}
    )
    dna_rna_k_tokens: int = field(
        default=1024,
        metadata={"help": "Max tokens for DNA/RNA sequences"}
    )
    protein_model_path: str = field(
        default=None,
        metadata={"help": "Path to the Protein model (for ESM and NT+ESM models)"}
    )
    protein_k_tokens: int = field(
        default=1024,
        metadata={"help": "Max tokens for Protein sequences"}
    )
    train_dataset_path: str = field(
        default=None,
        metadata={"help": "Path to the training dataset (parquet file)"}
    )
    eval_dataset_path: str = field(
        default=None,
        metadata={"help": "Path to the evaluation dataset (parquet file)"}
    )
    num_labels: int = field(
        default=2,
        metadata={"help": "Number of labels for classification"}
    )
    eval_metrics: str = field(
        default="acc",
        metadata={"help": "Metric for evaluation: acc or mcc"}
    )
    save_safetensors: bool = field(default=False, metadata={"help": "Disable safetensors to avoid shared memory issue"})
    label2id_path: str = field(
        default=None,
        metadata={"help": "Path to label2id mapping file (optional)"}
    )
    multi_label: bool = field(
        default=False,
        metadata={"help": "Whether it's a multi-label classification task"}
    )


def get_tokenizer(args: BackboneTrainConfig):
    if args.model_type in ["NT", "NT+ESM", "NT+NT"]:
        assert args.dna_rna_model_path is not None, "dna_rna_model_path is required for NT models"
        dna_rna_tokenizer = AutoTokenizer.from_pretrained(args.dna_rna_model_path, use_fast=True)
    else:
        dna_rna_tokenizer = None

    if args.model_type in ["ESM", "NT+ESM", "ESM+ESM"]:
        assert args.protein_model_path is not None, "protein_model_path is required for ESM models"
        protein_tokenizer = AutoTokenizer.from_pretrained(args.protein_model_path, use_fast=True)
    else:
        protein_tokenizer = None

    return dna_rna_tokenizer, protein_tokenizer


def get_model(args: BackboneTrainConfig):
    model = BackboneWithClsHead(
        model_type=args.model_type,
        nt_model=args.dna_rna_model_path,
        esm_model=args.protein_model_path,
        num_labels=args.num_labels,
        multi_label=args.multi_label
    )
    return model


def get_datasets(args: BackboneTrainConfig, dna_rna_tokenizer=None, protein_tokenizer=None):

    if args.label2id_path is not None:
        with open(args.label2id_path, 'r') as f:
            label_list = json.load(f)
        args.num_labels = len(label_list)
        print(f"Loaded label2id mapping with {args.num_labels} labels.")
        label2id = {label: idx for idx, label in enumerate(label_list)}
    else:    
        label2id = None


    train_dataset = ClassificationDataset(
        file_path=args.train_dataset_path,
        dna_rna_tokenizer=dna_rna_tokenizer,
        protein_tokenizer=protein_tokenizer,
        model_type=args.model_type,
        dna_rna_k_tokens=args.dna_rna_k_tokens,
        protein_k_tokens=args.protein_k_tokens,
        label2id=label2id,
        multi_label=args.multi_label,
        shuffle=True
    )

    eval_dataset = ClassificationDataset(
        file_path=args.eval_dataset_path,
        dna_rna_tokenizer=dna_rna_tokenizer,
        protein_tokenizer=protein_tokenizer,
        model_type=args.model_type,
        dna_rna_k_tokens=args.dna_rna_k_tokens,
        protein_k_tokens=args.protein_k_tokens,
        label2id=label2id,
        multi_label=args.multi_label,
        shuffle=False
    )
    return train_dataset, eval_dataset


def get_compute_metrics_fn(eval_name: str):
    def compute_acc_metrics(p):
        preds = p.predictions.argmax(-1)
        labels = p.label_ids
        acc = accuracy_score(labels, preds)
        return {"eval_acc": acc}
    
    def compute_mcc_metrics(p):
        preds = p.predictions.argmax(-1)
        labels = p.label_ids
        mcc = matthews_corrcoef(labels, preds)
        return {"eval_mcc": mcc}

    
    def compute_fmax_metrics(p):
        pred_np  = p.predictions
        targ_np  = p.label_ids

        pred_np = 1 / (1 + np.exp(-pred_np))
            
        pred   = torch.tensor(pred_np,  dtype=torch.float32)
        target = torch.tensor(targ_np, dtype=torch.float32)

        if pred.numel() == 0 or target.numel() == 0:
            return {"f1": 0.0}

        order = pred.argsort(descending=True, dim=1, stable=True)
        target = target.gather(1, order)
        precision = target.cumsum(1) / torch.ones_like(target).cumsum(1)
        recall = target.cumsum(1) / (target.sum(1, keepdim=True) + 1e-10)

        is_start = torch.zeros_like(target).bool()
        is_start[:, 0] = 1
        is_start = torch.scatter(is_start, 1, order, is_start)

        all_order = pred.flatten().argsort(descending=True, stable=True)
        order = order + torch.arange(order.shape[0], device=order.device).unsqueeze(1) * order.shape[1]
        order = order.flatten()
        inv_order = torch.zeros_like(order)
        inv_order[order] = torch.arange(order.shape[0], device=order.device)
        is_start = is_start.flatten()[all_order]
        all_order = inv_order[all_order]

        precision = precision.flatten()
        recall = recall.flatten()

        all_precision = precision[all_order] - \
                        torch.where(is_start, torch.zeros_like(precision), precision[all_order - 1])
        all_precision = all_precision.cumsum(0) / is_start.cumsum(0)
        all_recall = recall[all_order] - \
                    torch.where(is_start, torch.zeros_like(recall), recall[all_order - 1])
        all_recall = all_recall.cumsum(0) / pred.shape[0]
        all_f1 = 2 * all_precision * all_recall / (all_precision + all_recall + 1e-10)

        if torch.isnan(all_f1).any():
            return {"eval_fmax": 0.0}

        return {"eval_fmax": all_f1.max().item()}

        
    if eval_name == "acc":
        return compute_acc_metrics
    elif eval_name == "mcc":
        return compute_mcc_metrics
    elif eval_name == "fmax":
        return compute_fmax_metrics
    else:
        raise ValueError(f"Invalid eval_name: {eval_name}")
    


def main():
    parser = HfArgumentParser(BackboneTrainConfig)
    args = parser.parse_args_into_dataclasses()[0]
    
    dna_rna_tokenizer, protein_tokenizer = get_tokenizer(args)

    print("Training configuration:")
    print(args)

    model = get_model(args)

    train_dataset, eval_dataset = get_datasets(args, dna_rna_tokenizer, protein_tokenizer)

    compute_metrics = get_compute_metrics_fn(args.eval_metrics)

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=ClassificationCollator(),
        compute_metrics=compute_metrics
    )

    trainer.train()

if __name__ == "__main__":
    main()
