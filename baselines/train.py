import torch
from dataclasses import dataclass, field
from transformers import TrainingArguments, HfArgumentParser, set_seed, AutoTokenizer, Trainer, AutoModelForMaskedLM
from dataset import ClassificationDataset, ClassificationCollator
from model import BackboneWithClsHead
from sklearn.metrics import accuracy_score

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
    bf16: bool = field(default=True, metadata={"help": "Use bfloat16 mixed precision"})


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
    )
    return model


def get_datasets(args: BackboneTrainConfig, dna_rna_tokenizer=None, protein_tokenizer=None):
    train_dataset = ClassificationDataset(
        file_path=args.train_dataset_path,
        dna_rna_tokenizer=dna_rna_tokenizer,
        protein_tokenizer=protein_tokenizer,
        model_type=args.model_type,
        dna_rna_k_tokens=args.dna_rna_k_tokens,
        protein_k_tokens=args.protein_k_tokens,
        shuffle=True
    )

    eval_dataset = ClassificationDataset(
        file_path=args.eval_dataset_path,
        dna_rna_tokenizer=dna_rna_tokenizer,
        protein_tokenizer=protein_tokenizer,
        model_type=args.model_type,
        dna_rna_k_tokens=args.dna_rna_k_tokens,
        protein_k_tokens=args.protein_k_tokens,
        shuffle=False
    )
    return train_dataset, eval_dataset

def compute_metrics(p):
    preds = p.predictions.argmax(-1)
    labels = p.label_ids
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc}



def main():
    parser = HfArgumentParser(BackboneTrainConfig)
    args = parser.parse_args_into_dataclasses()[0]
    
    dna_rna_tokenizer, protein_tokenizer = get_tokenizer(args)

    print("Training configuration:")
    print(args)

    model = get_model(args)

    train_dataset, eval_dataset = get_datasets(args, dna_rna_tokenizer, protein_tokenizer)

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