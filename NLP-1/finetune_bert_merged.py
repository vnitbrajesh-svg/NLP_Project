"""
finetune_bert_merged.py  —  Fine-Tune bert-base-uncased on Financial Sentiment Dataset
=======================================================================================
Fine-tunes bert-base-uncased on data/financial_sentiment.csv.

Prerequisites:
  - data/financial_sentiment.csv must exist
  - data/tokenized_dataset_merged/ must exist (run data_preparation.py first)

Output:
  models/bert_base_finetuned_merged/   (best checkpoint, ready for inference)

Architecture choices:
  - All 12 encoder layers trainable (FREEZE_LAYERS = 0)
  - Class-weighted cross-entropy to handle class imbalance
  - Early stopping (patience 3) to prevent overfitting
  - fp16 training when a CUDA GPU is available

Label mapping:
  0 → positive  |  1 → negative  |  2 → neutral
"""

import os
import argparse
from typing import Optional
import numpy as np
import torch
import torch.nn as nn
from datasets import DatasetDict
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
)
from sklearn.metrics import f1_score, accuracy_score
from sklearn.utils.class_weight import compute_class_weight

# ── Config ────────────────────────────────────────────────────────────────────
_HERE         = os.path.dirname(os.path.abspath(__file__))
MODEL_NAME    = "bert-base-uncased"
DATASET_DIR   = os.path.join(_HERE, "data", "tokenized_dataset_merged")
OUTPUT_DIR    = os.path.join(_HERE, "models", "bert_base_finetuned_merged")
LOGGING_DIR   = os.path.join(_HERE, "logs", "bert_merged")

LEARNING_RATE  = 2e-5       # fixed for next sweep while tuning batch and grad accumulation
NUM_EPOCHS     = 10         # more epochs to compensate for generic pre-training
BATCH_SIZE     = 16         # per-device batch size
GRAD_ACCUM     = 2          # effective batch = BATCH_SIZE * GRAD_ACCUM
WEIGHT_DECAY   = 0.01
WARMUP_RATIO   = 0.1
EARLY_STOP_PAT = 3          # more patience to match longer training

ID2LABEL = {0: "positive", 1: "negative", 2: "neutral"}
LABEL2ID = {v: k for k, v in ID2LABEL.items()}



def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune bert-base-uncased on merged dataset")
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default=DATASET_DIR,
        help="Path to tokenized dataset directory (DatasetDict on disk)",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=LEARNING_RATE,
        help="Learning rate to use for this run (default: config LEARNING_RATE)",
    )
    parser.add_argument(
        "--output-suffix",
        type=str,
        default="",
        help="Optional suffix for output/log directories, e.g. lr2e5",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=BATCH_SIZE,
        help="Per-device batch size",
    )
    parser.add_argument(
        "--grad-accum",
        type=int,
        default=GRAD_ACCUM,
        help="Gradient accumulation steps (effective batch = batch-size * grad-accum)",
    )
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=NUM_EPOCHS,
        help="Maximum number of training epochs",
    )
    parser.add_argument(
        "--early-stop-pat",
        type=int,
        default=EARLY_STOP_PAT,
        help="Early stopping patience in epochs",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=WEIGHT_DECAY,
        help="Weight decay",
    )
    parser.add_argument(
        "--warmup-ratio",
        type=float,
        default=WARMUP_RATIO,
        help="Warmup ratio (ignored if --warmup-steps is set)",
    )
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=None,
        help="Warmup steps (overrides --warmup-ratio when set)",
    )
    parser.add_argument(
        "--class-weight-strategy",
        type=str,
        default="balanced",
        choices=["balanced", "sqrt_balanced", "none"],
        help="Class weight strategy for cross-entropy loss",
    )
    return parser.parse_args()

# ── Class weights ─────────────────────────────────────────────────────────────
def compute_class_weights(label_ids, strategy: str) -> torch.Tensor:
    weights = compute_class_weight(
        class_weight="balanced",
        classes=np.array(list(ID2LABEL.keys())),
        y=np.array(label_ids),
    )
    if strategy == "none":
        weights = np.ones(len(ID2LABEL), dtype=float)
    elif strategy == "sqrt_balanced":
        weights = np.sqrt(weights)
    print(f"Class weight strategy: {strategy}")
    print(f"Class weights: { {ID2LABEL[i]: round(float(w), 3) for i, w in enumerate(weights)} }")
    return torch.tensor(weights, dtype=torch.float)

# ── Weighted trainer (handles class imbalance) ────────────────────────────────
class WeightedTrainer(Trainer):
    def __init__(self, class_weights: torch.Tensor, **kwargs):
        super().__init__(**kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels  = inputs.pop("labels")
        outputs = model(**inputs)
        loss_fn = nn.CrossEntropyLoss(
            weight=self.class_weights.to(outputs.logits.device)
        )
        loss = loss_fn(outputs.logits, labels)
        return (loss, outputs) if return_outputs else loss

# ── Model ─────────────────────────────────────────────────────────────────────
def load_model():
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=3,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {trainable:,} trainable / {total:,} total")
    return model

# ── Metrics ───────────────────────────────────────────────────────────────────
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy":    accuracy_score(labels, preds),
        "f1_weighted": f1_score(labels, preds, average="weighted"),
        "f1_macro":    f1_score(labels, preds, average="macro"),
    }

# ── Training args ─────────────────────────────────────────────────────────────
def get_training_args(
    learning_rate: float,
    output_dir: str,
    logging_dir: str,
    batch_size: int,
    grad_accum: int,
    num_epochs: int,
    weight_decay: float,
    warmup_ratio: float,
    warmup_steps: Optional[int],
):
    kwargs = {
        "output_dir": output_dir,
        "logging_dir": logging_dir,
        "num_train_epochs": num_epochs,
        "per_device_train_batch_size": batch_size,
        "per_device_eval_batch_size": batch_size,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "gradient_accumulation_steps": grad_accum,
        "eval_strategy": "epoch",
        "save_strategy": "epoch",
        "load_best_model_at_end": True,
        "metric_for_best_model": "f1_weighted",
        "greater_is_better": True,
        "lr_scheduler_type": "cosine",
        "logging_steps": 50,
        "report_to": "none",
        "fp16": torch.cuda.is_available(),
        "seed": 42,
    }
    if warmup_steps is not None:
        kwargs["warmup_steps"] = warmup_steps
    else:
        kwargs["warmup_ratio"] = warmup_ratio
    return TrainingArguments(**kwargs)

# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    args = parse_args()
    run_suffix = f"_{args.output_suffix}" if args.output_suffix else ""
    run_output_dir = f"{OUTPUT_DIR}{run_suffix}"
    run_logging_dir = f"{LOGGING_DIR}{run_suffix}"

    os.makedirs(run_output_dir, exist_ok=True)
    os.makedirs(run_logging_dir, exist_ok=True)

    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    dataset_dir = args.dataset_dir
    dataset   = DatasetDict.load_from_disk(dataset_dir)
    model     = load_model()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    label_ids     = [int(l) for l in dataset["train"]["labels"]]
    class_weights = compute_class_weights(label_ids, args.class_weight_strategy)

    trainer = WeightedTrainer(
        class_weights   = class_weights,
        model           = model,
        args            = get_training_args(
            args.learning_rate,
            run_output_dir,
            run_logging_dir,
            args.batch_size,
            args.grad_accum,
            args.num_epochs,
            args.weight_decay,
            args.warmup_ratio,
            args.warmup_steps,
        ),
        train_dataset   = dataset["train"],
        eval_dataset    = dataset["val"],
        compute_metrics = compute_metrics,
        callbacks       = [EarlyStoppingCallback(early_stopping_patience=args.early_stop_pat)],
    )

    print("\n── Fine-tuning bert-base-uncased on merged dataset ──────────────")
    print(f"   Model:      {MODEL_NAME}")
    print(f"   Dataset:    {dataset_dir}  ({len(dataset['train'])} train samples)")
    print(
        f"   Epochs:     {args.num_epochs}  |  LR: {args.learning_rate}  |  "
        f"Batch: {args.batch_size}  |  GradAccum: {args.grad_accum}  |  "
        f"Effective batch: {args.batch_size * args.grad_accum}"
    )
    print(
        f"   WeightDecay: {args.weight_decay}  |  Warmup: "
        f"{'steps=' + str(args.warmup_steps) if args.warmup_steps is not None else 'ratio=' + str(args.warmup_ratio)}  |  "
        f"EarlyStopPat: {args.early_stop_pat}"
    )
    trainer.train()

    print("\n── Saving best model ─────────────────────────────────────────────")
    trainer.save_model(run_output_dir)
    tokenizer.save_pretrained(run_output_dir)
    print(f"✓ Saved → {run_output_dir}/")
