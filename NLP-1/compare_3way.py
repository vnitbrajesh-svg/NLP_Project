"""
compare_3way.py  —  3-Way Model Comparison
==========================================
Evaluates and compares three models on the Financial PhraseBank
(sentences_allagree) held-out test set, or on freshly collected unseen data.

  ┌───────────────────────────┬───────────────────────────────────────────┐
  │ Model                     │ Description                               │
  ├───────────────────────────┼───────────────────────────────────────────┤
  │ BERT (Baseline)           │ bert-base-uncased, zero-shot via fill-mask│
  │ BERT (Fine-tuned)         │ bert-base-uncased fine-tuned on           │
  │                           │   financial_sentiment.csv                 │
  │ FinBERT                   │ ProsusAI/finbert, zero-shot               │
  │                           │   (finance-domain pre-trained)            │
  └───────────────────────────┴───────────────────────────────────────────┘

Usage:
  # Original PhraseBank held-out test set
  python compare_3way.py

  # Collected unseen test set (data/collected_test_data.csv)
  python compare_3way.py --collected

Charts saved to:
  results/comparison_3way/           — original test set run
  results/comparison_3way_collected/ — collected test set run

Requires models/bert_base_finetuned_merged/ to exist.
Run finetune_bert_merged.py first if it doesn't.
"""

import os
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    accuracy_score,
    f1_score,
    precision_recall_fscore_support,
)

# ── Config ─────────────────────────────────────────────────────────────────────
_HERE               = os.path.dirname(os.path.abspath(__file__))
BERT_BASELINE_MODEL = "bert-base-uncased"
BERT_FINETUNED_DIR  = os.path.join(_HERE, "models", "bert_base_finetuned_merged")
FINBERT_MODEL       = "ProsusAI/finbert"
# Held-out 20% PhraseBank test set created by prepare_merged.py
PHRASEBANK_TEST     = os.path.join(_HERE, "data", "phrasebank_test.csv")
# Freshly collected unseen data
COLLECTED_TEST      = os.path.join(_HERE, "data", "Collected_Test_Data.csv")
BATCH_SIZE          = 32
DEVICE              = 0 if torch.cuda.is_available() else -1

# BERT zero-shot fill-mask settings
ZS_CANDIDATES = ["positive", "negative", "neutral"]
ZS_TEMPLATE   = "Overall the financial sentiment is [MASK] ."

ID2LABEL    = {0: "positive", 1: "negative", 2: "neutral"}
LABEL2ID    = {v: k for k, v in ID2LABEL.items()}
LABEL_NAMES = list(ID2LABEL.values())

COLORS = {
    "BERT (Baseline)":   "#adb5bd",   # gray
    "BERT (Fine-tuned)": "#74c0fc",   # blue
    "FinBERT":           "#198754",   # green
}

# ── Argument parsing ───────────────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(description="3-Way model comparison")
    parser.add_argument(
        "--collected", action="store_true",
        help="Evaluate on collected_test_data.csv instead of the PhraseBank held-out set"
    )
    return parser.parse_args()

# ── Test set loaders ───────────────────────────────────────────────────────────
def load_phrasebank_test() -> tuple:
    """Held-out 20% PhraseBank split created by prepare_merged.py."""
    if not os.path.exists(PHRASEBANK_TEST):
        raise FileNotFoundError(
            f"Test set not found at {PHRASEBANK_TEST}.\n"
            "Run prepare_merged.py first to generate the 80/20 PhraseBank split."
        )
    df     = pd.read_csv(PHRASEBANK_TEST)
    texts  = df["text"].tolist()
    labels = np.array(df["label_id"].tolist())
    print(f"[PhraseBank test set]  {len(texts)} samples  |  "
          f"pos={sum(labels==0)}  neg={sum(labels==1)}  neu={sum(labels==2)}")
    return texts, labels


def load_collected_test() -> tuple:
    """Freshly collected unseen data from collected_test_data.csv."""
    if not os.path.exists(COLLECTED_TEST):
        raise FileNotFoundError(f"Collected test data not found at {COLLECTED_TEST}.")
    df = pd.read_csv(COLLECTED_TEST)
    df = df.dropna(subset=["Text", "Sentiment"])
    df["label_id"] = df["Sentiment"].str.lower().str.strip().map(LABEL2ID)
    df = df[df["label_id"].notna()]

    texts  = [str(t) for t in df["Text"].tolist()]
    labels = df["label_id"].to_numpy()
    print(f"[Collected test set]  {len(texts)} samples  |  "
          f"pos={sum(labels==0)}  neg={sum(labels==1)}  neu={sum(labels==2)}")
    return texts, labels

# ── Offline-safe model loader ──────────────────────────────────────────────────
def _load_pretrained(cls, model_id: str, **kwargs):
    """
    Try loading with network first; fall back to local cache if offline.
    Raises a clear error if the model has never been downloaded.
    """
    try:
        return cls.from_pretrained(model_id, **kwargs)
    except Exception as e:
        if "connection" in str(e).lower() or "nodename" in str(e).lower() \
                or "ConnectError" in type(e).__name__:
            print(f"  Network unavailable — loading {model_id} from local cache ...")
            try:
                return cls.from_pretrained(model_id, local_files_only=True, **kwargs)
            except Exception:
                raise RuntimeError(
                    f"\n❌  '{model_id}' is not in the local HuggingFace cache "
                    f"and the network is unreachable.\n"
                    f"   Run this once while online to pre-download all models:\n"
                    f"       python download_models.py\n"
                )
        raise

# ── BERT zero-shot via fill-mask ───────────────────────────────────────────────
def predict_bert_baseline(texts: list) -> np.ndarray:
    print(f"\n[BERT (Baseline)] Loading {BERT_BASELINE_MODEL} ...")
    try:
        filler = pipeline("fill-mask", model=BERT_BASELINE_MODEL, device=DEVICE)
    except Exception as e:
        if "connection" in str(e).lower() or "nodename" in str(e).lower():
            print("  Network unavailable — loading from local cache ...")
            filler = pipeline("fill-mask", model=BERT_BASELINE_MODEL,
                              device=DEVICE, local_files_only=True)
        else:
            raise
    tokenizer = filler.tokenizer

    for w in ZS_CANDIDATES:
        ids = tokenizer.encode(w, add_special_tokens=False)
        if len(ids) != 1:
            raise ValueError(f"'{w}' is not a single token — cannot use as fill-mask target.")

    print(f"[BERT (Baseline)] Scoring {len(texts)} samples ...")
    preds = []
    for text in texts:
        prompt  = f"{ZS_TEMPLATE} {text[:300]}"
        results = filler(prompt, targets=ZS_CANDIDATES)
        best    = max(results, key=lambda r: r["score"])
        preds.append(LABEL2ID[best["token_str"].lower()])

    print("[BERT (Baseline)] Done.")
    return np.array(preds)

# ── Text-classification inference (fine-tuned BERT + FinBERT) ─────────────────
def predict_classifier(model_id: str, tag: str, texts: list) -> np.ndarray:
    print(f"\n[{tag}] Loading: {model_id}")
    try:
        tokenizer = _load_pretrained(AutoTokenizer, model_id)
    except ValueError as e:
        msg = str(e).lower()
        if "backend tokenizer" in msg or "sentencepiece" in msg or "tiktoken" in msg:
            # Some repos only expose a slow tokenizer in certain environments.
            # Fall back to use_fast=False instead of failing hard.
            print(f"[{tag}] Fast tokenizer unavailable; retrying with use_fast=False ...")
            tokenizer = _load_pretrained(AutoTokenizer, model_id, use_fast=False)
        else:
            raise
    model     = _load_pretrained(AutoModelForSequenceClassification, model_id)
    clf = pipeline(
        "text-classification",
        model=model,
        tokenizer=tokenizer,
        device=DEVICE,
        batch_size=BATCH_SIZE,
        truncation=True,
    )
    print(f"[{tag}] Running inference on {len(texts)} samples ...")
    raw   = clf(texts)
    preds = np.array([LABEL2ID[p["label"].lower()] for p in raw])
    print(f"[{tag}] Done.")
    return preds

# ── Compute & display metrics ──────────────────────────────────────────────────
def compute_metrics(true: np.ndarray, preds: np.ndarray, tag: str, results_dir: str) -> dict:
    report_str = classification_report(
        true, preds, target_names=LABEL_NAMES, zero_division=0
    )
    print(f"\n{'='*60}\n  {tag}\n{'='*60}")
    print(report_str)

    os.makedirs(results_dir, exist_ok=True)
    safe = tag.replace(" ", "_").replace("(", "").replace(")", "").replace("-", "_")
    with open(os.path.join(results_dir, f"{safe}_report.txt"), "w") as fh:
        fh.write(f"Model: {tag}\n{'='*60}\n{report_str}")

    prec, rec, f1_per, _ = precision_recall_fscore_support(
        true, preds, labels=[0, 1, 2], average=None, zero_division=0
    )
    return {
        "accuracy":     round(accuracy_score(true, preds), 4),
        "f1_macro":     round(f1_score(true, preds, average="macro",    zero_division=0), 4),
        "f1_weighted":  round(f1_score(true, preds, average="weighted", zero_division=0), 4),
        "per_class": {
            LABEL_NAMES[i]: {
                "precision": round(float(prec[i]), 4),
                "recall":    round(float(rec[i]),  4),
                "f1":        round(float(f1_per[i]), 4),
            }
            for i in range(3)
        },
        "confusion_matrix": confusion_matrix(true, preds).tolist(),
    }

# ── Plot: overall metrics bar chart ───────────────────────────────────────────
def plot_metrics(metrics: dict, results_dir: str, subtitle: str):
    keys   = ["accuracy", "f1_macro", "f1_weighted"]
    xlbls  = ["Accuracy", "F1 Macro", "F1 Weighted"]
    models = list(metrics.keys())
    x, w   = np.arange(len(keys)), 0.25

    fig, ax = plt.subplots(figsize=(10, 5.5))
    for i, m in enumerate(models):
        bars = ax.bar(
            x + i * w, [metrics[m][k] for k in keys], w,
            label=m, color=COLORS[m], alpha=0.9, edgecolor="white",
        )
        for bar in bars:
            v = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2, v + 0.007,
                f"{v:.3f}", ha="center", va="bottom", fontsize=8, fontweight="bold",
            )

    ax.set_xticks(x + w)
    ax.set_xticklabels(xlbls, fontsize=11)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Score", fontsize=11)
    ax.yaxis.grid(True, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title(
        f"3-Way Model Comparison — Overall Metrics\n{subtitle}",
        fontsize=11, fontweight="bold", pad=10,
    )
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(results_dir, "metrics_comparison.png"), dpi=150)
    plt.close(fig)
    print(f"Saved → {results_dir}/metrics_comparison.png")


# ── Plot: per-class F1 ────────────────────────────────────────────────────────
def plot_per_class_f1(metrics: dict, results_dir: str):
    models = list(metrics.keys())
    x, w   = np.arange(len(LABEL_NAMES)), 0.25

    fig, ax = plt.subplots(figsize=(10, 5.5))
    for i, m in enumerate(models):
        vals = [metrics[m]["per_class"][c]["f1"] for c in LABEL_NAMES]
        bars = ax.bar(
            x + i * w, vals, w,
            label=m, color=COLORS[m], alpha=0.9, edgecolor="white",
        )
        for bar in bars:
            v = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2, v + 0.007,
                f"{v:.3f}", ha="center", va="bottom", fontsize=8, fontweight="bold",
            )

    ax.set_xticks(x + w)
    ax.set_xticklabels([c.capitalize() for c in LABEL_NAMES], fontsize=11)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("F1 Score", fontsize=11)
    ax.yaxis.grid(True, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title("Per-Class F1  —  3-Way Comparison", fontsize=13, fontweight="bold", pad=10)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(results_dir, "per_class_f1.png"), dpi=150)
    plt.close(fig)
    print(f"Saved → {results_dir}/per_class_f1.png")


# ── Plot: confusion matrices ──────────────────────────────────────────────────
def plot_confusion_matrices(metrics: dict, results_dir: str, subtitle: str):
    models = list(metrics.keys())
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(
        f"Confusion Matrices  —  3-Way Model Comparison\n({subtitle})",
        fontsize=12, fontweight="bold",
    )
    for ax, m in zip(axes, models):
        cm = np.array(metrics[m]["confusion_matrix"])
        ConfusionMatrixDisplay(
            cm, display_labels=[c.capitalize() for c in LABEL_NAMES]
        ).plot(ax=ax, cmap="Blues", colorbar=False)
        ax.set_title(
            f"{m}\nAcc={metrics[m]['accuracy']:.3f}  F1={metrics[m]['f1_weighted']:.3f}",
            fontsize=9, fontweight="bold", pad=8,
        )
        ax.set_xlabel("Predicted", fontsize=9)
        ax.set_ylabel("True", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(results_dir, "confusion_matrices.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {results_dir}/confusion_matrices.png")


# ── Plot: improvement over BERT baseline ──────────────────────────────────────
def plot_improvement(metrics: dict, results_dir: str):
    baseline = metrics["BERT (Baseline)"]
    compare  = ["BERT (Fine-tuned)", "FinBERT"]
    keys     = ["accuracy", "f1_macro", "f1_weighted"]
    xlbls    = ["Accuracy", "F1 Macro", "F1 Weighted"]
    x, w     = np.arange(len(keys)), 0.3

    fig, ax = plt.subplots(figsize=(9, 5))
    for i, m in enumerate(compare):
        deltas = [metrics[m][k] - baseline[k] for k in keys]
        bars   = ax.bar(
            x + i * w, deltas, w,
            label=m, color=COLORS[m], alpha=0.9, edgecolor="white",
        )
        for bar, d in zip(bars, deltas):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + (0.004 if d >= 0 else -0.018),
                f"{d:+.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold",
            )

    ax.axhline(0, color="black", linewidth=0.9, linestyle="--")
    ax.set_xticks(x + w / 2)
    ax.set_xticklabels(xlbls, fontsize=11)
    ax.set_ylabel("Δ Score vs BERT Baseline", fontsize=11)
    ax.yaxis.grid(True, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title(
        "Improvement Over BERT Baseline (Zero-Shot)\n"
        "Shows contribution of fine-tuning and domain pre-training",
        fontsize=11, fontweight="bold", pad=10,
    )
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(results_dir, "improvement_over_bert.png"), dpi=150)
    plt.close(fig)
    print(f"Saved → {results_dir}/improvement_over_bert.png")


# ── Terminal summary table ─────────────────────────────────────────────────────
def print_summary_table(metrics: dict):
    models = list(metrics.keys())
    cols   = ["accuracy", "f1_macro", "f1_weighted"]
    col_w  = 22
    sep    = "=" * (24 + col_w * len(models))

    print(f"\n{sep}")
    print(f"  {'Metric':<22}" + "".join(f"{m:>{col_w}}" for m in models))
    print(sep)
    for c in cols:
        row = f"  {c:<22}" + "".join(f" {metrics[m][c]:>{col_w-1}.4f}" for m in models)
        print(row)
    print(sep)
    print("\nPer-class F1:")
    for cls in LABEL_NAMES:
        row = f"  F1 {cls:<18}" + "".join(
            f" {metrics[m]['per_class'][cls]['f1']:>{col_w-1}.4f}" for m in models
        )
        print(row)
    print(sep)


# ── Main ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    args = parse_args()

    if args.collected:
        results_dir  = os.path.join(_HERE, "results", "comparison_3way_collected")
        subtitle     = "Test: Collected Unseen Data (collected_test_data.csv)"
        preds_prefix = "collected_"
        texts, true_labels = load_collected_test()
    else:
        results_dir  = os.path.join(_HERE, "results", "comparison_3way")
        subtitle     = "Test: Financial PhraseBank (AllAgree)"
        preds_prefix = ""
        texts, true_labels = load_phrasebank_test()

    os.makedirs(results_dir, exist_ok=True)
    all_metrics = {}

    # 1. BERT Baseline (zero-shot fill-mask)
    preds = predict_bert_baseline(texts)
    all_metrics["BERT (Baseline)"] = compute_metrics(true_labels, preds, "BERT (Baseline)", results_dir)
    np.save(os.path.join(results_dir, f"{preds_prefix}bert_baseline_preds.npy"), preds)

    # 2. BERT Fine-tuned on financial_sentiment.csv
    if not os.path.isdir(BERT_FINETUNED_DIR):
        raise FileNotFoundError(
            f"Fine-tuned model not found at {BERT_FINETUNED_DIR}.\n"
            "Run finetune_bert_merged.py first."
        )
    preds = predict_classifier(BERT_FINETUNED_DIR, "BERT (Fine-tuned)", texts)
    all_metrics["BERT (Fine-tuned)"] = compute_metrics(true_labels, preds, "BERT (Fine-tuned)", results_dir)
    np.save(os.path.join(results_dir, f"{preds_prefix}bert_finetuned_preds.npy"), preds)

    # 3. FinBERT zero-shot (finance-domain pre-trained)
    preds = predict_classifier(FINBERT_MODEL, "FinBERT", texts)
    all_metrics["FinBERT"] = compute_metrics(true_labels, preds, "FinBERT", results_dir)
    np.save(os.path.join(results_dir, f"{preds_prefix}finbert_preds.npy"), preds)

    # Print table
    print_summary_table(all_metrics)

    # Save JSON
    summary = {
        k: {m: v for m, v in vals.items() if m != "confusion_matrix"}
        for k, vals in all_metrics.items()
    }
    with open(os.path.join(results_dir, "comparison_3way.json"), "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"\nSaved → {results_dir}/comparison_3way.json")

    # Generate charts
    print("\nGenerating charts ...")
    plot_metrics(all_metrics, results_dir, subtitle)
    plot_per_class_f1(all_metrics, results_dir)
    plot_confusion_matrices(all_metrics, results_dir, subtitle)
    plot_improvement(all_metrics, results_dir)

    print(f"\n✓ Done. Results saved to {results_dir}/")
