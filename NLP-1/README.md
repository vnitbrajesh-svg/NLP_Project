# Financial News Sentiment Analysis

A pipeline for financial news sentiment classification using transformer models.
Compares three approaches — BERT zero-shot, BERT fine-tuned, and FinBERT zero-shot —
on financial news headlines, with support for evaluation on both a held-out PhraseBank
test set and freshly collected unseen data.

---

## Models Compared

| Model | Type | Description |
|---|---|---|
| BERT (Zero-shot) | `bert-base-uncased` | Zero-shot via fill-mask — no fine-tuning |
| BERT (Fine-tuned) | `bert-base-uncased` | Fine-tuned on `financial_sentiment.csv` |
| FinBERT (Zero-shot) | `ProsusAI/finbert` | Finance-domain pre-trained, no fine-tuning |

---

## Project Structure

```
NLP-1/
│
├── data_preparation.py       # Stage 1 — tokenize & split financial_sentiment.csv
├── finetune_bert_merged.py   # Stage 2 — fine-tune bert-base-uncased
├── compare_3way.py           # Stage 3 — 3-way model comparison (main evaluation)
├── plot_training.py          # Plot training/validation curves from checkpoints
├── download_models.py        # Download all required models (run once after clone)
├── upload_model_to_hub.py    # [Maintainer only] Upload fine-tuned model to HF Hub
├── requirements.txt
│
├── data/
│   ├── financial_sentiment.csv       # Labelled training data (text | sentiment)
│   ├── phrasebank_test.csv           # Held-out PhraseBank test set
│   └── Collected_Test_Data.csv       # Unseen collected data (--collected flag)
│
├── models/                           # Created by download_models.py (git-ignored)
│   └── bert_base_finetuned_merged/   # Fine-tuned BERT weights
│
├── results/                          # Auto-created on each run (git-ignored)
└── logs/                             # Auto-created during training (git-ignored)
```

> `models/`, `results/`, and `logs/` are git-ignored (large files / auto-generated).
> Run `python download_models.py` once after cloning to restore the fine-tuned model.

---

## Quickstart for Teammates

### 1 — Clone the repo

```bash
git clone https://github.com/chndnchugh/NLP_Project.git
cd NLP_Project/NLP-1
```

### 2 — Create a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate        # macOS / Linux
# .venv\Scripts\activate         # Windows
```

### 3 — Install dependencies

**CPU only:**
```bash
pip install -r requirements.txt
```

**With GPU (CUDA 12.1):**
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

### 4 — Download all models (one-time, ~1 GB total)

```bash
python download_models.py
```

This downloads:
- `bert-base-uncased` and `ProsusAI/finbert` into the HuggingFace cache
- The fine-tuned BERT weights (~418 MB) into `models/bert_base_finetuned_merged/`

---

## Running the Pipeline

All commands are run from inside the `NLP-1/` directory.

### Option A — Skip training, use pre-downloaded model (recommended)

After running `download_models.py`, jump straight to evaluation:

```bash
python compare_3way.py              # PhraseBank held-out test set
python compare_3way.py --collected  # Freshly collected unseen data
```

---

### Option B — Full pipeline (retrain from scratch)

#### Stage 1: Prepare & tokenize the dataset
```bash
python data_preparation.py
```
Loads `data/financial_sentiment.csv`, splits 80/10/10 (stratified), tokenizes with
`bert-base-uncased`, saves to `data/tokenized_dataset/` and `data/tokenized_dataset_merged/`.

#### Stage 2: Fine-tune BERT
```bash
python finetune_bert_merged.py
```
Fine-tunes `bert-base-uncased` with class-weighted cross-entropy loss and early
stopping (patience = 3 on F1 weighted). Saves best checkpoint to
`models/bert_base_finetuned_merged/`.

> Training time: ~20 min on GPU, ~2–3 hours on CPU.

#### Stage 3: 3-way comparison
```bash
python compare_3way.py              # PhraseBank held-out test set
python compare_3way.py --collected  # Freshly collected unseen data
```

---

## Output Charts

Each run of `compare_3way.py` produces four charts in `results/comparison_3way/`
(or `results/comparison_3way_collected/` with `--collected`):

| File | Description |
|---|---|
| `metrics_comparison.png` | Grouped bar chart — Accuracy, F1 Macro, F1 Weighted |
| `per_class_f1.png` | Per-class F1 for Positive / Negative / Neutral |
| `confusion_matrices.png` | 1×3 confusion matrix grid |
| `improvement_over_bert.png` | Delta over BERT zero-shot baseline |

## Report Summary

The generated reports show a clear ranking difference between the two evaluation sets:

| Dataset | Model | Accuracy | F1 Macro | F1 Weighted | Positive F1 | Negative F1 | Neutral F1 |
|---|---|---:|---:|---:|---:|---:|---:|
| PhraseBank held-out | BERT (Baseline) | 0.2737 | 0.2373 | 0.1436 | 0.4080 | 0.3038 | 0.0000 |
| PhraseBank held-out | BERT (Fine-tuned) | 0.7174 | 0.5691 | 0.6989 | 0.6245 | 0.2564 | 0.8265 |
| PhraseBank held-out | FinBERT | 0.9757 | 0.9646 | 0.9761 | 0.9780 | 0.9302 | 0.9855 |
| Collected unseen | BERT (Baseline) | 0.3870 | 0.2657 | 0.2660 | 0.5317 | 0.2654 | 0.0000 |
| Collected unseen | BERT (Fine-tuned) | 0.7490 | 0.7380 | 0.7381 | 0.8601 | 0.7242 | 0.6296 |
| Collected unseen | FinBERT | 0.5940 | 0.5707 | 0.5709 | 0.7469 | 0.5866 | 0.3786 |

The reports indicate that FinBERT is the strongest model on the held-out PhraseBank test set, while the fine-tuned BERT model generalises best on the collected unseen data. The zero-shot BERT baseline is consistently the weakest, especially for the neutral class.

---

## Plot Training Curves

```bash
python plot_training.py
```

Reads `trainer_state.json` from the best checkpoint and saves to `results/`:

| File | Description |
|---|---|
| `training_loss.png` | Raw + smoothed (window=10) training loss vs step |
| `validation_loss.png` | Validation loss per epoch, best checkpoint highlighted |
| `validation_metrics.png` | Accuracy, F1 Weighted, F1 Macro per epoch |

---

## Fine-Tuning Hyperparameters

Configured in `finetune_bert_merged.py`:

| Parameter | Value | Notes |
|---|---|---|
| `LEARNING_RATE` | `1e-5` | Conservative — gentle adaptation of pre-trained weights |
| `NUM_EPOCHS` | `10` | Early stopping typically triggers before this |
| `BATCH_SIZE` | `8` | With `GRAD_ACCUM=2` → effective batch of 16 |
| `EARLY_STOP_PAT` | `3` | Stops if val F1 weighted doesn't improve for 3 epochs |
| `WEIGHT_DECAY` | `0.01` | L2 regularisation |
| `WARMUP_RATIO` | `0.1` | 10% of steps for LR warmup, cosine decay after |

Best checkpoint: **epoch 8, step 3440** — Accuracy = 0.8215, F1 Weighted = 0.8217

---

## Dataset Format

`data/financial_sentiment.csv` — columns used by the pipeline:

| text | sentiment |
|---|---|
| Sebi clears 4 IPOs amid rising demand. | positive |
| IT rout drags Nifty to third straight loss. | negative |
| Sales were broadly in line with expectations. | neutral |

`data/Collected_Test_Data.csv` must use `Text` and `Sentiment` column names to work with `--collected`.

---

## Troubleshooting

**`FileNotFoundError: models/bert_base_finetuned_merged`**
→ Run `python download_models.py` first.

**`FileNotFoundError: data/tokenized_dataset`**
→ Run `python data_preparation.py` first (only needed for full pipeline / retraining).

**CUDA out of memory during training**
→ Reduce `BATCH_SIZE` to `4` and set `GRAD_ACCUM = 4` in `finetune_bert_merged.py`.

**Slow inference on CPU**
→ The BERT fill-mask zero-shot step is the bottleneck. Full evaluation on 500+ samples
  takes ~10 min on CPU — this is expected.
