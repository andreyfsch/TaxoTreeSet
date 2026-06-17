#!/usr/bin/env python3
"""
Example: DNABERT-2 + LoRA fine-tuning for a single TaxoTreeSet classification head.

This script is NOT part of the taxotreeset package and is not installed with it.
It is provided as a reference for how to consume the parquet datasets produced by
`taxotreeset generate`. For real fine-tuning runs, copy this script to your own
project and manage its dependencies there.

Dependencies (not in taxotreeset's pyproject.toml):
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
    pip install transformers peft datasets scikit-learn accelerate sentencepiece

Usage:
    python examples/finetune_head.py \
        --data-dir   data/datasets/<lineage>/<taxid> \
        --output-dir runs/<taxid>

The data directory must contain train.parquet, val.parquet, and test.parquet
with columns [seq: str, class_idx: int32].

Outputs written to --output-dir:
    adapter/          LoRA adapter weights (loadable with PeftModel.from_pretrained)
    metrics.json      train/val loss + val/test accuracy per epoch, final test accuracy
    run_config.json   all hyperparameters and paths (for reproducibility)
"""

import argparse
import json
import logging
import math
import os
import time
from pathlib import Path

# Must be set before any CUDA allocation; expandable_segments eliminates
# fragmentation-induced OOM on long training runs.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    classification_report,
    confusion_matrix,
)
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)

import sys
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Fixed hyperparameters (validated in Andrey's dissertation on viral data,
# 7 taxonomic ranks, DNABERT-2 + LoRA rank 8)
# ---------------------------------------------------------------------------
MODEL_ID = "zhihan1996/DNABERT-2-117M"
LORA_RANK = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.1
LORA_TARGET_MODULES = ["Wqkv"]  # DNABERT-2 fuses Q/K/V into a single projection
LEARNING_RATE = 1e-3
NUM_EPOCHS = 5
WARMUP_RATIO = 0.06
WEIGHT_DECAY = 0.01
MAX_LENGTH = 128
# ---------------------------------------------------------------------------


def load_splits(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train = pd.read_parquet(data_dir / "train.parquet")
    val = pd.read_parquet(data_dir / "val.parquet")
    test = pd.read_parquet(data_dir / "test.parquet")
    for name, df in [("train", train), ("val", val), ("test", test)]:
        assert "seq" in df.columns and "class_idx" in df.columns, (
            f"{name}.parquet missing expected columns"
        )
    return train, val, test


def build_hf_dataset(df: pd.DataFrame) -> Dataset:
    return Dataset.from_dict({"seq": df["seq"].tolist(), "label": df["class_idx"].astype(int).tolist()})


def tokenize_fn(batch, tokenizer):
    return tokenizer(
        batch["seq"],
        max_length=MAX_LENGTH,
        padding=False,  # dynamic padding via DataCollatorWithPadding
        truncation=True,
    )


def preprocess_logits_for_metrics(logits, labels):
    """Keep only the classification logits before eval-prediction accumulation.

    DNABERT-2 via PEFT returns a tuple ``(logits, hidden_states, ...)``. Without
    this hook the Trainer accumulates every returned tensor across the whole eval
    set on the GPU, and the hidden states (``[N, seq_len, hidden]``) exhaust VRAM
    even for tiny label counts. Returning just the logits keeps the accumulated
    tensor at ``[N, num_labels]``.

    Args:
        logits: Model output; a tuple whose first element is the logits, or the
            logits tensor directly.
        labels: Ground-truth labels (unused; required by the Trainer signature).

    Returns:
        The classification logits tensor.
    """
    if isinstance(logits, tuple):
        logits = logits[0]
    return logits


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    # logits already reduced to [N, num_labels] by preprocess_logits_for_metrics,
    # but stay defensive in case the hook is ever removed.
    if isinstance(logits, tuple):
        logits = logits[0]
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy":       accuracy_score(labels, preds),
        "f1_macro":       f1_score(labels, preds, average="macro",    zero_division=0),
        "f1_weighted":    f1_score(labels, preds, average="weighted", zero_division=0),
        "precision_macro": precision_score(labels, preds, average="macro",    zero_division=0),
        "recall_macro":    recall_score(labels, preds,    average="macro",    zero_division=0),
    }


class EpochMetricsCallback(TrainerCallback):
    """Accumulates per-epoch metrics and writes progress.json after each step.

    progress.json is written directly to disk (bypasses stdout/stderr buffering)
    so it's always readable during training regardless of pipe buffering.
    """

    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.epoch_logs: list[dict] = []
        self._progress_path = output_dir / "progress.json"

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs:
            return
        if "epoch" in logs:
            self.epoch_logs.append({k: v for k, v in logs.items()})
        progress = {
            "global_step": state.global_step,
            "max_steps": state.max_steps,
            "epoch": round(state.epoch or 0, 3),
            "num_train_epochs": args.num_train_epochs,
            "pct_done": round(100 * state.global_step / max(state.max_steps, 1), 1),
            "recent_logs": logs,
            "epoch_logs": self.epoch_logs,
            "updated_at": time.strftime("%H:%M:%S"),
        }
        self._progress_path.write_text(json.dumps(progress, indent=2))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", required=True, type=Path,
                   help="Directory with train/val/test.parquet")
    p.add_argument("--output-dir", required=True, type=Path,
                   help="Where to write adapter/ and metrics.json")
    p.add_argument("--batch-size", type=int, default=8,
                   help="Per-device batch size")
    p.add_argument("--grad-accum", type=int, default=1,
                   help="Gradient accumulation steps")
    p.add_argument("--fp16", action="store_true", default=True,
                   help="Use mixed-precision (default: on if CUDA available)")
    p.add_argument("--no-fp16", dest="fp16", action="store_false")
    p.add_argument("--resume-from-checkpoint", type=Path, default=None,
                   help="Resume training from a saved checkpoint directory")
    return p.parse_args()


def main():
    args = parse_args()

    data_dir = args.data_dir.resolve()
    output_dir = args.output_dir.resolve()
    adapter_dir = output_dir / "adapter"
    output_dir.mkdir(parents=True, exist_ok=True)

    use_fp16 = args.fp16 and torch.cuda.is_available()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info("Device: %s | fp16: %s", device, use_fp16)

    # ---- data ----
    log.info("Loading splits from %s", data_dir)
    df_train, df_val, df_test = load_splits(data_dir)
    num_labels = int(df_train["class_idx"].max()) + 1
    log.info("train=%d val=%d test=%d classes=%d", len(df_train), len(df_val), len(df_test), num_labels)

    # ---- tokenizer ----
    log.info("Loading tokenizer %s", MODEL_ID)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)

    ds_train = build_hf_dataset(df_train).map(lambda b: tokenize_fn(b, tokenizer), batched=True, remove_columns=["seq"])
    ds_val   = build_hf_dataset(df_val).map(lambda b: tokenize_fn(b, tokenizer), batched=True, remove_columns=["seq"])
    ds_test  = build_hf_dataset(df_test).map(lambda b: tokenize_fn(b, tokenizer), batched=True, remove_columns=["seq"])

    # ---- model + LoRA ----
    log.info("Loading base model %s", MODEL_ID)
    from transformers import AutoConfig
    model_config = AutoConfig.from_pretrained(MODEL_ID, trust_remote_code=True)
    model_config.num_labels = num_labels
    base_model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_ID,
        config=model_config,
        trust_remote_code=True,
    )

    lora_cfg = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=LORA_TARGET_MODULES,
        bias="none",
    )
    model = get_peft_model(base_model, lora_cfg)
    model.print_trainable_parameters()

    # ---- training args ----
    # Use ceil to match what the Trainer actually counts as steps per epoch.
    train_steps_per_epoch = math.ceil(len(df_train) / (args.batch_size * args.grad_accum))
    # Checkpoint ~3× per epoch for crash resilience. With load_best_model_at_end,
    # Transformers requires save_steps to be a multiple of eval_steps, so the only
    # way to save mid-epoch is to also eval mid-epoch: set eval_steps == save_steps.
    # eval_steps need not divide the epoch length; eval is cheap on the small val
    # split, so eval'ing ~3×/epoch is acceptable.
    eval_save_every = max(train_steps_per_epoch // 3, 50)
    # Early stopping is measured per eval, so its patience must scale with eval
    # frequency. Evaluating ~3×/epoch with patience=2 would stop after only ~0.7
    # epoch without improvement; tolerate ~2 full epochs of no improvement instead.
    evals_per_epoch = max(1, round(train_steps_per_epoch / eval_save_every))
    early_stopping_patience = 2 * evals_per_epoch
    training_args = TrainingArguments(
        output_dir=str(output_dir / "checkpoints"),
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size // 2,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=LEARNING_RATE,
        warmup_ratio=WARMUP_RATIO,
        weight_decay=WEIGHT_DECAY,
        lr_scheduler_type="cosine",
        eval_strategy="steps",
        eval_steps=eval_save_every,
        eval_accumulation_steps=20,  # offload eval preds to CPU; avoids VRAM OOM
        save_strategy="steps",
        save_steps=eval_save_every,
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        label_smoothing_factor=0.1,
        fp16=use_fp16,
        logging_steps=25,
        report_to="none",
        save_total_limit=2,
        dataloader_num_workers=0,
    )

    metrics_cb = EpochMetricsCallback(output_dir)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ds_train,
        eval_dataset=ds_val,
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        callbacks=[metrics_cb, EarlyStoppingCallback(early_stopping_patience=early_stopping_patience)],
    )

    # ---- train ----
    log.info("Starting training")
    t0 = time.time()
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    elapsed = time.time() - t0
    log.info("Training done in %.1f min", elapsed / 60)

    # ---- evaluate on test set ----
    log.info("Evaluating on test set")
    test_results = trainer.evaluate(ds_test, metric_key_prefix="test")
    log.info("Test f1_macro=%.4f  accuracy=%.4f",
             test_results.get("test_f1_macro", float("nan")),
             test_results.get("test_accuracy", float("nan")))

    # ---- full classification report + confusion matrix on test set ----
    raw_preds = trainer.predict(ds_test)
    logits_test = raw_preds.predictions
    if isinstance(logits_test, tuple):
        logits_test = logits_test[0]
    preds_test  = np.argmax(logits_test, axis=-1)
    labels_test = raw_preds.label_ids
    report = classification_report(labels_test, preds_test, output_dict=True, zero_division=0)
    cm     = confusion_matrix(labels_test, preds_test).tolist()

    # ---- save adapter ----
    log.info("Saving LoRA adapter to %s", adapter_dir)
    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)

    # ---- save metrics ----
    metrics = {
        "epoch_logs": metrics_cb.epoch_logs,
        "test": test_results,
        "test_classification_report": report,
        "test_confusion_matrix": cm,
        "elapsed_seconds": elapsed,
    }
    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2))
    log.info("Metrics saved to %s", metrics_path)

    # ---- save run config ----
    config = {
        "data_dir": str(data_dir),
        "output_dir": str(output_dir),
        "model_id": MODEL_ID,
        "num_labels": num_labels,
        "train_size": len(df_train),
        "val_size": len(df_val),
        "test_size": len(df_test),
        "lora_rank": LORA_RANK,
        "lora_alpha": LORA_ALPHA,
        "lora_dropout": LORA_DROPOUT,
        "lora_target_modules": LORA_TARGET_MODULES,
        "learning_rate": LEARNING_RATE,
        "num_epochs": NUM_EPOCHS,
        "batch_size": args.batch_size,
        "grad_accum": args.grad_accum,
        "max_length": MAX_LENGTH,
        "fp16": use_fp16,
    }
    (output_dir / "run_config.json").write_text(json.dumps(config, indent=2))

    log.info("Done. Adapter + metrics at: %s", output_dir)


if __name__ == "__main__":
    main()
