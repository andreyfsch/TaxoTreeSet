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
    set_seed,
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
SEED = 42
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


def tokenize_fn(batch, tokenizer, max_length: int = MAX_LENGTH):
    return tokenizer(
        batch["seq"],
        max_length=max_length,
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
        # Atomic: write beside the target, then rename. `write_text` truncates first,
        # so a machine that dies mid-write leaves a zero- or half-length file, and
        # every reader of it fails with JSONDecodeError at char 0. Three power cuts
        # in 24 h produced exactly that, and the corrupted file was the only record
        # of how far the run had got. rename() within a directory is atomic on ext4.
        tmp = self._progress_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(progress, indent=2))
        tmp.replace(self._progress_path)


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
    p.add_argument("--learning-rate", type=float, default=LEARNING_RATE,
                   help="Peak LR (default %(default)s; balanced 2-class binary "
                        "heads need a lower value, e.g. 2e-4, to avoid collapse)")
    p.add_argument("--fp16", action="store_true", default=True,
                   help="Use mixed-precision (default: on if CUDA available)")
    p.add_argument("--no-fp16", dest="fp16", action="store_false")
    p.add_argument("--resume-from-checkpoint", type=Path, default=None,
                   help="Resume training from a saved checkpoint directory")
    p.add_argument("--lora-rank", type=int, default=LORA_RANK,
                   help="LoRA rank. Lower means less capacity, which is the lever "
                        "for a head that fits its training data and not its val.")
    p.add_argument("--lora-dropout", type=float, default=LORA_DROPOUT,
                   help="LoRA dropout, raised alongside a lower rank to regularise.")
    p.add_argument("--max-length", type=int, default=MAX_LENGTH,
                   help="Tokeniser cap. At 128, a 1,100 bp window tokenises to ~133 "
                        "tokens and 51%% of long windows are TRUNCATED: the model "
                        "sees ~585 bp of them, against ~185 bp from the old 250 bp "
                        "windows. Raising this to 256 lets it see the whole window, "
                        "at roughly 4x the attention cost.")
    p.add_argument("--num-epochs", type=int, default=NUM_EPOCHS,
                   help="Training epochs. Also sets the cosine schedule's horizon, "
                        "so lowering it anneals the LR over the window where these "
                        "heads actually peak (val turns over at epoch ~0.33) rather "
                        "than holding a near-peak LR through 4.7 epochs of decay.")
    p.add_argument("--freeze-pooler", action="store_true",
                   help="Hold the pooler at init (still saved). It is 67%% of the "
                        "trainable parameters at rank 8 and 89%% at rank 2, so "
                        "without this the LoRA rank barely controls capacity.")
    p.add_argument("--seed", type=int, default=SEED,
                   help="Random seed; fixes the frozen pooler init so the adapter is reproducible")
    return p.parse_args()


def main():
    args = parse_args()
    # Seed BEFORE any model construction: the BertPooler is frozen at its random
    # init under LoRA, so without a fixed seed the (unsaved) pooler is different
    # on every run and the saved adapter cannot be reproduced.
    set_seed(args.seed)

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

    ds_train = build_hf_dataset(df_train).map(lambda b: tokenize_fn(b, tokenizer, args.max_length), batched=True, remove_columns=["seq"])
    ds_val   = build_hf_dataset(df_val).map(lambda b: tokenize_fn(b, tokenizer, args.max_length), batched=True, remove_columns=["seq"])
    ds_test  = build_hf_dataset(df_test).map(lambda b: tokenize_fn(b, tokenizer, args.max_length), batched=True, remove_columns=["seq"])

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
        r=args.lora_rank,
        lora_alpha=2 * args.lora_rank,   # alpha/r held at 2, as in the defaults
        lora_dropout=args.lora_dropout,
        target_modules=LORA_TARGET_MODULES,
        bias="none",
        # SEQ_CLS auto-saves "classifier"/"score" but NOT the pooler, which the
        # DNABERT-2 head uses (BertPooler.dense feeds the classifier). Saving it
        # explicitly makes it trainable AND persisted, so the adapter reloads
        # correctly standalone (e.g. in PhyloCascadeGLM inference).
        modules_to_save=["classifier", "score", "pooler"],
    )
    model = get_peft_model(base_model, lora_cfg)
    if args.freeze_pooler:
        # The pooler is 590,592 of the 887,042 trainable parameters -- 67% at rank
        # 8, 89% at rank 2. That is why the earlier "rank 8 -> 2" experiment moved
        # nothing: it cut total capacity by 25%, not 4x. Freezing it is the actual
        # capacity test, taking rank 8 from 887k to 296k.
        #
        # Frozen but still SAVED: it stays in modules_to_save so the adapter
        # reloads standalone. The pooler is randomly initialised (DNABERT-2 ships
        # no pooler weights), so dropping it from modules_to_save would have
        # PhyloCascadeGLM draw a *different* random pooler at inference and the
        # head would not reproduce its own eval.
        n_frozen = 0
        for name, p in model.named_parameters():
            if "pooler" in name and p.requires_grad:
                p.requires_grad = False
                n_frozen += p.numel()
        log.info("Pooler frozen: %d parameters held at init (still persisted).",
                 n_frozen)
    model.print_trainable_parameters()

    # ---- training args ----
    # Use ceil to match what the Trainer actually counts as steps per epoch.
    train_steps_per_epoch = math.ceil(len(df_train) / (args.batch_size * args.grad_accum))
    # Evaluate ~3x per epoch but SAVE only once. Transformers requires save_steps to
    # be a multiple of eval_steps, which is why these were previously equal -- but
    # they only have to be a multiple, not identical.
    #
    # The split matters because the two operations cost completely different things.
    # Eval reads a small val split and writes nothing. Saving writes a ~450 MB
    # checkpoint, and with the output on /mnt/f that crosses the 9p/drvfs boundary
    # into Windows. Sustained writes there are the open (NOT confirmed) hypothesis
    # for the three WSL lockups of 2026-08-07, whose signature is a task stuck in
    # uninterruptible sleep that even `wsl --shutdown` cannot reap.
    #
    # Cutting saves 3x costs at most one epoch of redone work on a crash and changes
    # no experiment: load_best_model_at_end only needs the best checkpoint, and the
    # eval trajectory -- the thing every finding here rests on -- is unaffected
    # because eval cadence is unchanged.
    eval_every = max(train_steps_per_epoch // 3, 50)
    save_every = eval_every * 3
    # Early stopping is measured per eval, so its patience must scale with eval
    # frequency. Evaluating ~3×/epoch with patience=2 would stop after only ~0.7
    # epoch without improvement; tolerate ~2 full epochs of no improvement instead.
    evals_per_epoch = max(1, round(train_steps_per_epoch / eval_every))
    early_stopping_patience = 2 * evals_per_epoch
    training_args = TrainingArguments(
        output_dir=str(output_dir / "checkpoints"),
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size // 2,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate,
        warmup_ratio=WARMUP_RATIO,
        weight_decay=WEIGHT_DECAY,
        lr_scheduler_type="cosine",
        eval_strategy="steps",
        eval_steps=eval_every,
        eval_accumulation_steps=20,  # offload eval preds to CPU; avoids VRAM OOM
        save_strategy="steps",
        save_steps=save_every,
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        label_smoothing_factor=0.1,
        fp16=use_fp16,
        # torch >=2.9 dispatches the default adamw_torch to the *foreach* Adam kernel,
        # which asserts grad_scale is None -- so resuming an fp16 run through GradScaler
        # dies with AssertionError in _multi_tensor_adam. The fused kernel accepts
        # grad_scale/found_inf natively. Same algorithm, different kernel.
        optim="adamw_torch_fused",
        logging_steps=25,
        report_to="none",
        # 1 + the best checkpoint, which load_best_model_at_end protects.
        save_total_limit=1,
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
    # Score the TRAIN split too. Without it a failed head cannot be triaged: a
    # model at chance on val may have fit the training data perfectly (overfitting
    # — fix with less capacity and more regularisation) or not at all (underfitting
    # — fix with a larger LR or more steps), and the two demand opposite changes.
    # Diagnosing this after the fact costs an adapter reload plus a full inference
    # pass per head, which does not scale to a 16k-head tree.
    #
    # Capped and evaluated with the best checkpoint already loaded, so it measures
    # the model that is actually saved.
    # RANDOM rows, not the first N: the parquet is written task by task, so the
    # head of the file is one class. Head 28344 has 2000 class-1 rows before the
    # first class-0 row, and f1_macro over a single-class slice is degenerate --
    # it read 0.498 against a val of 0.844, which is what exposed this.
    _n_train_eval = min(len(ds_train), 2000)
    _train_idx = np.random.default_rng(SEED).choice(
        len(ds_train), _n_train_eval, replace=False).tolist()
    train_results = trainer.evaluate(
        ds_train.select(_train_idx), metric_key_prefix="train_eval")
    log.info("Train f1_macro=%.4f (on %d rows)",
             train_results.get("train_eval_f1_macro", float("nan")), _n_train_eval)

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
        "train_eval": train_results,
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
        "learning_rate": args.learning_rate,
        "num_epochs": args.num_epochs,
        "batch_size": args.batch_size,
        "grad_accum": args.grad_accum,
        "max_length": args.max_length,
        "fp16": use_fp16,
    }
    (output_dir / "run_config.json").write_text(json.dumps(config, indent=2))

    log.info("Done. Adapter + metrics at: %s", output_dir)


if __name__ == "__main__":
    main()
