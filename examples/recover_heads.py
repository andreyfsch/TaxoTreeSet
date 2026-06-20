"""Recover LoRA classifier heads whose saved adapter cannot be reloaded.

Heads trained by an older ``finetune_head.py`` saved only the LoRA weights and the
classifier, but NOT the ``BertPooler`` — which under LoRA was frozen at a random
initialization and feeds the classifier. With no fixed seed that pooler is
irreproducible, so reloading such an adapter standalone yields ~chance accuracy.

The expensive part (the LoRA encoder) is intact in the adapter, so the fix does
NOT require retraining: this script reuses the frozen LoRA encoder, refits a fresh
pooler+classifier on its ``[CLS]`` features, and saves a COMPLETE adapter
(``modules_to_save`` including the pooler, current PEFT key format) that reloads
correctly with ``PeftModel.from_pretrained``. Each recovered head is validated by
a fresh reload + real forward against the original ``metrics.json``.

Non-destructive: the recovered adapter is written to a sibling directory
(``adapter_recovered/`` by default); the original is left untouched.

Example:
    python examples/recover_heads.py \\
        --finetune-dir /mnt/f/taxotreeset_viruses/finetune \\
        --manifest /mnt/f/taxotreeset_viruses/datasets/manifest_viruses.json
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time

import numpy as np
import pyarrow.parquet as pq
import torch
import torch.nn as nn
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from safetensors.torch import load_file
from sklearn.metrics import accuracy_score, f1_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer, set_seed

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S"
)
log = logging.getLogger(__name__)

CLASSIFIER_WEIGHT_KEY = "base_model.model.classifier.weight"


def remap_lora_keys(state_dict: dict) -> dict:
    """Remap old-format LoRA keys to the current PEFT layout.

    Older PEFT saved ``...lora_A.weight``; current PEFT expects
    ``...lora_A.default.weight``. Only LoRA tensors are returned (the
    classifier/pooler are refit, not loaded).

    Args:
        state_dict: Tensors loaded from the saved ``adapter_model.safetensors``.

    Returns:
        A state dict with LoRA keys in the current ``.default`` layout.
    """
    out = {}
    for key, value in state_dict.items():
        if key.endswith(".lora_A.weight") or key.endswith(".lora_B.weight"):
            out[key[: -len(".weight")] + ".default.weight"] = value
    return out


def active_module(module: nn.Module) -> nn.Module:
    """Return the trainable submodule of a PEFT ``ModulesToSaveWrapper``.

    Args:
        module: A module that may be wrapped by PEFT's ``modules_to_save``.

    Returns:
        The active ``modules_to_save["default"]`` submodule, or ``module`` itself
        if it is not wrapped.
    """
    return module.modules_to_save["default"] if hasattr(module, "modules_to_save") else module


class _Head(nn.Module):
    """Standalone pooler+classifier head trained on cached ``[CLS]`` features.

    Mirrors DNABERT-2's classification head: ``classifier(dropout(tanh(dense(x))))``,
    where ``dense``+``tanh`` is the ``BertPooler`` and ``classifier`` the final layer.
    """

    def __init__(self, num_labels: int, hidden: int = 768) -> None:
        super().__init__()
        self.dense = nn.Linear(hidden, hidden)
        self.act = nn.Tanh()
        self.drop = nn.Dropout(0.1)
        self.cls = nn.Linear(hidden, num_labels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.cls(self.drop(self.act(self.dense(x))))


@torch.no_grad()
def cache_cls_features(
    bert, tokenizer, parquet_path: str, n: int, device: str
) -> tuple[np.ndarray, np.ndarray]:
    """Run the frozen encoder over a split and cache the ``[CLS]`` hidden state.

    Args:
        bert: The LoRA-adapted ``BertModel`` (frozen encoder).
        tokenizer: Matching tokenizer.
        parquet_path: Path to a ``{seq, class_idx}`` parquet split.
        n: Max number of evenly-sampled rows to cache.
        device: Torch device string.

    Returns:
        A tuple ``(features, labels)`` with shapes ``[m, 768]`` and ``[m]``.
    """
    table = pq.read_table(parquet_path, columns=["seq", "class_idx"])
    seqs = table.column("seq").to_pylist()
    labels = np.array(table.column("class_idx").to_pylist())
    if n and n < len(seqs):
        idx = np.linspace(0, len(seqs) - 1, n).astype(int)
        seqs = [seqs[i] for i in idx]
        labels = labels[idx]
    feats = np.zeros((len(seqs), 768), dtype=np.float32)
    for i in range(0, len(seqs), 32):
        enc = tokenizer(
            seqs[i : i + 32], return_tensors="pt", max_length=128, truncation=True, padding=True
        ).to(device)
        out = bert(enc["input_ids"], attention_mask=enc.get("attention_mask"))
        feats[i : i + 32] = out[0][:, 0].float().cpu().numpy()
    return feats, labels


def recover_head(taxid: str, args: argparse.Namespace) -> dict:
    """Recover one head: refit the classification head, save, reload, validate.

    Args:
        taxid: Head identifier (directory name under ``--finetune-dir``).
        args: Parsed CLI arguments.

    Returns:
        A result dict with recovered/target metrics for the head.
    """
    manifest = json.load(open(args.manifest))
    data_dir = manifest[taxid]["directory_path"]
    adapter_dir = f"{args.finetune_dir}/{taxid}/adapter"
    old = load_file(f"{adapter_dir}/adapter_model.safetensors")
    num_labels = int(old[CLASSIFIER_WEIGHT_KEY].shape[0])
    target = json.load(open(f"{args.finetune_dir}/{taxid}/metrics.json")).get("test", {})

    set_seed(args.seed)
    base = AutoModelForSequenceClassification.from_pretrained(
        args.backbone, num_labels=num_labels, trust_remote_code=True
    )
    cfg = LoraConfig(
        task_type=TaskType.SEQ_CLS, r=8, lora_alpha=16, lora_dropout=0.1, target_modules=["Wqkv"],
        bias="none", modules_to_save=["classifier", "score", "pooler"],
    )
    model = get_peft_model(base, cfg)
    model.load_state_dict(remap_lora_keys(old), strict=False)
    bert = model.base_model.model.bert.to(args.device).eval()

    x_train, y_train = cache_cls_features(
        bert, args.tokenizer, f"{data_dir}/train.parquet", args.train_n, args.device
    )

    set_seed(args.seed)
    head = _Head(num_labels).to(args.device)
    opt = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=0.01)
    loss_fn = nn.CrossEntropyLoss()
    xt = torch.tensor(x_train, device=args.device)
    yt = torch.tensor(y_train, device=args.device)
    for _ in range(args.epochs):
        head.train()
        perm = torch.randperm(len(xt), device=args.device)
        for i in range(0, len(perm), 256):
            batch = perm[i : i + 256]
            opt.zero_grad()
            loss_fn(head(xt[batch]), yt[batch]).backward()
            opt.step()
    head.eval()

    with torch.no_grad():
        pooler = active_module(model.base_model.model.bert.pooler)
        classifier = active_module(model.base_model.model.classifier)
        pooler.dense.weight.copy_(head.dense.weight)
        pooler.dense.bias.copy_(head.dense.bias)
        classifier.weight.copy_(head.cls.weight)
        classifier.bias.copy_(head.cls.bias)
    out_dir = f"{args.finetune_dir}/{taxid}/{args.out_name}"
    model.save_pretrained(out_dir)
    del model, base, bert
    torch.cuda.empty_cache()

    # Validate by reloading FRESH (the production path) under a different seed.
    set_seed(args.seed + 1)
    base_reload = AutoModelForSequenceClassification.from_pretrained(
        args.backbone, num_labels=num_labels, trust_remote_code=True
    )
    reloaded = PeftModel.from_pretrained(base_reload, out_dir).to(args.device).eval()
    table = pq.read_table(f"{data_dir}/test.parquet", columns=["seq", "class_idx"])
    seqs = table.column("seq").to_pylist()
    labels = np.array(table.column("class_idx").to_pylist())
    if args.eval_n < len(seqs):
        idx = np.linspace(0, len(seqs) - 1, args.eval_n).astype(int)
        seqs = [seqs[i] for i in idx]
        labels = labels[idx]
    preds = []
    with torch.no_grad():
        for i in range(0, len(seqs), 32):
            enc = args.tokenizer(
                seqs[i : i + 32], return_tensors="pt", max_length=128, truncation=True, padding=True
            ).to(args.device)
            logits = reloaded(**enc).logits
            logits = logits[0] if isinstance(logits, tuple) else logits
            preds.append(logits.argmax(1).cpu().numpy())
    pred = np.concatenate(preds)
    del reloaded, base_reload
    torch.cuda.empty_cache()
    return {
        "taxid": taxid, "num_labels": num_labels, "n_eval": int(len(labels)),
        "target_acc": target.get("test_accuracy"), "target_f1": target.get("test_f1_macro"),
        "recovered_acc": float(accuracy_score(labels, pred)),
        "recovered_f1": float(f1_score(labels, pred, average="macro")),
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--finetune-dir", required=True,
                   help="Directory with one <taxid>/adapter subdir per head")
    p.add_argument("--manifest", required=True,
                   help="manifest_*.json mapping taxid -> directory_path / labels")
    p.add_argument("--backbone", default="zhihan1996/DNABERT-2-117M", help="Base model id")
    p.add_argument("--heads", nargs="*", default=None,
                   help="Specific taxids to recover (default: all discovered)")
    p.add_argument("--out-name", default="adapter_recovered",
                   help="Sibling dir name for the recovered adapter")
    p.add_argument("--train-n", type=int, default=12000,
                   help="Max train windows cached for the head refit")
    p.add_argument("--eval-n", type=int, default=6000, help="Max test windows for validation")
    p.add_argument("--epochs", type=int, default=30, help="Head-refit epochs on cached features")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--report", default=None,
                   help="JSON report path (default: <finetune-dir>/recovery_report.json)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.tokenizer = AutoTokenizer.from_pretrained(args.backbone, trust_remote_code=True)
    manifest = json.load(open(args.manifest))
    if args.heads:
        heads = args.heads
    else:
        heads = []
        for d in sorted(os.listdir(args.finetune_dir)):
            adapter = f"{args.finetune_dir}/{d}/adapter/adapter_model.safetensors"
            if d in manifest and os.path.exists(adapter):
                heads.append(d)
    report_path = args.report or f"{args.finetune_dir}/recovery_report.json"
    log.info("Recovering %d heads -> %s", len(heads), report_path)

    results = []
    for k, taxid in enumerate(heads, 1):
        start = time.time()
        try:
            res = recover_head(taxid, args)
            res["status"] = "ok"
            if res["target_acc"] is None:
                delta = ""
            else:
                delta = f", d={res['recovered_acc'] - res['target_acc']:+.3f}"
            log.info(
                "[%2d/%d] %s N=%d acc %.3f (tgt %s%s) f1 %.3f [%.0fs]",
                k, len(heads), taxid, res["num_labels"], res["recovered_acc"],
                "n/a" if res["target_acc"] is None else f"{res['target_acc']:.3f}", delta,
                res["recovered_f1"], time.time() - start,
            )
        except Exception as exc:  # noqa: BLE001 - report and continue the batch
            res = {"taxid": taxid, "status": "ERROR", "error": str(exc)}
            log.exception("[%2d/%d] %s ERROR", k, len(heads), taxid)
            torch.cuda.empty_cache()
        results.append(res)
        json.dump(results, open(report_path, "w"), indent=2)

    ok = [r for r in results if r.get("status") == "ok" and r.get("target_acc") is not None]
    if ok:
        mean_delta = float(np.mean([abs(r["recovered_acc"] - r["target_acc"]) for r in ok]))
        log.info("DONE: %d/%d ok; mean abs acc delta = %.4f", len(ok), len(heads), mean_delta)


if __name__ == "__main__":
    main()
