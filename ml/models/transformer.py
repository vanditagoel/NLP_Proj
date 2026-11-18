"""Hugging Face Indic & multilingual transformer models (MuRIL, mBERT, XLM-R, IndicBERT)."""
from __future__ import annotations

import gc
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

TRANSFORMER_MODEL_SPECS: dict[str, dict[str, str]] = {
    "muril": {
        "name": "MuRIL (google/muril-base-cased)",
        "hf_id": "google/muril-base-cased",
        "family": "Indic Transformer",
        "description": "Google Multilingual Representations for Indian Languages pre-trained on 17 Indic languages.",
    },
    "mbert": {
        "name": "mBERT (bert-base-multilingual-cased)",
        "hf_id": "bert-base-multilingual-cased",
        "family": "Multilingual Transformer",
        "description": "Multilingual BERT cased model covering 104 languages including Hindi, Bengali, and English.",
    },
    "xlmr": {
        "name": "XLM-RoBERTa (xlm-roberta-base)",
        "hf_id": "xlm-roberta-base",
        "family": "Multilingual Transformer",
        "description": "Cross-lingual RoBERTa trained on CommonCrawl across 100 languages including Hindi and Bengali.",
    },
    "indicbert": {
        "name": "IndicBERT (ai4bharat/indic-bert)",
        "hf_id": "ai4bharat/indic-bert",
        "family": "Indic Transformer",
        "description": "AI4Bharat ALBERT-based model trained specifically on 12 major Indian languages.",
    },
}


class StanceDataset(Dataset):
    def __init__(self, texts: list[str], labels: list[int] | None, tokenizer: Any, max_length: int = 96) -> None:
        self.encodings = tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt",
        )
        self.labels = labels

    def __len__(self) -> int:
        return int(self.encodings["input_ids"].shape[0])

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        item = {k: v[idx] for k, v in self.encodings.items()}
        if self.labels is not None:
            item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


def check_transformer_compatibility(model_key: str) -> dict[str, Any]:
    """Verify tokenizer/config loadability on Hugging Face Hub without fabricating results."""
    spec = TRANSFORMER_MODEL_SPECS[model_key]
    hf_id = spec["hf_id"]
    try:
        cfg = AutoConfig.from_pretrained(hf_id)
        tok = AutoTokenizer.from_pretrained(hf_id)
        sample_enc = tok("जलवायु परिवर्तन गंभीर है। জলবায়ু পরিবর্তন গুরুতর।", truncation=True)
        return {
            "model_key": model_key,
            "hf_id": hf_id,
            "compatible": True,
            "vocab_size": int(getattr(cfg, "vocab_size", len(tok))),
            "model_type": str(getattr(cfg, "model_type", "unknown")),
            "sample_token_count": len(sample_enc["input_ids"]),
        }
    except Exception as exc:
        return {
            "model_key": model_key,
            "hf_id": hf_id,
            "compatible": False,
            "error": str(exc),
        }


def train_transformer_model(
    model_key: str,
    train_texts: list[str],
    train_labels: list[int],
    val_texts: list[str],
    val_labels: list[int],
    label_names: list[str],
    output_dir: Path,
    epochs: int = 2,
    batch_size: int = 8,
    learning_rate: float = 3e-5,
    max_length: int = 96,
    weight_decay: float = 0.01,
    gradient_accumulation_steps: int = 2,
    early_stopping_patience: int = 2,
    use_fp16: bool = True,
    seed: int = 42,
) -> dict[str, Any]:
    """Fine-tune one transformer at a time with gradient accumulation, validation checkpointing, and early stopping."""
    torch.manual_seed(seed)
    torch.set_num_threads(2)
    spec = TRANSFORMER_MODEL_SPECS[model_key]
    hf_id = spec["hf_id"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fp16_active = bool(use_fp16 and device.type == "cuda")
    print(f"\n=== Training {spec['name']} on {device} (fp16={fp16_active}) ===")

    tokenizer = AutoTokenizer.from_pretrained(hf_id)
    id2label = {i: lbl for i, lbl in enumerate(label_names)}
    label2id = {lbl: i for i, lbl in enumerate(label_names)}
    model = AutoModelForSequenceClassification.from_pretrained(
        hf_id,
        num_labels=len(label_names),
        id2label=id2label,
        label2id=label2id,
    )
    model.to(device)

    train_ds = StanceDataset(train_texts, train_labels, tokenizer, max_length=max_length)
    val_ds = StanceDataset(val_texts, val_labels, tokenizer, max_length=max_length)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    total_steps = max(1, (len(train_loader) // max(1, gradient_accumulation_steps)) * epochs)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=max(1, int(0.1 * total_steps)),
        num_training_steps=total_steps,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=fp16_active)

    output_dir.mkdir(parents=True, exist_ok=True)
    best_val_f1 = -1.0
    no_improve = 0
    history: list[dict[str, float]] = []
    t_start = time.perf_counter()

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        optimizer.zero_grad()

        for step, batch in enumerate(train_loader):
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.amp.autocast("cuda", enabled=fp16_active):
                out = model(**batch)
                loss = out.loss / max(1, gradient_accumulation_steps)

            if fp16_active:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            total_loss += float(out.loss.item())
            if (step + 1) % max(1, gradient_accumulation_steps) == 0 or (step + 1) == len(train_loader):
                if fp16_active:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

        model.eval()
        val_preds: list[int] = []
        with torch.inference_mode():
            for batch in val_loader:
                inputs = {k: v.to(device) for k, v in batch.items() if k != "labels"}
                logits = model(**inputs).logits
                val_preds.extend(torch.argmax(logits, dim=-1).cpu().tolist())

        val_f1 = float(f1_score(val_labels, val_preds, average="macro", zero_division=0))
        avg_loss = total_loss / max(1, len(train_loader))
        print(f"Epoch {epoch}/{epochs} | train_loss={avg_loss:.4f} | val_macro_f1={val_f1:.4f}")
        history.append({"epoch": float(epoch), "train_loss": avg_loss, "val_macro_f1": val_f1})

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            no_improve = 0
            model.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)
        else:
            no_improve += 1
            if no_improve >= early_stopping_patience:
                print(f"Early stopping triggered at epoch {epoch}.")
                break

    elapsed = round(time.perf_counter() - t_start, 2)
    meta = {
        "model_key": model_key,
        "hf_id": hf_id,
        "label_names": label_names,
        "best_val_macro_f1": round(best_val_f1, 4),
        "training_seconds": elapsed,
        "epochs_completed": len(history),
        "history": history,
        "device_used": str(device),
    }
    with open(output_dir / "training_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    del model, optimizer, scheduler, train_loader, val_loader
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return meta


def predict_with_saved_transformer(
    model_dir: Path,
    texts: list[str],
    max_length: int = 96,
    batch_size: int = 16,
) -> tuple[list[str], np.ndarray]:
    if not (model_dir / "config.json").exists():
        raise FileNotFoundError(f"Model not trained at {model_dir}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.to(device)
    model.eval()

    ds = StanceDataset(texts, None, tokenizer, max_length=max_length)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)

    probs_all: list[np.ndarray] = []
    with torch.inference_mode():
        for batch in loader:
            inputs = {k: v.to(device) for k, v in batch.items()}
            logits = model(**inputs).logits
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
            probs_all.append(probs)

    probs_mat = np.vstack(probs_all) if probs_all else np.zeros((0, len(model.config.id2label)))
    id2label = model.config.id2label
    pred_labels = [str(id2label[int(idx)]) for idx in np.argmax(probs_mat, axis=1)]

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return pred_labels, probs_mat
