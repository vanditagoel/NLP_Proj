"""ML inference, model registry, evaluation, dataset inspection, and background training service."""
from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from ml.evaluation.evaluate import evaluate_all_trained_models
from ml.models.classical import CLASSICAL_MODEL_SPECS
from ml.models.transformer import TRANSFORMER_MODEL_SPECS, predict_with_saved_transformer
from ml.preprocessing.cleaner import clean_text, detect_script_language
from ml.training.train_classical import train_classical_models
from ml.training.train_transformer import run_transformer_training
from ml.utils import (
    CHARTS_DIR,
    LABELS,
    METRICS_DIR,
    MODELS_DIR,
    PROCESSED_DIR,
    TRAINED_CLASSICAL_DIR,
    load_config,
)

_CLASSICAL_CACHE: dict[str, Any] = {}
_TRAIN_LOCK = threading.Lock()
_TRAIN_STATUS: dict[str, Any] = {
    "running": False,
    "current_model": None,
    "last_completed": [],
    "last_error": None,
    "started_at": None,
    "finished_at": None,
}


def get_project_overview() -> dict[str, Any]:
    cfg = load_config()
    eval_data = get_evaluation_summary()
    stats = get_dataset_stats()
    return {
        "project_title": cfg["project_title"],
        "course_code": cfg["course_code"],
        "target_claim": cfg["target_claim"],
        "labels": cfg["labels"],
        "languages": cfg["languages"],
        "dataset_source": cfg["dataset_source"],
        "best_model_key": eval_data.get("best_model_key"),
        "best_model_name": eval_data.get("best_model_name"),
        "total_samples": stats.get("total_samples", 0) if stats else 0,
        "trained_models_count": sum(1 for m in list_models() if m["trained"]),
    }


def list_models() -> list[dict[str, Any]]:
    eval_summary = get_evaluation_summary()
    metrics_by_key = {m["model_key"]: m for m in eval_summary.get("models", [])}

    models: list[dict[str, Any]] = []
    for key, spec in CLASSICAL_MODEL_SPECS.items():
        art_path = TRAINED_CLASSICAL_DIR / f"{key}.joblib"
        trained = art_path.exists()
        m_metrics = metrics_by_key.get(key)
        models.append(
            {
                "model_key": key,
                "name": spec["name"],
                "family": spec["family"],
                "kind": "classical",
                "description": spec["description"],
                "trained": trained,
                "status": "Trained" if trained else "Model not trained",
                "metrics": m_metrics,
            }
        )

    for key, spec in TRANSFORMER_MODEL_SPECS.items():
        model_dir = MODELS_DIR / key
        trained = (model_dir / "config.json").exists()
        m_metrics = metrics_by_key.get(key)
        models.append(
            {
                "model_key": key,
                "name": spec["name"],
                "family": spec["family"],
                "kind": "transformer",
                "hf_id": spec["hf_id"],
                "description": spec["description"],
                "trained": trained,
                "status": "Trained" if trained else "Model not trained",
                "metrics": m_metrics,
            }
        )
    return models


def get_evaluation_summary() -> dict[str, Any]:
    summary_path = METRICS_DIR / "evaluation_summary.json"
    if not summary_path.exists():
        return {"best_model_key": None, "best_model_name": None, "models": []}
    with open(summary_path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_dataset_stats() -> dict[str, Any] | None:
    stats_path = METRICS_DIR / "dataset_stats.json"
    prepare_meta_path = METRICS_DIR / "dataset_prepare_meta.json"
    pol_inspect_path = METRICS_DIR / "pol_nli_inspection.json"
    pol_aux_path = METRICS_DIR / "pol_nli_prepare_meta.json"

    if not stats_path.exists():
        return None
    with open(stats_path, "r", encoding="utf-8") as f:
        stats = json.load(f)

    if prepare_meta_path.exists():
        with open(prepare_meta_path, "r", encoding="utf-8") as f:
            stats["prepare_meta"] = json.load(f)
    if pol_inspect_path.exists():
        with open(pol_inspect_path, "r", encoding="utf-8") as f:
            stats["pol_nli_inspection"] = json.load(f)
    if pol_aux_path.exists():
        with open(pol_aux_path, "r", encoding="utf-8") as f:
            stats["pol_nli_auxiliary"] = json.load(f)

    available_charts = sorted([p.name for p in CHARTS_DIR.glob("*.png")]) if CHARTS_DIR.exists() else []
    stats["charts"] = available_charts
    return stats


def get_dataset_samples(
    language: str | None = None,
    label: str | None = None,
    split: str | None = None,
    limit: int = 25,
) -> list[dict[str, Any]]:
    csv_path = PROCESSED_DIR / "climate_stance_dataset.csv"
    if not csv_path.exists():
        return []
    df = pd.read_csv(csv_path, keep_default_na=False)
    if language and language != "All":
        df = df[df["language"] == language]
    if label and label != "All":
        df = df[df["label"] == label]
    if split and split != "All":
        df = df[df["split"] == split]
    cols = [
        c
        for c in ("id", "text", "original_text", "language", "label", "split", "translation_status", "original_source")
        if c in df.columns
    ]
    return df[cols].head(limit).fillna("").to_dict(orient="records")


def _load_classical_model(model_key: str) -> Any:
    art_path = TRAINED_CLASSICAL_DIR / f"{model_key}.joblib"
    if not art_path.exists():
        raise FileNotFoundError("Model not trained")
    if model_key not in _CLASSICAL_CACHE:
        _CLASSICAL_CACHE[model_key] = joblib.load(art_path)
    return _CLASSICAL_CACHE[model_key]


def _format_prob_dict(classes: list[str], probs_row: np.ndarray) -> dict[str, float]:
    out = {lbl: 0.0 for lbl in LABELS}
    for cls_name, val in zip(classes, probs_row):
        if cls_name in out:
            out[cls_name] = round(float(val), 4)
    return out


def predict_single(text: str, language: str = "Auto", model_key: str = "tfidf_logreg") -> dict[str, Any]:
    cleaned = clean_text(text)
    if not cleaned:
        raise ValueError("Input text cannot be empty.")

    det_lang, det_code = detect_script_language(cleaned)
    resolved_lang = det_lang if language in ("Auto", "", None) else language

    t0 = time.perf_counter()
    if model_key in CLASSICAL_MODEL_SPECS:
        pipe = _load_classical_model(model_key)
        pred = str(pipe.predict([cleaned])[0])
        probs_row = pipe.predict_proba([cleaned])[0]
        classes = [str(c) for c in pipe.classes_]
        prob_map = _format_prob_dict(classes, probs_row)
        model_name = CLASSICAL_MODEL_SPECS[model_key]["name"]
    elif model_key in TRANSFORMER_MODEL_SPECS:
        model_dir = MODELS_DIR / model_key
        if not (model_dir / "config.json").exists():
            raise FileNotFoundError("Model not trained")
        preds, probs_mat = predict_with_saved_transformer(model_dir, [cleaned])
        pred = preds[0]
        from transformers import AutoConfig

        cfg = AutoConfig.from_pretrained(model_dir)
        classes = [str(cfg.id2label[i]) for i in range(len(cfg.id2label))]
        prob_map = _format_prob_dict(classes, probs_mat[0])
        model_name = TRANSFORMER_MODEL_SPECS[model_key]["name"]
    else:
        raise ValueError(f"Unknown model key '{model_key}'.")

    elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 2)
    confidence = round(float(max(prob_map.values()) if prob_map else 0.0), 4)

    return {
        "text": text,
        "cleaned_text": cleaned,
        "detected_language": det_lang,
        "detected_language_code": det_code,
        "selected_language": resolved_lang,
        "model_key": model_key,
        "model_name": model_name,
        "stance": pred,
        "confidence": confidence,
        "probabilities": prob_map,
        "processing_time_ms": elapsed_ms,
    }


def predict_batch_texts(texts: list[str], model_key: str) -> list[dict[str, Any]]:
    cleaned_list = [clean_text(t) for t in texts]
    if model_key in CLASSICAL_MODEL_SPECS:
        pipe = _load_classical_model(model_key)
        t0 = time.perf_counter()
        preds = [str(p) for p in pipe.predict(cleaned_list)]
        probs = pipe.predict_proba(cleaned_list)
        classes = [str(c) for c in pipe.classes_]
        per_row_ms = round(((time.perf_counter() - t0) * 1000.0) / max(1, len(texts)), 2)
    elif model_key in TRANSFORMER_MODEL_SPECS:
        model_dir = MODELS_DIR / model_key
        if not (model_dir / "config.json").exists():
            raise FileNotFoundError("Model not trained")
        t0 = time.perf_counter()
        preds, probs = predict_with_saved_transformer(model_dir, cleaned_list)
        per_row_ms = round(((time.perf_counter() - t0) * 1000.0) / max(1, len(texts)), 2)
        from transformers import AutoConfig

        cfg = AutoConfig.from_pretrained(model_dir)
        classes = [str(cfg.id2label[i]) for i in range(len(cfg.id2label))]
    else:
        raise ValueError(f"Unknown model key '{model_key}'.")

    rows: list[dict[str, Any]] = []
    for raw, cln, pred, p_row in zip(texts, cleaned_list, preds, probs):
        det_lang, _ = detect_script_language(cln)
        prob_map = _format_prob_dict(classes, p_row)
        rows.append(
            {
                "text": raw,
                "detected_language": det_lang,
                "stance": pred,
                "confidence": round(float(max(prob_map.values())), 4),
                "probabilities": prob_map,
                "processing_time_ms": per_row_ms,
            }
        )
    return rows


def predict_compare(
    text: str,
    language: str = "Auto",
    model_keys: list[str] | None = None,
) -> dict[str, Any]:
    trained_keys = [m["model_key"] for m in list_models() if m["trained"]]
    target_keys = model_keys if model_keys else trained_keys
    if not target_keys:
        raise FileNotFoundError("Model not trained")

    predictions = []
    errors = []
    for key in target_keys:
        try:
            predictions.append(predict_single(text=text, language=language, model_key=key))
        except FileNotFoundError:
            errors.append({"model_key": key, "error": "Model not trained"})

    if not predictions:
        raise FileNotFoundError("Model not trained")

    stances = [p["stance"] for p in predictions]
    unique_stances = set(stances)
    if len(unique_stances) == 1:
        agreement = {
            "status": "unanimous",
            "consensus": stances[0],
            "summary": f"All {len(predictions)} models agree on '{stances[0]}'.",
        }
    else:
        counts = {s: stances.count(s) for s in unique_stances}
        top_s, top_c = max(counts.items(), key=lambda kv: kv[1])
        agreement = {
            "status": "disagreement",
            "consensus": top_s if top_c > len(predictions) / 2 else None,
            "summary": f"Models disagree ({', '.join(f'{k}: {v}' for k, v in counts.items())}).",
        }

    return {
        "text": text,
        "detected_language": predictions[0]["detected_language"],
        "predictions": predictions,
        "untrained": errors,
        "agreement": agreement,
    }


def start_background_training(model_keys: list[str], epochs: int | None = None) -> dict[str, Any]:
    with _TRAIN_LOCK:
        if _TRAIN_STATUS["running"]:
            return {"started": False, "reason": "Training already in progress", "status": dict(_TRAIN_STATUS)}

        _TRAIN_STATUS.update(
            {
                "running": True,
                "current_model": ", ".join(model_keys),
                "last_completed": [],
                "last_error": None,
                "started_at": time.time(),
                "finished_at": None,
            }
        )

    def _worker() -> None:
        try:
            classical_keys = [k for k in model_keys if k in CLASSICAL_MODEL_SPECS]
            transformer_keys = [k for k in model_keys if k in TRANSFORMER_MODEL_SPECS]
            if classical_keys:
                train_classical_models(mode="climate_stance", selected_keys=classical_keys)
                for ck in classical_keys:
                    _CLASSICAL_CACHE.pop(ck, None)
                    _TRAIN_STATUS["last_completed"].append(ck)

            for tk in transformer_keys:
                _TRAIN_STATUS["current_model"] = tk
                run_transformer_training(model_key=tk, epochs=epochs)
                _TRAIN_STATUS["last_completed"].append(tk)

            evaluate_all_trained_models()
        except Exception as exc:
            _TRAIN_STATUS["last_error"] = str(exc)
        finally:
            _TRAIN_STATUS["running"] = False
            _TRAIN_STATUS["current_model"] = None
            _TRAIN_STATUS["finished_at"] = time.time()

    threading.Thread(target=_worker, daemon=True).start()
    return {"started": True, "status": dict(_TRAIN_STATUS)}


def get_training_status() -> dict[str, Any]:
    return dict(_TRAIN_STATUS)
