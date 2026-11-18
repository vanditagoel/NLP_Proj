"""FastAPI endpoints for Indic Climate Stance Detection."""
from __future__ import annotations

import io
from typing import Any

import pandas as pd
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from backend.app.services.ml_service import (
    get_dataset_samples,
    get_dataset_stats,
    get_evaluation_summary,
    get_project_overview,
    get_training_status,
    list_models,
    predict_batch_texts,
    predict_compare,
    predict_single,
    start_background_training,
)

router = APIRouter(prefix="/api", tags=["indic-climate-stance"])


class PredictRequest(BaseModel):
    text: str = Field(..., min_length=1)
    language: str = "Auto"
    model_key: str = "tfidf_logreg"


class CompareRequest(BaseModel):
    text: str = Field(..., min_length=1)
    language: str = "Auto"
    model_keys: list[str] | None = None


class TrainRequest(BaseModel):
    model_keys: list[str] = Field(default_factory=lambda: ["tfidf_logreg", "tfidf_svm", "tfidf_nb"])
    epochs: int | None = None


@router.get("/health")
def health() -> dict[str, Any]:
    return {"status": "ok"}


@router.get("/project")
def project_overview() -> dict[str, Any]:
    return get_project_overview()


@router.get("/dataset/stats")
def dataset_stats() -> dict[str, Any]:
    stats = get_dataset_stats()
    if stats is None:
        raise HTTPException(
            status_code=404,
            detail="Dataset stats not generated yet. Run dataset preparation and analysis first.",
        )
    return stats


@router.get("/dataset/samples")
def dataset_samples(
    language: str = "All",
    label: str = "All",
    split: str = "All",
    limit: int = 25,
) -> dict[str, Any]:
    rows = get_dataset_samples(language=language, label=label, split=split, limit=limit)
    return {"count": len(rows), "samples": rows}


@router.get("/models")
def models() -> dict[str, Any]:
    return {"models": list_models()}


@router.get("/evaluation")
def evaluation() -> dict[str, Any]:
    return get_evaluation_summary()


@router.post("/train")
def trigger_training(req: TrainRequest) -> dict[str, Any]:
    return start_background_training(req.model_keys, epochs=req.epochs)


@router.get("/train/status")
def training_status() -> dict[str, Any]:
    return get_training_status()


@router.post("/predict")
def predict(req: PredictRequest) -> dict[str, Any]:
    try:
        return predict_single(text=req.text, language=req.language, model_key=req.model_key)
    except FileNotFoundError:
        raise HTTPException(status_code=400, detail="Model not trained")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@router.post("/predict/compare")
def compare(req: CompareRequest) -> dict[str, Any]:
    try:
        return predict_compare(text=req.text, language=req.language, model_keys=req.model_keys)
    except FileNotFoundError:
        raise HTTPException(status_code=400, detail="Model not trained")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@router.post("/predict/batch")
async def batch_predict(
    file: UploadFile = File(...),
    text_column: str = Form("text"),
    model_keys: str = Form("tfidf_logreg"),
    return_csv: bool = Form(False),
) -> Any:
    raw_bytes = await file.read()
    if not raw_bytes:
        raise HTTPException(status_code=400, detail="Uploaded CSV file is empty.")

    try:
        df = pd.read_csv(io.BytesIO(raw_bytes))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid CSV file: {exc}")

    if text_column not in df.columns:
        raise HTTPException(
            status_code=400,
            detail=f"Column '{text_column}' not found. Available columns: {list(df.columns)}",
        )

    selected_keys = [k.strip() for k in model_keys.split(",") if k.strip()]
    if not selected_keys:
        raise HTTPException(status_code=400, detail="Select at least one model.")

    df = df.head(500).copy()
    texts = df[text_column].fillna("").astype(str).tolist()
    primary_key = selected_keys[0]

    try:
        primary_preds = predict_batch_texts(texts, primary_key)
    except FileNotFoundError:
        raise HTTPException(status_code=400, detail="Model not trained")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    out_df = df.copy()
    out_df["detected_language"] = [r["detected_language"] for r in primary_preds]
    out_df["prediction"] = [r["stance"] for r in primary_preds]
    out_df["confidence"] = [r["confidence"] for r in primary_preds]

    for extra_key in selected_keys[1:]:
        try:
            extra_preds = predict_batch_texts(texts, extra_key)
            out_df[f"pred_{extra_key}"] = [r["stance"] for r in extra_preds]
        except FileNotFoundError:
            raise HTTPException(status_code=400, detail="Model not trained")

    if return_csv:
        buf = io.StringIO()
        out_df.to_csv(buf, index=False, encoding="utf-8-sig")
        buf.seek(0)
        return StreamingResponse(
            iter([buf.getvalue()]),
            media_type="text/csv; charset=utf-8",
            headers={"Content-Disposition": 'attachment; filename="climate_stance_predictions.csv"'},
        )

    counts = {lbl: int((out_df["prediction"] == lbl).sum()) for lbl in ("Favour", "Against", "None")}
    return {
        "model_key": primary_key,
        "selected_models": selected_keys,
        "total_rows": int(len(out_df)),
        "counts": counts,
        "columns": list(out_df.columns),
        "rows": out_df.fillna("").to_dict(orient="records"),
    }
