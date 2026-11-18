"""Comprehensive unit and integration tests for dataset, preprocessing, splitting, leakage prevention, translation, models, evaluation, and FastAPI endpoints."""
from __future__ import annotations

import io
from pathlib import Path

import pandas as pd
from fastapi.testclient import TestClient

from backend.app.main import app
from ml.dataset.prepare import group_stratified_split
from ml.dataset.seed_climate_data import build_seed_multilingual_dataframe
from ml.evaluation.metrics import compute_classification_metrics
from ml.models.classical import build_classical_pipeline
from ml.preprocessing.cleaner import clean_text, detect_script_language
from ml.translation.translate import (
    assign_translation_targets,
    run_translation,
    translate_with_retry,
)


def test_text_cleaning_preserves_indic_scripts():
    raw_hi = "https://example.com @user जलवायु परिवर्तन   एक गंभीर संकट है!"
    cleaned_hi = clean_text(raw_hi)
    assert "https://" not in cleaned_hi
    assert "@user" not in cleaned_hi
    assert "जलवायु परिवर्तन एक गंभीर संकट है!" == cleaned_hi

    raw_bn = "জলবায়ু পরিবর্তন   একটি গুরুতর হুমকি।"
    assert clean_text(raw_bn) == "জলবায়ু পরিবর্তন একটি গুরুতর হুমকি।"


def test_script_language_detection():
    assert detect_script_language("जलवायु परिवर्तन गंभीर संकट है")[0] == "Hindi"
    assert detect_script_language("জলবায়ু পরিবর্তন একটি গুরুতর হুমকি")[0] == "Bengali"
    assert detect_script_language("Climate change is a serious concern.")[0] == "English"


def test_group_stratified_split_prevents_leakage():
    df = build_seed_multilingual_dataframe()
    split_df = group_stratified_split(
        df, group_col="group_id", label_col="label", train_ratio=0.7, val_ratio=0.15, seed=42
    )
    train_groups = set(split_df[split_df["split"] == "Train"]["group_id"])
    val_groups = set(split_df[split_df["split"] == "Validation"]["group_id"])
    test_groups = set(split_df[split_df["split"] == "Test"]["group_id"])

    assert len(train_groups) > 0
    assert len(val_groups) > 0
    assert len(test_groups) > 0
    assert train_groups.isdisjoint(val_groups)
    assert train_groups.isdisjoint(test_groups)
    assert val_groups.isdisjoint(test_groups)


def test_translation_allocation_and_mock_pipeline(tmp_path: Path):
    src_df = pd.DataFrame(
        [
            {"id": "1", "group_id": "g1", "original_text": "Climate change is serious.", "label": "Favour", "original_source": "test"},
            {"id": "2", "group_id": "g2", "original_text": "Global warming is hype.", "label": "Against", "original_source": "test"},
            {"id": "3", "group_id": "g3", "original_text": "Rain forecast tomorrow.", "label": "None", "original_source": "test"},
            {"id": "4", "group_id": "g4", "original_text": "Solar energy helps.", "label": "Favour", "original_source": "test"},
        ]
    )
    allocated = assign_translation_targets(src_df, seed=42)
    counts = allocated["assigned_target_lang"].value_counts().to_dict()
    assert counts.get("hi", 0) == 2
    assert counts.get("bn", 0) == 2

    class MockTranslator:
        def __init__(self, target: str) -> None:
            self.target = target

        def translate(self, text: str) -> str:
            if self.target == "hi":
                return "जलवायु परिवर्तन गंभीर है।"
            return "জলবায়ু পরিবর্তন গুরুতর।"

    in_csv = tmp_path / "src.csv"
    src_df.to_csv(in_csv, index=False)

    hi_txt, err = translate_with_retry("Climate change is serious.", "hi", translator=MockTranslator("hi"))
    assert err is None and hi_txt is not None

    out_df = run_translation(
        input_csv=in_csv,
        target="both",
        batch_size=2,
        seed=42,
        translator_factory=lambda t: MockTranslator(t),
        output_csv=tmp_path / "translated.csv",
        cache_path=tmp_path / "cache.json",
    )
    assert len(out_df) == 8
    assert set(out_df["language_code"].unique()) == {"en", "hi", "bn"}


def test_classical_model_training_prediction_and_evaluation():
    df = build_seed_multilingual_dataframe()
    split_df = group_stratified_split(df, seed=42)
    train_df = split_df[split_df["split"] == "Train"]
    test_df = split_df[split_df["split"] == "Test"]

    pipe = build_classical_pipeline("tfidf_logreg", seed=42)
    pipe.fit(train_df["text"].tolist(), train_df["label"].tolist())

    preds = pipe.predict(test_df["text"].tolist()).tolist()
    probs = pipe.predict_proba(test_df["text"].tolist())
    assert len(preds) == len(test_df)
    assert probs.shape == (len(test_df), 3)

    metrics = compute_classification_metrics(
        y_true=test_df["label"].tolist(),
        y_pred=preds,
        labels=["Favour", "Against", "None"],
        languages=test_df["language"].tolist(),
        model_key="tfidf_logreg",
        model_name="TF-IDF + Logistic Regression",
    )
    assert 0.0 <= metrics["macro_f1"] <= 1.0
    assert len(metrics["confusion_matrix"]) == 3
    assert "English" in metrics["language_metrics"]


def test_fastapi_endpoints_and_csv_batch():
    client = TestClient(app)

    r_health = client.get("/api/health")
    assert r_health.status_code == 200
    assert r_health.json()["status"] == "ok"

    r_proj = client.get("/api/project")
    assert r_proj.status_code == 200
    assert "Indic" in r_proj.json()["project_title"]

    r_models = client.get("/api/models")
    assert r_models.status_code == 200
    models = r_models.json()["models"]
    assert any(m["model_key"] == "muril" for m in models)

    r_untrained = client.post(
        "/api/predict",
        json={"text": "जलवायु परिवर्तन गंभीर है", "language": "Auto", "model_key": "muril"},
    )
    if not next(m for m in models if m["model_key"] == "muril")["trained"]:
        assert r_untrained.status_code == 400
        assert "Model not trained" in r_untrained.json()["detail"]

    if next(m for m in models if m["model_key"] == "tfidf_logreg")["trained"]:
        r_pred = client.post(
            "/api/predict",
            json={
                "text": "Climate change is a serious global threat.",
                "language": "Auto",
                "model_key": "tfidf_logreg",
            },
        )
        assert r_pred.status_code == 200
        body = r_pred.json()
        assert body["stance"] in {"Favour", "Against", "None"}
        assert "probabilities" in body

        r_cmp = client.post(
            "/api/predict/compare",
            json={"text": "जलवायु परिवर्तन गंभीर है", "language": "Hindi"},
        )
        assert r_cmp.status_code == 200
        assert "predictions" in r_cmp.json()

        csv_bytes = b"id,comment\n1,Solar power helps fight global warming\n2,Global warming is a hoax\n"
        r_batch = client.post(
            "/api/predict/batch",
            files={"file": ("sample.csv", io.BytesIO(csv_bytes), "text/csv")},
            data={"text_column": "comment", "model_keys": "tfidf_logreg", "return_csv": "true"},
        )
        assert r_batch.status_code == 200
        assert "prediction" in r_batch.text
