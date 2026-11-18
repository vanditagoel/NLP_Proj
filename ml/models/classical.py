"""Classical TF-IDF and dense Word2Vec/FastText classification pipelines."""
from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from ml.features.embeddings import GensimMeanEmbeddingVectorizer

CLASSICAL_MODEL_SPECS: dict[str, dict[str, Any]] = {
    "tfidf_logreg": {
        "name": "TF-IDF + Logistic Regression",
        "family": "Classical Baseline",
        "description": "Word (1-2) & character (3-5) TF-IDF features with class-balanced Logistic Regression.",
    },
    "tfidf_svm": {
        "name": "TF-IDF + Linear SVM",
        "family": "Classical Baseline",
        "description": "Word & subword character n-gram TF-IDF with probability-calibrated LinearSVC.",
    },
    "tfidf_nb": {
        "name": "TF-IDF + Multinomial Naive Bayes",
        "family": "Classical Baseline",
        "description": "Multilingual word & character n-gram TF-IDF with Complement/Multinomial Naive Bayes.",
    },
    "word2vec_lr": {
        "name": "Word2Vec + Logistic Regression",
        "family": "Dense Word Embeddings",
        "description": "Train-split-only Gensim Word2Vec mean-pooled sentence embeddings + Logistic Regression.",
    },
    "fasttext_lr": {
        "name": "FastText + Logistic Regression",
        "family": "Dense Subword Embeddings",
        "description": "Train-split-only Gensim FastText subword embeddings (robust to Indic morphology) + Logistic Regression.",
    },
}


def _multilingual_tfidf() -> FeatureUnion:
    return FeatureUnion(
        [
            (
                "word",
                TfidfVectorizer(
                    ngram_range=(1, 2),
                    min_df=1,
                    max_features=20000,
                    sublinear_tf=True,
                    token_pattern=r"(?u)\b\w+\b",
                ),
            ),
            (
                "char",
                TfidfVectorizer(
                    analyzer="char_wb",
                    ngram_range=(3, 5),
                    min_df=2,
                    max_features=25000,
                    sublinear_tf=True,
                ),
            ),
        ]
    )


def build_classical_pipeline(model_key: str, seed: int = 42) -> Pipeline:
    if model_key == "tfidf_logreg":
        return Pipeline(
            [
                ("features", _multilingual_tfidf()),
                (
                    "clf",
                    LogisticRegression(
                        max_iter=1500,
                        class_weight="balanced",
                        C=2.0,
                        random_state=seed,
                    ),
                ),
            ]
        )

    if model_key == "tfidf_svm":
        base_svc = LinearSVC(class_weight="balanced", C=1.0, random_state=seed)
        return Pipeline(
            [
                ("features", _multilingual_tfidf()),
                ("clf", CalibratedClassifierCV(base_svc, cv=3, method="sigmoid")),
            ]
        )

    if model_key == "tfidf_nb":
        return Pipeline(
            [
                ("features", _multilingual_tfidf()),
                ("clf", MultinomialNB(alpha=0.5)),
            ]
        )

    if model_key == "word2vec_lr":
        return Pipeline(
            [
                ("features", GensimMeanEmbeddingVectorizer(algo="word2vec", vector_size=100, seed=seed)),
                ("scaler", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        max_iter=1500,
                        class_weight="balanced",
                        C=1.0,
                        random_state=seed,
                    ),
                ),
            ]
        )

    if model_key == "fasttext_lr":
        return Pipeline(
            [
                ("features", GensimMeanEmbeddingVectorizer(algo="fasttext", vector_size=100, seed=seed)),
                ("scaler", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        max_iter=1500,
                        class_weight="balanced",
                        C=1.0,
                        random_state=seed,
                    ),
                ),
            ]
        )

    raise ValueError(f"Unknown classical model key: {model_key}")


def predict_classical_batch(pipeline: Pipeline, texts: list[str]) -> tuple[list[str], np.ndarray, list[str]]:
    preds = [str(p) for p in pipeline.predict(texts)]
    probs = pipeline.predict_proba(texts)
    classes = [str(c) for c in pipeline.classes_]
    return preds, probs, classes
