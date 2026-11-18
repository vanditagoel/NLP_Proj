"""Word2Vec and FastText mean-pooling scikit-learn feature transformers trained strictly on train splits."""
from __future__ import annotations

from typing import Iterable

import numpy as np
from gensim.models import FastText, Word2Vec
from sklearn.base import BaseEstimator, TransformerMixin


def _tokenize(text: str) -> list[str]:
    return [tok for tok in str(text).lower().split() if tok]


class GensimMeanEmbeddingVectorizer(BaseEstimator, TransformerMixin):
    """Fit Word2Vec or FastText on training texts only and mean-pool token vectors."""

    def __init__(
        self,
        algo: str = "word2vec",
        vector_size: int = 100,
        window: int = 5,
        min_count: int = 1,
        epochs: int = 15,
        bucket: int = 50000,
        seed: int = 42,
    ) -> None:
        self.algo = algo
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.epochs = epochs
        self.bucket = bucket
        self.seed = seed
        self.model_: Word2Vec | FastText | None = None

    def fit(self, X: Iterable[str], y: Iterable[str] | None = None) -> "GensimMeanEmbeddingVectorizer":
        sentences = [_tokenize(t) for t in X]
        if self.algo == "fasttext":
            self.model_ = FastText(
                sentences=sentences,
                vector_size=self.vector_size,
                window=self.window,
                min_count=self.min_count,
                epochs=self.epochs,
                bucket=self.bucket,
                seed=self.seed,
                workers=1,
            )
        else:
            self.model_ = Word2Vec(
                sentences=sentences,
                vector_size=self.vector_size,
                window=self.window,
                min_count=self.min_count,
                epochs=self.epochs,
                seed=self.seed,
                workers=1,
            )
        return self

    def transform(self, X: Iterable[str]) -> np.ndarray:
        if self.model_ is None:
            raise RuntimeError("GensimMeanEmbeddingVectorizer must be fitted before transform().")
        wv = self.model_.wv
        rows = []
        for text in X:
            vecs = [wv[tok] for tok in _tokenize(text) if tok in wv]
            if vecs:
                rows.append(np.mean(vecs, axis=0))
            else:
                rows.append(np.zeros(self.vector_size, dtype=np.float32))
        return np.vstack(rows).astype(np.float32)
