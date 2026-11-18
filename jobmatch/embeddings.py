"""Text embedding backends behind one interface.

* SentenceTransformerBackend: dense transformer embeddings (all-MiniLM-L6-v2). Best quality, needs
  `pip install sentence-transformers` and a one-off model download.
* LSABackend: TF-IDF + truncated SVD (latent semantic analysis). Pure scikit-learn, works offline, and
  captures co-occurrence, so "k8s deployments" and "container orchestration" land close together.
"""
from __future__ import annotations
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize


class EmbeddingBackend:
    name = "base"
    needs_fit = False

    def fit(self, corpus: list[str]) -> "EmbeddingBackend":
        return self

    def encode(self, texts: list[str]) -> np.ndarray:  # rows are L2-normalised
        raise NotImplementedError


class LSABackend(EmbeddingBackend):
    name = "tfidf-lsa"
    needs_fit = True

    def __init__(self, dims: int = 24):
        self.dims = dims
        self.vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, stop_words="english",
                                   token_pattern=r"(?u)\b[\w+#.]+\b", min_df=1)
        self.svd: TruncatedSVD | None = None

    def fit(self, corpus: list[str]) -> "LSABackend":
        X = self.vec.fit_transform(corpus)
        k = max(2, min(self.dims, X.shape[0] - 1, X.shape[1] - 1))
        self.svd = TruncatedSVD(n_components=k, random_state=0).fit(X)
        return self

    def encode(self, texts: list[str]) -> np.ndarray:
        X = self.vec.transform(texts)
        return normalize(self.svd.transform(X))


class SentenceTransformerBackend(EmbeddingBackend):
    name = "sentence-transformers/all-MiniLM-L6-v2"

    def __init__(self, model: str = "sentence-transformers/all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer  # optional dependency
        self.model = SentenceTransformer(model)
        self.name = model

    def encode(self, texts: list[str]) -> np.ndarray:
        return np.asarray(self.model.encode(texts, normalize_embeddings=True, show_progress_bar=False))


def get_backend(kind: str = "auto") -> EmbeddingBackend:
    """kind: 'auto' (transformer if available else LSA), 'transformer', or 'lsa'."""
    if kind in ("auto", "transformer"):
        try:
            return SentenceTransformerBackend()
        except Exception:
            if kind == "transformer":
                raise
    return LSABackend()
