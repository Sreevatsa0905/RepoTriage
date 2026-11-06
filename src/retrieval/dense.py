"""
Dense Retrieval with BAAI/bge-base-en-v1.5 for RepoTriage
=========================================================
Implements dense embedding generation with query instruction prefix,
disk caching keyed by content_hash, and exact numpy cosine similarity search.
"""

from __future__ import annotations

import gc
import logging
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

logger = logging.getLogger(__name__)

MODEL_NAME = "BAAI/bge-base-en-v1.5"
QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "
DEFAULT_BATCH_SIZE = 32

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
EMBEDDINGS_CACHE_PATH = PROCESSED_DIR / "embeddings_cache.npz"


class DenseRetriever:
    """
    Dense semantic retriever powered by BAAI/bge-base-en-v1.5.
    Caches chunk embeddings by content_hash on disk.
    """

    def __init__(
        self,
        chunks: list[dict[str, Any]] | None = None,
        model_name: str = MODEL_NAME,
        cache_path: Path = EMBEDDINGS_CACHE_PATH,
        device: str | None = None,
    ):
        self.chunks = chunks or []
        self.model_name = model_name
        self.cache_path = cache_path
        self._model = None

        if device is not None:
            self.device = device
        else:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # In-memory hash -> embedding map
        self.cache: dict[str, np.ndarray] = {}
        self._load_cache()

        # Matrix for active chunks
        self.active_embeddings: np.ndarray | None = None
        if self.chunks:
            self.prepare_corpus()

    def _load_cache(self) -> None:
        """Load cached embeddings from .npz if available."""
        if self.cache_path.exists():
            try:
                data = np.load(str(self.cache_path), allow_pickle=True)
                hashes = data["hashes"]
                vectors = data["embeddings"]
                for h, vec in zip(hashes, vectors):
                    self.cache[str(h)] = vec
                logger.info("Loaded %d cached embeddings from %s", len(self.cache), self.cache_path.name)
            except Exception as e:
                logger.warning("Failed loading embeddings cache: %s. Starting fresh.", e)
                self.cache = {}

    def _save_cache(self) -> None:
        """Persist in-memory cache to disk atomically."""
        if not self.cache:
            return
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        hashes = np.array(list(self.cache.keys()), dtype=object)
        vectors = np.stack(list(self.cache.values()), axis=0).astype(np.float32)

        temp_path = self.cache_path.with_suffix(".tmp.npz")
        np.savez_compressed(temp_path, hashes=hashes, embeddings=vectors)
        temp_path.replace(self.cache_path)
        logger.info("Saved %d embeddings to cache %s", len(hashes), self.cache_path.name)

    def _get_model(self):
        """Lazy loader for SentenceTransformer model."""
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            logger.info("Loading dense model %s on %s...", self.model_name, self.device)
            self._model = SentenceTransformer(self.model_name, device=self.device)
        return self._model

    def unload_model(self) -> None:
        """Free model from memory and clear GPU cache."""
        if self._model is not None:
            logger.info("Unloading dense model %s from %s", self.model_name, self.device)
            del self._model
            self._model = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def prepare_corpus(self, chunks: list[dict[str, Any]] | None = None) -> None:
        """
        Ensure all chunks in the corpus have embeddings in cache,
        computing and persisting any missing embeddings in batches.
        """
        if chunks is not None:
            self.chunks = chunks

        if not self.chunks:
            self.active_embeddings = None
            return

        missing_indices: list[int] = []
        missing_texts: list[str] = []

        for idx, c in enumerate(self.chunks):
            chash = c["content_hash"]
            if chash not in self.cache:
                missing_indices.append(idx)
                # Passages are embedded as-is without query instruction prefix
                missing_texts.append(c["text"])

        if missing_texts:
            logger.info(
                "Embedding %d new chunks with %s (batch_size=%d)...",
                len(missing_texts), self.model_name, DEFAULT_BATCH_SIZE,
            )
            model = self._get_model()
            new_embeddings = model.encode(
                missing_texts,
                batch_size=DEFAULT_BATCH_SIZE,
                show_progress_bar=len(missing_texts) > 50,
                normalize_embeddings=True,
                convert_to_numpy=True,
            )
            for idx, emb in zip(missing_indices, new_embeddings):
                chash = self.chunks[idx]["content_hash"]
                self.cache[chash] = emb.astype(np.float32)

            self._save_cache()

        # Build contiguous active embeddings matrix
        vectors = [self.cache[c["content_hash"]] for c in self.chunks]
        self.active_embeddings = np.stack(vectors, axis=0).astype(np.float32)

    def encode_query(self, query: str) -> np.ndarray:
        """
        Encode a search query with the required BGE instruction prefix.
        Returns a normalized 1D float32 numpy vector.
        """
        model = self._get_model()
        prompted_query = f"{QUERY_INSTRUCTION}{query.strip()}"
        q_emb = model.encode(
            prompted_query,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return q_emb.astype(np.float32)

    def search(self, query: str, top_k: int = 50) -> list[dict[str, Any]]:
        """
        Search corpus using exact cosine similarity over normalized embeddings.
        Returns top_k chunks with dense_score and dense_rank.
        """
        if not self.chunks or self.active_embeddings is None:
            return []

        q_vec = self.encode_query(query)
        # Cosine similarity for unit vectors is simply dot product
        scores = np.dot(self.active_embeddings, q_vec)

        top_indices = np.argsort(-scores)[:top_k]

        results = []
        for rank, idx in enumerate(top_indices, 1):
            chunk = dict(self.chunks[idx])
            chunk["dense_score"] = float(scores[idx])
            chunk["dense_rank"] = rank
            results.append(chunk)

        return results
