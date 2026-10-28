"""Local dense embeddings for deterministic Regularization atom retrieval.

The Regularization compares short English rule atoms symmetrically.  It therefore
encodes both sides without a query instruction, L2-normalizes the vectors,
and uses cosine similarity.  Model resolution is deliberately local-only so
an experiment cannot silently switch to a network service or a different
model when the machine is offline.
"""
from __future__ import annotations

import os
import threading
from functools import lru_cache
from pathlib import Path
from typing import Sequence

import numpy as np


DEFAULT_EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"
DEFAULT_EMBEDDING_MATCH_THRESHOLD = 0.75
EMBEDDING_METRIC = "cosine"


class LocalEmbeddingError(RuntimeError):
    """Raised when the pinned local embedding runtime is unavailable."""


def _resolve_model_root(model_id_or_path: str) -> Path:
    override = os.environ.get("SKILLEVOREG_EMBEDDING_MODEL_PATH", "").strip()
    requested = override or model_id_or_path
    direct = Path(requested).expanduser()
    if direct.is_dir():
        return direct.resolve()
    try:
        from huggingface_hub import snapshot_download

        return Path(snapshot_download(
            repo_id=requested,
            local_files_only=True,
        )).resolve()
    except Exception as exc:  # pragma: no cover - exact hub exception varies
        raise LocalEmbeddingError(
            f"embedding model {requested!r} is not available locally; run "
            f"`hf download {requested}` before starting the experiment"
        ) from exc


class _OnnxSentenceEmbedder:
    def __init__(self, model_id_or_path: str) -> None:
        try:
            import onnxruntime as ort
            from tokenizers import Tokenizer
        except ImportError as exc:  # pragma: no cover - dependency preflight
            raise LocalEmbeddingError(
                "local embedding retrieval requires onnxruntime and tokenizers"
            ) from exc

        self.model_id = model_id_or_path
        self.model_root = _resolve_model_root(model_id_or_path)
        tokenizer_path = self.model_root / "tokenizer.json"
        model_path = self.model_root / "onnx" / "model.onnx"
        if not tokenizer_path.is_file() or not model_path.is_file():
            raise LocalEmbeddingError(
                f"embedding snapshot is incomplete: {self.model_root}"
            )

        self.tokenizer = Tokenizer.from_file(str(tokenizer_path))
        self.tokenizer.enable_truncation(max_length=512)
        pad_id = self.tokenizer.token_to_id("[PAD]")
        self.tokenizer.enable_padding(
            pad_id=0 if pad_id is None else pad_id,
            pad_token="[PAD]",
        )

        options = ort.SessionOptions()
        options.log_severity_level = 3
        options.intra_op_num_threads = max(
            1, int(os.environ.get("SKILLEVOREG_EMBEDDING_THREADS", "2"))
        )
        options.inter_op_num_threads = 1
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(
            str(model_path),
            sess_options=options,
            providers=["CPUExecutionProvider"],
        )
        self._cache: dict[str, np.ndarray] = {}
        self._lock = threading.Lock()

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, 768), dtype=np.float32)
        normalized_texts = [str(text).strip() for text in texts]
        missing = list(dict.fromkeys(
            text for text in normalized_texts if text not in self._cache
        ))
        if missing:
            encodings = self.tokenizer.encode_batch(missing)
            feeds = {
                "input_ids": np.asarray(
                    [encoding.ids for encoding in encodings], dtype=np.int64
                ),
                "attention_mask": np.asarray(
                    [encoding.attention_mask for encoding in encodings],
                    dtype=np.int64,
                ),
                "token_type_ids": np.asarray(
                    [encoding.type_ids for encoding in encodings], dtype=np.int64
                ),
            }
            with self._lock:
                hidden = self.session.run(["last_hidden_state"], feeds)[0]
            vectors = np.asarray(hidden[:, 0, :], dtype=np.float32)
            norms = np.linalg.norm(vectors, axis=1, keepdims=True)
            vectors = vectors / np.maximum(norms, np.finfo(np.float32).eps)
            self._cache.update(zip(missing, vectors, strict=True))
        return np.stack([self._cache[text] for text in normalized_texts])


@lru_cache(maxsize=4)
def _embedder(model_id_or_path: str) -> _OnnxSentenceEmbedder:
    return _OnnxSentenceEmbedder(model_id_or_path)


def cosine_similarity_matrix(
    left: Sequence[str],
    right: Sequence[str],
    *,
    model_id_or_path: str = DEFAULT_EMBEDDING_MODEL,
) -> np.ndarray:
    """Return normalized dense cosine similarities for two atom collections."""
    if not left or not right:
        return np.empty((len(left), len(right)), dtype=np.float32)
    model = _embedder(model_id_or_path)
    return model.encode(left) @ model.encode(right).T


__all__ = [
    "DEFAULT_EMBEDDING_MATCH_THRESHOLD",
    "DEFAULT_EMBEDDING_MODEL",
    "EMBEDDING_METRIC",
    "LocalEmbeddingError",
    "cosine_similarity_matrix",
]
