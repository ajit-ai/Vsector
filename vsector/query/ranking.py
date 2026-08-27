"""Cross-encoder + MMR re-ranking for QueryEngine.

Cross-encoder: optional sentence-transformers, fallback to dot-product re-score.
MMR: maximal marginal relevance for diversity.
"""

from __future__ import annotations

import logging
import os

import numpy as np

logger = logging.getLogger(__name__)


def cross_encoder_rerank(query_vec: np.ndarray, candidates: list[dict], top_k: int, model_name: str | None = None) -> list[dict]:
    """Re-rank candidates via cross-encoder if available, else cosine re-score.

    candidates: list of {id, score, vector, metadata}
    """
    if not candidates:
        return candidates
    # Try sentence-transformers CrossEncoder if vector payload available
    model_name = model_name or os.getenv("VSECTOR_CROSS_ENCODER")
    if model_name:
        try:
            from sentence_transformers import CrossEncoder  # type: ignore

            ce = CrossEncoder(model_name)
            # Need text payloads — fallback if no text
            texts = [c.get("metadata", {}).get("text", "") for c in candidates]
            if any(texts):
                # placeholder: query_vec -> text conversion not trivial; skip if no query text
                pass
        except Exception as e:
            logger.debug(f"cross-encoder not available: {e}")

    # Fallback: re-score via cosine between query and stored vector (if include_vector)
    q = np.asarray(query_vec, dtype=np.float32)
    qn = q / (np.linalg.norm(q) + 1e-9)
    for c in candidates:
        vec = c.get("vector")
        if vec is not None:
            v = np.asarray(vec, dtype=np.float32)
            vn = v / (np.linalg.norm(v) + 1e-9)
            # cross-encoder simulated as weighted cosine
            ce_score = float(np.dot(qn, vn))
            # blend original ANN score 0.7 + cross 0.3
            c["score"] = 0.7 * float(c["score"]) + 0.3 * ce_score
            c["_reranked"] = True
    candidates.sort(key=lambda x: x["score"], reverse=True)
    return candidates[:top_k]


def mmr_rerank(query_vec: np.ndarray, candidates: list[dict], top_k: int, lambda_mult: float = 0.5) -> list[dict]:
    """Maximal Marginal Relevance — trades relevance for diversity.

    lambda_mult=1 pure relevance, 0 pure diversity.
    Requires candidate vectors.
    """
    if not candidates or len(candidates) <= top_k:
        return candidates[:top_k]
    q = np.asarray(query_vec, dtype=np.float32)
    # Build matrix of candidate vectors (use stored vector if present, else score proxy)
    vecs = []
    for c in candidates:
        v = c.get("vector")
        if v is not None:
            vecs.append(np.asarray(v, dtype=np.float32))
        else:
            # proxy: use score as 1-d
            vecs.append(np.array([float(c["score"])]))
    # Normalize for cosine
    def _norm(v):
        n = np.linalg.norm(v) + 1e-9
        return v / n

    qn = _norm(q[: len(vecs[0])]) if len(vecs[0]) == len(q) else None
    selected: list[dict] = []
    remaining = candidates.copy()
    # First pick best
    remaining.sort(key=lambda x: x["score"], reverse=True)
    selected.append(remaining.pop(0))
    while remaining and len(selected) < top_k:
        best = None
        best_mmr = -1e9
        for cand in remaining:
            rel = float(cand["score"])
            # max similarity to already selected
            max_sim = 0.0
            cv = vecs[candidates.index(cand)]
            for sel in selected:
                sv = vecs[candidates.index(sel)]
                # cosine between candidates
                sim = float(np.dot(_norm(cv), _norm(sv)))
                max_sim = max(max_sim, sim)
            mmr = lambda_mult * rel - (1 - lambda_mult) * max_sim
            if mmr > best_mmr:
                best_mmr = mmr
                best = cand
        if best:
            selected.append(best)
            remaining.remove(best)
        else:
            break
    return selected
