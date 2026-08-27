"""Quantization: SQ8 (scalar 8-bit) + BF16 (bfloat16) — used when Namespace.compression != NONE."""

from __future__ import annotations

import numpy as np


def sq8_quantize(v: np.ndarray) -> tuple[np.ndarray, float, float]:
    """Per-vector SQ8: float32 → uint8 + scale/zero-point. Returns (codes, scale, zp)."""
    v = np.asarray(v, dtype=np.float32)
    vmin, vmax = float(v.min()), float(v.max())
    if vmax == vmin:
        return np.zeros_like(v, dtype=np.uint8), 1.0, vmin
    scale = (vmax - vmin) / 255.0
    zp = vmin
    codes = np.clip(np.round((v - zp) / scale), 0, 255).astype(np.uint8)
    return codes, scale, zp


def sq8_dequantize(codes: np.ndarray, scale: float, zp: float) -> np.ndarray:
    return codes.astype(np.float32) * scale + zp


def bf16_quantize(v: np.ndarray) -> np.ndarray:
    """BF16: truncate mantissa to 7 bits (simulated via float32 view). Returns uint16 codes."""
    v = np.asarray(v, dtype=np.float32)
    # view as uint32, zero lower 16 bits
    u32 = v.view(np.uint32)
    u32_bf16 = u32 & np.uint32(0xFFFF0000)
    # store as uint16 high bits for compactness
    codes = (u32_bf16 >> np.uint32(16)).astype(np.uint16)
    return codes


def bf16_dequantize(codes: np.ndarray) -> np.ndarray:
    u32 = codes.astype(np.uint32) << np.uint32(16)
    return u32.view(np.float32)


# Helpers for index wrapper
def compress_vectors(vectors: np.ndarray, compression: str) -> tuple[np.ndarray, dict]:
    c = compression.upper()
    if c == "SQ8":
        # per-vector pack — return stacked codes + scales/zps
        codes_list, scales, zps = [], [], []
        for row in vectors:
            codes, sc, zp = sq8_quantize(row)
            codes_list.append(codes)
            scales.append(sc)
            zps.append(zp)
        return np.stack(codes_list), {"scales": np.array(scales, dtype=np.float32), "zps": np.array(zps, dtype=np.float32), "type": "SQ8"}
    elif c == "BF16":
        codes = np.stack([bf16_quantize(r) for r in vectors])
        return codes, {"type": "BF16"}
    else:
        return vectors, {"type": "NONE"}


def decompress_query(vector: np.ndarray, compression: str, meta: dict | None = None) -> np.ndarray:
    # query stays float32 — decompression is for DB vectors during scoring
    return vector.astype(np.float32)
