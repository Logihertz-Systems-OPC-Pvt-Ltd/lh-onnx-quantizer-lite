# ==============================================================================
# Product:       Logihertz ONNX Quantizer Lite (lh-onnx-quantizer-lite)
# Module:        src/lh_onnx_quantizer_lite/quant_engine.py
# Role:          Symmetric Per-Channel INT8, Group-Wise INT4 & Asymmetric Activation Quantization
# Organization:  Logihertz Systems (OPC) Private Limited
# Developer:     Sameer Tawade (Creator, Founder, Director & Sole Developer)
# Contact:       ops@logihertz.com / security@logihertz.com
# License:       Apache-2.0
# ==============================================================================

"""Pure Python 3.11+ zero-dependency quantization math, nibble packing, and SQNR engine."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Sequence, Tuple, Union


def clamp(val: float | int, low: float | int, high: float | int) -> float | int:
    """Clamp value to [low, high]."""
    return max(low, min(high, val))


def compute_sqnr_db(original: Sequence[float], reconstructed: Sequence[float]) -> float:
    """Calculate Signal-to-Quantization-Noise Ratio in decibels (SQNR dB)."""
    if len(original) != len(reconstructed) or not original:
        return 0.0
    signal_pwr = sum(x * x for x in original)
    noise_pwr = sum((x - r) * (x - r) for x, r in zip(original, reconstructed))
    if noise_pwr < 1e-12:
        return 120.0  # Cap near-infinite SQNR
    if signal_pwr < 1e-12:
        return 0.0
    return 10.0 * math.log10(signal_pwr / (noise_pwr + 1e-12))


def compute_mse(original: Sequence[float], reconstructed: Sequence[float]) -> float:
    """Calculate Mean Squared Error (MSE)."""
    if len(original) != len(reconstructed) or not original:
        return 0.0
    return sum((x - r) * (x - r) for x, r in zip(original, reconstructed)) / len(original)


def compute_cosine_similarity(original: Sequence[float], reconstructed: Sequence[float]) -> float:
    """Calculate Cosine Similarity between original and reconstructed tensors."""
    if len(original) != len(reconstructed) or not original:
        return 1.0
    dot = sum(x * r for x, r in zip(original, reconstructed))
    norm_orig = math.sqrt(sum(x * x for x in original))
    norm_rec = math.sqrt(sum(r * r for r in reconstructed))
    if norm_orig < 1e-12 or norm_rec < 1e-12:
        return 1.0
    return dot / (norm_orig * norm_rec + 1e-12)


@dataclass
class PerChannelInt8Result:
    """Result of per-channel symmetric INT8 quantization."""
    quantized_weights: List[List[int]]     # [C_out, K] signed int8 in [-127, 127]
    scales: List[float]                    # [C_out] per-channel scale factors
    dequantized_weights: List[List[float]] # [C_out, K] reconstructed floats
    sqnr_db: float                         # Global SQNR in dB
    mse: float                             # Global MSE
    cosine_similarity: float               # Global Cosine Similarity
    channels: int                          # C_out
    elements_per_channel: int              # K


@dataclass
class GroupWiseInt4Result:
    """Result of group-wise symmetric INT4 quantization with 2-nibble packing."""
    packed_bytes: bytes                    # Packed 4-bit nibbles (2 per byte)
    group_scales: List[float]              # Scale factor per group
    group_size: int                        # Number of elements per group G
    total_elements: int                    # Total unpacked element count
    dequantized_weights: List[float]       # Reconstructed FP32 weights
    sqnr_db: float                         # Global SQNR in dB
    mse: float                             # Global MSE
    cosine_similarity: float               # Global Cosine Similarity
    compression_ratio: float               # Memory reduction over FP32 (8.0x)


@dataclass
class AsymmetricQuantResult:
    """Result of asymmetric per-tensor activation quantization."""
    quantized_activations: List[int]       # Clamped values in [qmin, qmax]
    scale: float                           # Step size s
    zero_point: int                        # Integer zero point z
    qmin: int                              # Lower bound (e.g. 0 or -128)
    qmax: int                              # Upper bound (e.g. 255 or 127)
    dequantized_activations: List[float]   # Reconstructed FP32 values
    sqnr_db: float                         # Activation SQNR in dB
    mse: float                             # Activation MSE


class QuantEngine:
    """Core mathematical quantization engine for INT4 and INT8 vision models."""

    @staticmethod
    def quantize_per_channel_int8(
        weights: Union[List[List[float]], Sequence[Sequence[float]]],
    ) -> PerChannelInt8Result:
        """Symmetric per-channel INT8 weight quantization.

        Each row in weights represents an output channel c.
        s_c = max(|W_c|) / 127.0 + 1e-12
        q_cj = clamp(round(W_cj / s_c), -127, 127)
        """
        c_out = len(weights)
        if c_out == 0:
            raise ValueError("weights tensor cannot be empty")
        k = len(weights[0])

        quant_w: List[List[int]] = []
        scales: List[float] = []
        dequant_w: List[List[float]] = []

        all_orig: List[float] = []
        all_dequant: List[float] = []

        for row in weights:
            if len(row) != k:
                raise ValueError(f"Inconsistent row length {len(row)} vs expected {k}")
            max_abs = max((abs(x) for x in row), default=0.0)
            scale = (max_abs / 127.0) + 1e-12
            scales.append(scale)

            q_row = [int(clamp(round(x / scale), -127, 127)) for x in row]
            dequant_row = [q * scale for q in q_row]

            quant_w.append(q_row)
            dequant_w.append(dequant_row)

            all_orig.extend(row)
            all_dequant.extend(dequant_row)

        sqnr = compute_sqnr_db(all_orig, all_dequant)
        mse = compute_mse(all_orig, all_dequant)
        cos_sim = compute_cosine_similarity(all_orig, all_dequant)

        return PerChannelInt8Result(
            quantized_weights=quant_w,
            scales=scales,
            dequantized_weights=dequant_w,
            sqnr_db=sqnr,
            mse=mse,
            cosine_similarity=cos_sim,
            channels=c_out,
            elements_per_channel=k,
        )

    @staticmethod
    def pack_int4_nibbles(signed_q: Sequence[int]) -> bytes:
        """Pack signed 4-bit integers [-8, 7] into 2 nibbles per byte.

        u = (q + 8) & 0x0F  in [0, 15]
        byte = u_even | (u_odd << 4)
        """
        nibbles = [(int(clamp(q, -8, 7)) + 8) & 0x0F for q in signed_q]
        if len(nibbles) % 2 != 0:
            nibbles.append(8)  # Pad with 0 (u=8 represents q=0)

        out_bytes = bytearray(len(nibbles) // 2)
        for i in range(0, len(nibbles), 2):
            low = nibbles[i]
            high = nibbles[i + 1]
            out_bytes[i // 2] = low | (high << 4)

        return bytes(out_bytes)

    @staticmethod
    def unpack_int4_nibbles(packed_bytes: bytes, total_elements: int) -> List[int]:
        """Unpack 2 nibbles per byte back into signed 4-bit integers [-8, 7]."""
        unpacked: List[int] = []
        for b in packed_bytes:
            low = (b & 0x0F) - 8
            high = ((b >> 4) & 0x0F) - 8
            unpacked.append(low)
            unpacked.append(high)

        return unpacked[:total_elements]

    @classmethod
    def quantize_group_wise_int4(
        cls,
        weights: Sequence[float],
        group_size: int = 32,
    ) -> GroupWiseInt4Result:
        """Group-wise symmetric INT4 quantization with 2-nibble packing.

        Partitions weights into groups of size group_size:
        s_g = max(|W_g|) / 7.0 + 1e-12
        q_j = clamp(round(W_j / s_g), -8, 7)
        Packs into 2 nibbles per byte (8.0x compression over FP32).
        """
        n = len(weights)
        if n == 0:
            raise ValueError("weights sequence cannot be empty")
        if group_size <= 0:
            raise ValueError(f"group_size must be positive, got {group_size}")

        num_groups = math.ceil(n / group_size)
        group_scales: List[float] = []
        quant_signed: List[int] = []
        dequant_w: List[float] = []

        for g in range(num_groups):
            start = g * group_size
            end = min(n, start + group_size)
            grp = weights[start:end]

            max_abs = max((abs(x) for x in grp), default=0.0)
            scale = (max_abs / 7.0) + 1e-12
            group_scales.append(scale)

            for x in grp:
                q = int(clamp(round(x / scale), -8, 7))
                quant_signed.append(q)
                dequant_w.append(q * scale)

        packed = cls.pack_int4_nibbles(quant_signed)

        # Theoretical compression ratio: 32 bits FP32 vs 4 bits INT4 = 8.0x
        fp32_bytes = n * 4
        comp_ratio = fp32_bytes / max(1, len(packed))

        sqnr = compute_sqnr_db(weights, dequant_w)
        mse = compute_mse(weights, dequant_w)
        cos_sim = compute_cosine_similarity(weights, dequant_w)

        return GroupWiseInt4Result(
            packed_bytes=packed,
            group_scales=group_scales,
            group_size=group_size,
            total_elements=n,
            dequantized_weights=dequant_w,
            sqnr_db=sqnr,
            mse=mse,
            cosine_similarity=cos_sim,
            compression_ratio=comp_ratio,
        )

    @staticmethod
    def quantize_activation_asymmetric(
        activations: Sequence[float],
        xmin: float,
        xmax: float,
        qmin: int = 0,
        qmax: int = 255,
    ) -> AsymmetricQuantResult:
        """Asymmetric per-tensor activation quantization.

        Ensures 0.0 is represented by expanding [xmin, xmax] if needed:
        xmin = min(xmin, 0.0)
        xmax = max(xmax, 0.0)
        s = (xmax - xmin) / (qmax - qmin) + 1e-12
        z = clamp(round(qmin - xmin / s), qmin, qmax)
        q(x) = clamp(round(x / s) + z, qmin, qmax)
        dequant(q) = (q - z) * s
        """
        if not activations:
            raise ValueError("activations cannot be empty")
        if qmin >= qmax:
            raise ValueError(f"qmin ({qmin}) must be less than qmax ({qmax})")

        # Range must include 0.0 for exact zero-point representation
        adj_xmin = min(xmin, 0.0)
        adj_xmax = max(xmax, 0.0)

        scale = ((adj_xmax - adj_xmin) / float(qmax - qmin)) + 1e-12
        zero_point = int(clamp(round(float(qmin) - (adj_xmin / scale)), qmin, qmax))

        quant_act = [int(clamp(round(x / scale) + zero_point, qmin, qmax)) for x in activations]
        dequant_act = [(q - zero_point) * scale for q in quant_act]

        sqnr = compute_sqnr_db(activations, dequant_act)
        mse = compute_mse(activations, dequant_act)

        return AsymmetricQuantResult(
            quantized_activations=quant_act,
            scale=scale,
            zero_point=zero_point,
            qmin=qmin,
            qmax=qmax,
            dequantized_activations=dequant_act,
            sqnr_db=sqnr,
            mse=mse,
        )
