# ==============================================================================
# Product:       Logihertz ONNX Quantizer Lite (lh-onnx-quantizer-lite)
# Module:        src/lh_onnx_quantizer_lite/calibration.py
# Role:          Kullback-Leibler (KL) Divergence Entropy, Percentile & MinMax Calibrators
# Organization:  Logihertz Systems (OPC) Private Limited
# Developer:     Sameer Tawade (Creator, Founder, Director & Sole Developer)
# Contact:       ops@logihertz.com / security@logihertz.com
# License:       Apache-2.0
# ==============================================================================

"""Pure Python 3.11+ zero-dependency Post-Training Quantization (PTQ) calibrators."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Sequence, Tuple


@dataclass
class CalibrationResult:
    """Container for activation calibration metrics."""
    method: str
    threshold: float
    min_val: float
    max_val: float
    kl_divergence: float | None = None
    percentile: float | None = None


class CalibrationEngine:
    """PTQ Calibration Engine implementing TensorRT-style KL-Divergence, Percentile, and MinMax."""

    @staticmethod
    def calibrate_minmax(samples: Sequence[float]) -> Tuple[float, float]:
        """Compute absolute minimum and maximum values of the sample set."""
        if not samples:
            return 0.0, 0.0
        return float(min(samples)), float(max(samples))

    @staticmethod
    def calibrate_percentile(samples: Sequence[float], percentile: float = 99.99) -> float:
        """Compute symmetric absolute threshold at the specified percentile (e.g. 99.99%)."""
        if not samples:
            return 0.0
        if not (0.0 < percentile <= 100.0):
            raise ValueError(f"percentile must be in (0, 100], got {percentile}")

        abs_sorted = sorted(abs(x) for x in samples)
        idx = int(round((percentile / 100.0) * (len(abs_sorted) - 1)))
        idx = max(0, min(len(abs_sorted) - 1, idx))
        return float(abs_sorted[idx])

    @classmethod
    def calibrate_kl_divergence(
        cls,
        samples: Sequence[float],
        num_bins: int = 512,
        target_bins: int = 128,
    ) -> Tuple[float, float]:
        """TensorRT-style Kullback-Leibler (KL) Divergence Entropy Calibration.

        Constructs an absolute-value histogram over [0, max(|samples|)], iterates candidate
        thresholds i in [target_bins, num_bins - 1], folds outliers into boundary bin i - 1,
        quantizes P to target_bins (128) levels, expands back to Q, and minimizes D_KL(P || Q).

        Returns:
            (optimal_threshold, min_kl_divergence)
        """
        if not samples:
            return 0.0, 0.0
        if num_bins <= target_bins:
            raise ValueError(f"num_bins ({num_bins}) must be greater than target_bins ({target_bins})")

        abs_samples = [abs(x) for x in samples]
        max_val = max(abs_samples)
        if max_val < 1e-12:
            return 0.0, 0.0

        bin_width = max_val / float(num_bins)

        # 1. Build histogram H of absolute values
        hist = [0.0] * num_bins
        for val in abs_samples:
            b = int(val / bin_width)
            if b >= num_bins:
                b = num_bins - 1
            hist[b] += 1.0

        min_kl = float("inf")
        optimal_bin = num_bins - 1

        # 2. Iterate candidate thresholds from target_bins to num_bins - 1
        for i in range(target_bins, num_bins):
            # Reference distribution P of length i
            p = hist[:i]

            # Fold outliers beyond bin i into the edge bin i - 1
            outlier_sum = sum(hist[i:])
            p[-1] += outlier_sum

            sum_p = sum(p)
            if sum_p < 1e-12:
                continue

            # Normalize P
            p_norm = [val / sum_p for val in p]

            # Quantize P into target_bins levels
            step = float(i) / float(target_bins)
            quantized = [0.0] * target_bins

            for q_idx in range(target_bins):
                start = int(round(q_idx * step))
                end = int(round((q_idx + 1) * step))
                end = min(end, i)
                if start < end:
                    quantized[q_idx] = sum(p_norm[start:end])

            # Expand quantized back to distribution Q of length i
            q_expanded = [0.0] * i
            for q_idx in range(target_bins):
                start = int(round(q_idx * step))
                end = int(round((q_idx + 1) * step))
                end = min(end, i)
                bin_count = end - start
                if bin_count <= 0:
                    continue

                # Count non-zero bins in P within this interval
                non_zeros = sum(1 for k in range(start, end) if p_norm[k] > 0.0)
                if non_zeros > 0:
                    val_per_bin = quantized[q_idx] / float(non_zeros)
                    for k in range(start, end):
                        if p_norm[k] > 0.0:
                            q_expanded[k] = val_per_bin
                else:
                    val_per_bin = quantized[q_idx] / float(bin_count)
                    for k in range(start, end):
                        q_expanded[k] = val_per_bin

            # Normalize Q
            sum_q = sum(q_expanded)
            if sum_q < 1e-12:
                continue
            q_norm = [val / sum_q for val in q_expanded]

            # Compute KL divergence: D_KL(P || Q) = sum( P * log(P / Q) )
            kl = 0.0
            for p_v, q_v in zip(p_norm, q_norm):
                if p_v > 1e-12:
                    kl += p_v * math.log((p_v + 1e-12) / (q_v + 1e-12))

            if kl < min_kl:
                min_kl = kl
                optimal_bin = i

        optimal_threshold = (optimal_bin + 0.5) * bin_width
        return float(optimal_threshold), float(min_kl)

    @classmethod
    def calibrate(
        cls,
        samples: Sequence[float],
        method: str = "kl_entropy",
        percentile: float = 99.99,
    ) -> CalibrationResult:
        """Unified calibration dispatcher."""
        min_v, max_v = cls.calibrate_minmax(samples)
        if method == "minmax":
            thresh = max(abs(min_v), abs(max_v))
            return CalibrationResult(
                method="minmax",
                threshold=thresh,
                min_val=min_v,
                max_val=max_v,
            )
        elif method == "percentile":
            thresh = cls.calibrate_percentile(samples, percentile=percentile)
            return CalibrationResult(
                method="percentile",
                threshold=thresh,
                min_val=min_v,
                max_val=max_v,
                percentile=percentile,
            )
        elif method == "kl_entropy":
            thresh, kl = cls.calibrate_kl_divergence(samples)
            return CalibrationResult(
                method="kl_entropy",
                threshold=thresh,
                min_val=min_v,
                max_val=max_v,
                kl_divergence=kl,
            )
        else:
            raise ValueError(f"Unknown calibration method '{method}'. Choose 'kl_entropy', 'percentile', or 'minmax'.")
