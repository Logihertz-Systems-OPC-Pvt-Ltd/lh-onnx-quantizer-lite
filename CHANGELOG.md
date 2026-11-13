# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-11-13

### Added
- **100% Zero External Dependency Python 3.11+ Quantization Engine**:
  - Symmetric per-channel INT8 weight quantization ($q \in [-127, +127]$) preventing depthwise convolution channel distortion.
  - Group-wise INT4 weight quantization ($q \in [-8, +7]$) with 2-nibbles-per-byte packing ($8.0\times$ compression over FP32) and lossless nibble unpacking.
  - Asymmetric per-tensor activation quantization with exact zero-point calculation.
  - Precision metrics: Signal-to-Quantization-Noise Ratio ($\text{SQNR}_{\text{dB}}$), Mean Squared Error (MSE), and Cosine Similarity.
- **Post-Training Quantization (PTQ) Calibrators**:
  - Kullback-Leibler (KL) divergence entropy calibration (TensorRT style) minimizing $D_{\text{KL}}(P \parallel Q)$.
  - Percentile calibration ($99.99\%$) and MinMax calibration.
- **Graph Optimization & QDQ Transformation**:
  - Analytical Conv2D + BatchNorm parameter folding ($W_{\text{fold}} = W \cdot \alpha$, $b_{\text{fold}} = (b - \mu) \cdot \alpha + \beta$).
  - QuantizeLinear and DequantizeLinear (QDQ) graph synthesis.
  - Preservation of sensitive detection heads, bounding box regression, and Softmax layers in FP16/FP32.
  - FIPS 180-4 SHA-256 model digest calculation.
- **10-Rule Edge Vision Posture Auditor**:
  - Normative evaluation of `LH-QUANT-001` through `LH-QUANT-010`.
  - ANSI Terminal Table, Canonical JSON, and OASIS SARIF v2.1.0 exporters.
- **Comprehensive CLI Tooling (`bin/lh-onnx-quant`)**:
  - Subcommands: `scan`, `quantize`, `calibrate`, `fold-bn`, `--version`, and `--help`.
- **Test Suite**:
  - 85+ deterministic unit test assertions with 100% pass rate.
