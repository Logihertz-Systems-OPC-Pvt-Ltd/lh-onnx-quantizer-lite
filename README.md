# Logihertz ONNX Quantizer Lite (`lh-onnx-quantizer-lite`)

[![CI](https://github.com/Logihertz-Systems-OPC-Pvt-Ltd/lh-onnx-quantizer-lite/actions/workflows/ci.yml/badge.svg)](https://github.com/Logihertz-Systems-OPC-Pvt-Ltd/lh-onnx-quantizer-lite/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.11%20%7C%203.12%20%7C%203.13-blue)](pyproject.toml)
[![Zero Dependencies](https://img.shields.io/badge/Dependencies-0%20External-success.svg)](pyproject.toml)

**`lh-onnx-quantizer-lite`** (`bin/lh-onnx-quant`) is a production-grade, zero-external-dependency Python 3.11+ Edge Vision INT4/INT8 ONNX model quantizer, Kullback-Leibler (KL) divergence entropy calibrator, Conv2D-BatchNorm parameter folder, and 10-rule deterministic Edge Robotics quantization posture auditor.

---

## Key Features

- **100% Zero External Runtime Dependencies**: Standard Python 3.11+ libraries only (`json`, `math`, `hashlib`, `struct`, `argparse`, `pathlib`, `dataclasses`, `typing`). No NumPy, SciPy, PyTorch, ONNX, or ONNXRuntime required.
- **Symmetric Per-Channel INT8 Quantization ($q \in [-127, +127]$)**: Prevents depthwise convolution channel distortion by computing unique scale factors $s_c$ per output filter channel.
- **Group-Wise Symmetric INT4 Quantization & 2-Nibble Packing**: Partitions weights into blocks ($G=32, 64, 128$), clamps to $[-8, +7]$, and packs two 4-bit weights into one byte ($8.0\times$ compression over FP32) with lossless unpacking.
- **Asymmetric Per-Tensor Activation Quantization**: Calibrated scale $s$ and zero-point $z$ ensuring exact representability of $0.0$ without zero-point drift.
- **Kullback-Leibler (KL) Divergence Calibration**: TensorRT-style entropy minimization over activation histograms to pick optimal clipping thresholds $T_{\text{opt}}$.
- **Analytical Conv2D + BatchNorm Parameter Folding**: Mathematically merges batch normalization parameters into convolutional weights and biases prior to quantization with $< 10^{-6}$ error.
- **10-Rule Edge Robotics Quantization Posture Auditor**: Audits deployment manifests against security, memory safety, and perception accuracy controls (`LH-QUANT-001`..`010`), exporting to ANSI Box Table, Canonical JSON, and OASIS SARIF v2.1.0.

---

## Mathematical Formulations

### 1. Symmetric Per-Channel INT8 Weight Quantization
For channel $c \in [0, C_{\text{out}}-1]$ of weight matrix $W$:
$$s_c = \frac{\max_{0 \le j < K} |W_{c,j}|}{127.0} + 10^{-12}$$
$$q_{c,j} = \text{clamp}\left(\text{round}\left(\frac{W_{c,j}}{s_c}\right), \; -127, \; +127\right)$$
$$\hat{W}_{c,j} = q_{c,j} \cdot s_c$$

### 2. Group-Wise Symmetric INT4 Weight Quantization & Nibble Packing
For each group $g$ of size $G$:
$$s_g = \frac{\max_{j \in g} |W_j|}{7.0} + 10^{-12}, \qquad q_j = \text{clamp}\left(\text{round}\left(\frac{W_j}{s_g}\right), \; -8, \; +7\right)$$
Signed integer $q \in [-8, +7]$ is mapped to unsigned nibble $u = (q + 8) \ \& \ \text{0x0F} \in [0, 15]$. Pairs $(u_{2k}, u_{2k+1})$ pack into single byte:
$$\text{byte}_k = u_{2k} \mid (u_{2k+1} \ll 4)$$

### 3. Kullback-Leibler (KL) Divergence Entropy Calibration
For candidate clipping bin $i \in [128, N_{\text{bins}}-1]$, fold outliers into boundary bin $P[i-1]$, quantize to 128 levels, expand back to $Q$, and minimize:
$$D_{\text{KL}}(P \parallel Q) = \sum_{b=0}^{i-1} P(b) \log\left(\frac{P(b) + 10^{-12}}{Q(b) + 10^{-12}}\right)$$

### 4. Conv2D + BatchNormalization Analytical Parameter Folding
$$\alpha_c = \frac{\gamma_c}{\sqrt{\sigma_c^2 + \epsilon}}, \qquad W_{\text{fold}, c} = W_c \cdot \alpha_c, \qquad b_{\text{fold}, c} = (b_c - \mu_c) \cdot \alpha_c + \beta_c$$

---

## 10-Rule Deterministic Quantization Posture Auditor

Post-quantization deployment policies are audited using the standardized score:
$$S_{\text{posture}} = \max\left(0, 100 - \left(25 \cdot c_{\text{CRIT}} + 10 \cdot c_{\text{HIGH}} + 4 \cdot c_{\text{MED}} + 1 \cdot c_{\text{LOW}}\right)\right)$$

| Rule ID | Severity | Weight | CWE | Audit Scope & Remediation |
| :--- | :--- | :---: | :--- | :--- |
| `LH-QUANT-001` | CRITICAL | 25 | CWE-502 | Prohibits untrusted custom C++/Python op plugins and external data outside root |
| `LH-QUANT-002` | CRITICAL | 25 | CWE-353 | Enforces cryptographic SHA-256 verification on model and calibration datasets |
| `LH-QUANT-003` | CRITICAL | 25 | CWE-682 | Prevents perception collapse in robotics (`SQNR >= 20.0 dB`, `mAP drop <= 2.0%`) |
| `LH-QUANT-004` | HIGH | 10 | CWE-682 | Mandates per-channel quantization on Conv2D layers over per-tensor quantization |
| `LH-QUANT-005` | HIGH | 10 | CWE-682 | Enforces calibration dataset size (`>= 100 samples`) and verified PTQ method |
| `LH-QUANT-006` | HIGH | 10 | CWE-682 | Preserves sensitive detection heads (NMS, BBox regression, Softmax) in FP16/FP32 |
| `LH-QUANT-007` | HIGH | 10 | CWE-400 | Enforces bounded dynamic input tensor shapes (`max_batch <= 32`, workspace cap) |
| `LH-QUANT-008` | MEDIUM | 4 | CWE-682 | Mandates pre-quantization Conv2D + BatchNorm parameter folding |
| `LH-QUANT-009` | MEDIUM | 4 | CWE-190 | Mandates modern ONNX opset (`>= 13`) and INT8 saturation arithmetic |
| `LH-QUANT-010` | LOW | 1 | CWE-755 | Requires NaN/Inf weight checks and secure calibration cache purge on exit |

---

## CLI Reference Guide

```bash
# Display help and version
bin/lh-onnx-quant --help
bin/lh-onnx-quant --version

# Audit quantization posture against 10 normative rules
bin/lh-onnx-quant scan fixtures/hardened_quant_config.json --format table
bin/lh-onnx-quant scan fixtures/hardened_quant_config.json --format json
bin/lh-onnx-quant scan fixtures/vulnerable_quant_config.json --format sarif --output audit.sarif

# Calibrate activations using KL-divergence entropy
bin/lh-onnx-quant calibrate fixtures/sample_vision_graph.json --method kl_entropy

# Fold Conv2D + BatchNorm layers
bin/lh-onnx-quant fold-bn fixtures/sample_vision_graph.json

# Quantize graph with per-channel INT8 or group-wise INT4
bin/lh-onnx-quant quantize fixtures/sample_vision_graph.json --bits 8
bin/lh-onnx-quant quantize fixtures/sample_vision_graph.json --bits 4 --group-size 32 --output /tmp/quant_model.json
```

---

## Attribution & Governance

- **Organization**: Logihertz Systems (OPC) Private Limited (`https://logihertz.com/`)
- **Creator, Founder, Director & Sole Developer**: Sameer Tawade (`ops@logihertz.com` / `security@logihertz.com`)
- **License**: Apache License 2.0
