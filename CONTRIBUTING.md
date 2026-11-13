# Contributing to Logihertz ONNX Quantizer Lite

Thank you for your interest in contributing to **`lh-onnx-quantizer-lite`** (`bin/lh-onnx-quant`), an open-source project by Logihertz Systems (OPC) Private Limited.

## Core Mandates & Architecture Principles

1. **Zero External Runtime Dependencies**:
   `lh-onnx-quantizer-lite` relies strictly on the Python 3.11+ Standard Library (`json`, `math`, `hashlib`, `struct`, `argparse`, `pathlib`, `dataclasses`, `typing`). No third-party packages (e.g., numpy, scipy, onnx, onnxruntime, torch) are permitted in runtime code.
2. **Deterministic Quantization Math**:
   - Symmetric per-channel INT8 quantization ($q \in [-127, +127]$).
   - Group-wise symmetric INT4 quantization ($q \in [-8, +7]$) with 2-nibbles-per-byte packing ($8.0\times$ compression over FP32).
   - Asymmetric per-tensor activation scaling with exact zero-point representation.
   - Kullback-Leibler (KL) divergence histogram calibration (TensorRT style) for optimal threshold selection.
   - Analytical Conv2D + BatchNorm parameter fusion ($W_{\text{fold}} = W \cdot \alpha$, $b_{\text{fold}} = (b - \mu) \cdot \alpha + \beta$).
3. **Deterministic Edge AI Posture Auditor**:
   Any new rule or check must be integrated into `src/lh_onnx_quantizer_lite/auditor.py` with assigned CWE, severity (`CRITICAL`, `HIGH`, `MEDIUM`, `LOW`), deduction weight, and remediation guidance.

## Development Workflow

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/Logihertz-Systems-OPC-Pvt-Ltd/lh-onnx-quantizer-lite.git
   cd lh-onnx-quantizer-lite
   ```

2. **Verify Python Environment**:
   Python 3.11, 3.12, or 3.13 is required:
   ```bash
   python3 --version
   ```

3. **Run Test Suite**:
   ```bash
   python3 -m unittest discover -s tests -v
   ```

4. **Run Syntax & Linter Verification**:
   ```bash
   python3 -m py_compile src/lh_onnx_quantizer_lite/*.py tests/*.py
   ```

5. **Commit Standards**:
   - Use conventional commit messages: `feat:`, `fix:`, `docs:`, `test:`, `refactor:`.
   - Maintain full attribution to Logihertz Systems (OPC) Private Limited and Sameer Tawade (`ops@logihertz.com`).
