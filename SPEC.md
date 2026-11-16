# Engineering & Technical Architecture Specification

**Specification ID**: `LH-SPEC-2026-040`  
**Repository**: `lh-onnx-quantizer-lite`  
**CLI Executable**: `lh-onnx-quant`  
**Standard Release Version**: `0.1.0`  
**Standard License**: `Apache-2.0`  
**Classification**: `Industrial Sovereign Cybersecurity & Critical Infrastructure Telemetry`

---

## Section 1: Document Control & Specification Metadata

| Metadata Attribute | Value |
| :--- | :--- |
| **Document Identification** | `LH-SPEC-2026-040` |
| **System Name** | `Logihertz Systems Sovereign Ecosystem` |
| **Target Component** | `lh-onnx-quantizer-lite` (`lh-onnx-quant`) |
| **Target Language / Platform** | `Python` |
| **Legal Entity** | Logihertz Systems (OPC) Private Limited |
| **Corporate Identity (CIN)** | `U72100MH2025OPC463857` |
| **GSTIN Registration** | `27AAGCL6394P1ZL` |
| **Principal Architect** | Sameer Tawade (`ops@logihertz.com` / `security@logihertz.com`) |
| **Headquarters & Registered Office** | Mumbai, Maharashtra, Republic of India |
| **Official Portal** | [https://www.logihertz.com](https://www.logihertz.com) |
| **Repository URL** | [https://github.com/Logihertz-Systems-OPC-Pvt-Ltd/lh-onnx-quantizer-lite](https://github.com/Logihertz-Systems-OPC-Pvt-Ltd/lh-onnx-quantizer-lite) |
| **Document Status** | `APPROVED FOR PRODUCTION & INDUSTRIAL CERTIFICATION` |
| **Effective Release Date** | `2026-11-16` |

---

## Section 2: Normative RFC 2119 Requirements & System Invariants

This technical architecture specification enforces strict deterministic execution invariants in accordance with **RFC 2119** (`MUST`, `MUST NOT`, `REQUIRED`, `SHALL`, `SHALL NOT`, `SHOULD`, `RECOMMENDED`, `MAY`).

1. **Zero-Daemon Execution Invariant**:
   - The CLI binary `lh-onnx-quant` `MUST` execute as a clean, self-contained, stateless, zero-daemon process.
   - It `MUST NOT` spawn background daemons, fork unmonitored worker processes, or establish persistent Unix domain socket listeners unless explicitly invoked in daemon mode.
   - All state transitions `MUST` be deterministic, ephemeral, and strictly bounded by the execution lifecycle.

2. **Zero External Runtime Dependency Mandate**:
   - The software `MUST NOT` load, link, or depend upon third-party runtime package registries (e.g. PyPI, npm, Packagist, crates.io, Maven) at runtime.
   - For `Python`, all cryptographic, mathematical, serialization, and networking primitives `MUST` be implemented exclusively via language standard libraries or clean-room internal zero-dependency modules.

3. **Deterministic Air-Gapped Operation**:
   - The system `MUST` function with 100% feature parity in air-gapped, isolated, and offline enclaves.
   - The software `MUST NOT` initiate unsolicited network calls, telemetry telemetry pings, phone-home beacons, or external CDN queries.
   - All verification baselines, heuristics, and schemas `MUST` be packaged locally within the distribution.

4. **Constant-Time Cryptographic Comparison**:
   - All cryptographic comparison routines (hashes, HMAC tags, signatures, secret tokens, and commitments) `MUST` execute in constant time (`subtle.ConstantTimeCompare`, `CRYPTO_memcmp`, or equivalent).
   - The implementation `MUST NOT` short-circuit string comparisons when evaluating sensitive keys, thereby mitigating timing side-channel attacks.

5. **Memory Safety & Explicit Zeroization**:
   - All transient buffers containing sensitive key material, decrypted plaintexts, seeds, and cryptographic nonces `MUST` be explicitly overwritten with zeroes immediately following completion of the cryptographic operation.
   - For C/C++ implementations, bounds checks and safe buffer wrappers `MUST` prevent stack/heap overflows. For Rust implementations, `#![forbid(unsafe_code)]` `MUST` be enforced.

6. **Standardized Telemetry Interoperability**:
   - All security assessment outputs `MUST` support machine-readable ingestion via OASIS SARIF v2.1.0 JSON and Canonical Indented JSON schemas, alongside ANSI console tables.

---

## Section 3: Core Mathematical & Algorithmic Specification

### 3.1 Domain Engine Mathematical Foundation: Kullback-Leibler Divergence Entropy Minimization Calibration

The underlying computational core of `lh-onnx-quantizer-lite` is formally governed by the following mathematical and algorithmic formulations:

$$D_{\text{KL}}(P \parallel Q) = \sum_{i=1}^{N} P(i) \log_2 \left(\frac{P(i)}{Q(i)}\right), \quad T^* = \arg\min_T D_{\text{KL}}(P_T \parallel Q_T)$$

### 3.2 Standardized Logihertz Security Posture Score Formulation

To achieve absolute cross-tool normalization across all 60 repositories in the Logihertz Systems Ecosystem, the security posture score $S_{\text{posture}}$ is evaluated on a standardized continuous integer scale $[0, 100]$:

$$S_{\text{posture}} = \max\left(0, \; 100 - \left(25 \cdot c_{\text{CRITICAL}} + 10 \cdot c_{\text{HIGH}} + 4 \cdot c_{\text{MEDIUM}} + 1 \cdot c_{\text{LOW}}\right)\right)$$

Where:
- $c_{\text{CRITICAL}}$ represents the total count of triggered Critical-severity normative rules (deduction: $-25$ per finding).
- $c_{\text{HIGH}}$ represents the total count of triggered High-severity normative rules (deduction: $-10$ per finding).
- $c_{\text{MEDIUM}}$ represents the total count of triggered Medium-severity normative rules (deduction: $-4$ per finding).
- $c_{\text{LOW}}$ represents the total count of triggered Low-severity normative rules (deduction: $-1$ per finding).

The final compliance verdict is strictly bifurcated:
$$\text{Verdict} = \begin{cases} \text{PASS} & \text{if } S_{\text{posture}} = 100 \land \sum c_i = 0 \\ \text{FAIL} & \text{otherwise} \end{cases}$$

---

## Section 4: Normative 10-Rule Detection Matrix

The detection engine evaluates configurations, manifests, firmware, protocols, and streams against the following 10 deterministic security rules:

| Rule ID | Severity | Weight | CWE ID | Trigger Predicate / Condition | Remediation Guideline |
| :--- | :--- | :---: | :---: | :--- | :--- |
| `LH-QUANT-001` | `CRITICAL` | `25` | `CWE-502` | CWE-502 | Prohibits untrusted custom C++/Python op plugins and external data outside root |
| `LH-QUANT-002` | `CRITICAL` | `25` | `CWE-353` | CWE-353 | Enforces cryptographic SHA-256 verification on model and calibration datasets |
| `LH-QUANT-003` | `CRITICAL` | `25` | `CWE-682` | CWE-682 | Prevents perception collapse in robotics (SQNR >= 20.0 dB, mAP drop <= 2.0%) |
| `LH-QUANT-004` | `HIGH` | `10` | `CWE-682` | CWE-682 | Mandates per-channel quantization on Conv2D layers over per-tensor quantization |
| `LH-QUANT-005` | `HIGH` | `10` | `CWE-682` | CWE-682 | Enforces calibration dataset size (>= 100 samples) and verified PTQ method |
| `LH-QUANT-006` | `HIGH` | `10` | `CWE-682` | CWE-682 | Preserves sensitive detection heads (NMS, BBox regression, Softmax) in FP16/FP32 |
| `LH-QUANT-007` | `HIGH` | `10` | `CWE-400` | CWE-400 | Enforces bounded dynamic input tensor shapes (max_batch <= 32, workspace cap) |
| `LH-QUANT-008` | `MEDIUM` | `4` | `CWE-682` | CWE-682 | Mandates pre-quantization Conv2D + BatchNorm parameter folding |
| `LH-QUANT-009` | `MEDIUM` | `4` | `CWE-190` | CWE-190 | Mandates modern ONNX opset (>= 13) and INT8 saturation arithmetic |
| `LH-QUANT-010` | `LOW` | `1` | `CWE-755` | CWE-755 | Requires NaN/Inf weight checks and secure calibration cache purge on exit |

---

## Section 5: CLI Contract, Exit Codes & OASIS SARIF v2.1.0 Specification

### 5.1 Command-Line Interface Syntax

The primary entrypoint `lh-onnx-quant` conforms to standard POSIX argument conventions:

```text
USAGE:
  lh-onnx-quant <command> [flags] [arguments]

CORE COMMANDS:
  scan <target> [--format table|json|sarif] [--output <file>]
      Execute deterministic security evaluation against target manifest or telemetry stream.
  verify <target>
      Validate operational parameters, cryptographic checksums, and structural invariants.
  --version, version
      Output release version, architecture, and legal attribution.
  --help, help
      Display comprehensive operational usage and command reference.
```

### 5.2 Deterministic Exit Code Protocol

| Exit Code | Classification | Operational Semantics |
| :---: | :--- | :--- |
| `0` | **SUCCESS / PASS** | Target passed all checks; zero Critical/High violations detected; $S_{\text{posture}} = 100$. |
| `1` | **FINDINGS / FAIL** | Target triggered one or more security rule violations; policy threshold breached. |
| `2` | **ERROR / USAGE** | Invalid CLI invocation, syntax error, missing input file, or fatal I/O failure. |

### 5.3 OASIS SARIF v2.1.0 Schema Compliance

Exported SARIF documents validate against the official OASIS schema (`sarif-schema-2.1.0.json`). Each result object maps:
- `ruleId`: Standardized identifier (`LH-QUANT-001` .. `LH-QUANT-010`).
- `level`: SARIF severity levels (`error` for Critical/High, `warning` for Medium, `note` for Low).
- `message.text`: Comprehensive explanation, CWE mapping, and actionable remediation steps.
- `locations[].physicalLocation.artifactLocation.uri`: Absolute or normalized relative path of target artifact.

---

## 6. Standalone Future Development Roadmap (`v0.2.0` — `v0.5.0`)

`lh-onnx-quantizer-lite` is developed as an independent, standalone open-source project. The following roadmap defines the planned architectural, algorithmic, and protocol enhancements for upcoming point releases of `lh-onnx-quantizer-lite`:

| Target Release | Milestone Focus | Planned Technical Deliverables for `lh-onnx-quantizer-lite` |
|---|---|---|
| **`v0.2.0`** | Parser & Protocol Expansion | Add SmoothQuant per-channel activation-to-weight migration ($\alpha = 0.5$) and Hessian-weighted (GPTQ/AWQ-style) INT4 weight rounding. |
| **`v0.3.0`** | Deep Algorithmic & Stateful Analysis | Implement a native binary Protobuf .onnx (ModelProto / TensorProto) reader and writer with zero external library dependencies. |
| **`v0.4.0`** | Remediation, Synthesis & Format Interop | Add an automated mixed-precision layer sensitivity solver (lh-onnx-quant auto-mixed) that keeps high-MSE layers in FP16. |
| **`v0.5.0`** | High-Throughput Performance & Formal Verification | Export NVIDIA TensorRT explicit-precision QDQ calibration caches (.calib) and layer-by-layer SQNR sensitivity heatmaps. |

### 6.1 `v0.2.0` — Parser & Protocol Expansion
- **Format & Protocol Ingestion**: Add SmoothQuant per-channel activation-to-weight migration ($\alpha = 0.5$) and Hessian-weighted (GPTQ/AWQ-style) INT4 weight rounding.
- **Zero-Dependency Architecture**: Pure standard library implementation with deterministic parsing guarantees and strict error handling.
- **Verification & Telemetry**: Regression test suites verifying schema edge-cases, malformed payload rejections, and SARIF v2.1.0 output fidelity.

### 6.2 `v0.3.0` — Deep Algorithmic & Stateful Analysis
- **Algorithmic Core**: Implement a native binary Protobuf .onnx (ModelProto / TensorProto) reader and writer with zero external library dependencies.
- **State Machine & Analysis Engine**: High-fidelity state tracking, cycle detection, and invariant checking with minimal computational overhead.
- **Differential & Regression Verification**: Comprehensive test harnesses covering stateful transitions, edge conditions, and mathematical boundary constraints.

### 6.3 `v0.4.0` — Remediation, Synthesis & Format Interoperability
- **Synthesis & Interoperability**: Add an automated mixed-precision layer sensitivity solver (lh-onnx-quant auto-mixed) that keeps high-MSE layers in FP16.
- **Automated Remediation Workflows**: Idempotent generation of verified configuration patches, policy rules, and standardized cross-format exports.
- **Contract & Artifact Validation**: Cryptographic baseline verification and end-to-end integration testing for synthesized deliverables.

### 6.4 `v0.5.0` — High-Throughput Performance & Formal Verification
- **High-Throughput Optimization**: Export NVIDIA TensorRT explicit-precision QDQ calibration caches (.calib) and layer-by-layer SQNR sensitivity heatmaps.
- **Hardware & Kernel Acceleration**: Vectorized SIMD/eBPF/sub-millisecond streaming execution with bounded memory allocation.
- **Formal Verification & Assurance**: Mathematical proof of invariant preservation, false-positive calibration, and high-assurance certification baselines.

---

## Section 7: Verification, Quality Assurance & Legal Attribution

### 7.1 Automated Verification Protocol

Prior to distribution, all builds must execute the standardized Logihertz test suite:
- Unit and mathematical property tests verifying all equations and edge cases.
- Static analysis with zero warnings under standard strict linters.
- Subprocess CLI verification covering exit codes `0`, `1`, and `2`.

### 7.2 Copyright & Governance Notice

```text
Copyright 2025-2026 Logihertz Systems (OPC) Private Limited.
Creator, Founder, Director & Sole Developer: Sameer Tawade <ops@logihertz.com> / <security@logihertz.com>
Official Corporate Portal: https://www.logihertz.com
```

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this software except in compliance with the License. You may obtain a copy of the License at `http://www.apache.org/licenses/LICENSE-2.0`.
