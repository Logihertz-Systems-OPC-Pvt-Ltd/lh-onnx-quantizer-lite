# Security Policy

## Supported Versions

Logihertz Systems (OPC) Private Limited maintains and supports the current major release of `lh-onnx-quantizer-lite`.

| Version | Supported          | Security Maintenance Status |
| :------ | :----------------- | :-------------------------- |
| 0.1.x   | :white_check_mark: | Active Security Maintenance |
| < 0.1.0 | :x:                | Unsupported / Deprecated    |

## Threat Model & Scope

`lh-onnx-quantizer-lite` executes edge vision model quantization, calibration, and graph optimization for safety-critical edge robotics and vision appliances. The threat model encompasses:
- Arbitrary code execution via untrusted custom C++/Python operator libraries.
- Path traversal and arbitrary file reads via unvalidated external tensor data references.
- Model weights tampering or calibration dataset poisoning due to unverified cryptographic checksums.
- Safety-critical accuracy collapse (SQNR degradation, mean Average Precision drops) leading to perception failure in robotics.
- Memory exhaustion and denial of service through unbounded dynamic tensor dimensions.

## Reporting a Vulnerability

If you discover a security vulnerability or safety hazard in `lh-onnx-quantizer-lite`, please report it promptly to the Logihertz security team:

- **Primary Security Contact**: Sameer Tawade (`security@logihertz.com`)
- **Operational Contact**: `ops@logihertz.com`
- **PGP Encryption**: Available upon request for sensitive disclosure materials.

Please include:
1. Description of the vulnerability, affected components, and potential safety or security impact.
2. Reproduction steps, including sample ONNX manifest or model graph files.
3. Relevant environment details (Python version, operating system, edge hardware target).

## Response & Disclosure SLA

- **Initial Triage & Acknowledgment**: Within 24–48 hours of receipt.
- **Root Cause & Remediation Plan**: Within 5 business days.
- **Security Advisory & Patch Release**: Within 14 calendar days via coordinated release.
