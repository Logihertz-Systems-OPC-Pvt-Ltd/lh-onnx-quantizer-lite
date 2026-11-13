# ==============================================================================
# Product:       Logihertz ONNX Quantizer Lite (lh-onnx-quantizer-lite)
# Module:        src/lh_onnx_quantizer_lite/auditor.py
# Role:          10-Rule Deterministic Edge Robotics Quantization Posture Auditor
# Organization:  Logihertz Systems (OPC) Private Limited
# Developer:     Sameer Tawade (Creator, Founder, Director & Sole Developer)
# Contact:       ops@logihertz.com / security@logihertz.com
# License:       Apache-2.0
# ==============================================================================

"""10-Rule deterministic Edge AI & Robotics quantization posture auditor."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class RuleDefinition:
    """Normative specification for a quantization posture rule."""
    id: str
    name: str
    severity: str
    cwe: str
    weight: int
    description: str
    remediation: str


@dataclass
class Finding:
    """Individual security or safety policy violation finding."""
    rule_id: str
    rule_name: str
    severity: str
    weight: int
    cwe: str
    message: str
    remediation: str


@dataclass
class AuditReport:
    """Aggregated audit assessment report."""
    manifest_path: str
    score: int
    status: str
    passed_rules: int
    total_rules: int
    findings: List[Finding] = field(default_factory=list)
    critical_count: int = 0
    high_count: int = 0
    medium_count: int = 0
    low_count: int = 0


class QuantizationAuditor:
    """Deterministic security and accuracy posture auditor for Edge AI quantization."""

    RULES: List[RuleDefinition] = [
        RuleDefinition(
            id="LH-QUANT-001",
            name="Untrusted Custom Operator Libraries or Path Traversal",
            severity="CRITICAL",
            cwe="CWE-502",
            weight=25,
            description="Custom C++/Python operator plugins allowed or external data outside model root permitted.",
            remediation="Set allow_custom_op_libraries: false and external_data_allow_outside_root: false.",
        ),
        RuleDefinition(
            id="LH-QUANT-002",
            name="Unverified Model Weights or Calibration Dataset Hash",
            severity="CRITICAL",
            cwe="CWE-353",
            weight=25,
            description="Missing cryptographic SHA-256 verification on source model weights or calibration datasets.",
            remediation="Enforce verify_model_sha256: true and verify_calibration_dataset_hash: true.",
        ),
        RuleDefinition(
            id="LH-QUANT-003",
            name="Severe Quantization Accuracy Degradation (Robotics Safety Risk)",
            severity="CRITICAL",
            cwe="CWE-682",
            weight=25,
            description="Quantization SQNR below 20.0 dB or mAP degradation exceeds 2.0% in safety-critical robotics.",
            remediation="Maintain sqnr_db >= 20.0 and map_drop_pct <= 2.0 via per-channel INT8 calibration.",
        ),
        RuleDefinition(
            id="LH-QUANT-004",
            name="Per-Tensor Weight Quantization on Convolutional Layers",
            severity="HIGH",
            cwe="CWE-682",
            weight=10,
            description="Per-tensor quantization applied to Conv2D layers causing inter-channel dynamic range collapse.",
            remediation="Enable conv_per_channel_quantization: true for all convolutional operators.",
        ),
        RuleDefinition(
            id="LH-QUANT-005",
            name="Undersized or Unverified PTQ Calibration Dataset",
            severity="HIGH",
            cwe="CWE-682",
            weight=10,
            description="Calibration dataset has fewer than 100 samples or uses an invalid calibration method.",
            remediation="Set calibration_samples >= 100 and use calibration_method in ['kl_entropy', 'percentile', 'minmax'].",
        ),
        RuleDefinition(
            id="LH-QUANT-006",
            name="Quantization of Sensitive Detection/NMS/Softmax Head",
            severity="HIGH",
            cwe="CWE-682",
            weight=10,
            description="Bounding-box regression, NMS, or classification Softmax head quantized to INT4/INT8.",
            remediation="Preserve sensitive vision heads in FP16/FP32 by setting preserve_fp16_detection_head: true.",
        ),
        RuleDefinition(
            id="LH-QUANT-007",
            name="Unbounded Dynamic Input Dimensions or Workspace Buffer",
            severity="HIGH",
            cwe="CWE-400",
            weight=10,
            description="Dynamic tensor shapes unbounded, batch size exceeds 32, or workspace buffer exceeds 1024 MB.",
            remediation="Constrain dynamic_axes_bounded: true, max_batch_size <= 32, and max_workspace_mb <= 1024.",
        ),
        RuleDefinition(
            id="LH-QUANT-008",
            name="Disabled Conv2D + BatchNorm Parameter Folding",
            severity="MEDIUM",
            cwe="CWE-682",
            weight=4,
            description="Batch normalization layers left unfused prior to quantization, increasing quantization noise.",
            remediation="Enable fold_conv_batchnorm: true prior to INT8/INT4 calibration.",
        ),
        RuleDefinition(
            id="LH-QUANT-009",
            name="Unsafe INT8 Saturation Arithmetic or Deprecated ONNX Opset",
            severity="MEDIUM",
            cwe="CWE-190",
            weight=4,
            description="ONNX opset version is older than 13 or INT8 saturation overflow arithmetic is disabled.",
            remediation="Set opset_version >= 13 and ensure saturate_int8_clipping: true.",
        ),
        RuleDefinition(
            id="LH-QUANT-010",
            name="Missing NaN/Inf Weight Sanitization or Calibration Cache Leak",
            severity="LOW",
            cwe="CWE-755",
            weight=1,
            description="Missing NaN/Inf weight sanitization or calibration cache purge on exit is disabled.",
            remediation="Enable check_nan_inf_weights: true and clear_calibration_cache_on_exit: true.",
        ),
    ]

    @classmethod
    def get_rules(cls) -> List[RuleDefinition]:
        """Return normative rule definitions."""
        return cls.RULES

    @classmethod
    def audit_policy(cls, policy: Dict[str, Any], manifest_path: str = "") -> AuditReport:
        """Audit quantization configuration policy against 10 normative rules."""
        findings: List[Finding] = []
        rules = cls.RULES

        # 1. LH-QUANT-001 (CRITICAL, 25)
        allow_custom = bool(policy.get("allow_custom_op_libraries", False))
        allow_outside = bool(policy.get("external_data_allow_outside_root", False))
        if allow_custom or allow_outside:
            findings.append(Finding(
                rule_id=rules[0].id,
                rule_name=rules[0].name,
                severity=rules[0].severity,
                weight=rules[0].weight,
                cwe=rules[0].cwe,
                message=f"allow_custom_op_libraries={allow_custom}, external_data_allow_outside_root={allow_outside}.",
                remediation=rules[0].remediation,
            ))

        # 2. LH-QUANT-002 (CRITICAL, 25)
        v_model = bool(policy.get("verify_model_sha256", False))
        v_calib = bool(policy.get("verify_calibration_dataset_hash", False))
        if not v_model or not v_calib:
            findings.append(Finding(
                rule_id=rules[1].id,
                rule_name=rules[1].name,
                severity=rules[1].severity,
                weight=rules[1].weight,
                cwe=rules[1].cwe,
                message=f"verify_model_sha256={v_model}, verify_calibration_dataset_hash={v_calib}.",
                remediation=rules[1].remediation,
            ))

        # 3. LH-QUANT-003 (CRITICAL, 25)
        sqnr = float(policy.get("sqnr_db", 0.0))
        map_drop = float(policy.get("map_drop_pct", 10.0))
        if sqnr < 20.0 or map_drop > 2.0:
            findings.append(Finding(
                rule_id=rules[2].id,
                rule_name=rules[2].name,
                severity=rules[2].severity,
                weight=rules[2].weight,
                cwe=rules[2].cwe,
                message=f"sqnr_db={sqnr} (< 20.0 dB) or map_drop_pct={map_drop}% (> 2.0%).",
                remediation=rules[2].remediation,
            ))

        # 4. LH-QUANT-004 (HIGH, 10)
        per_channel = bool(policy.get("conv_per_channel_quantization", False))
        if not per_channel:
            findings.append(Finding(
                rule_id=rules[3].id,
                rule_name=rules[3].name,
                severity=rules[3].severity,
                weight=rules[3].weight,
                cwe=rules[3].cwe,
                message=f"conv_per_channel_quantization={per_channel} (per-tensor Conv quantization detected).",
                remediation=rules[3].remediation,
            ))

        # 5. LH-QUANT-005 (HIGH, 10)
        calib_samples = int(policy.get("calibration_samples", 0))
        calib_method = str(policy.get("calibration_method", "")).lower()
        valid_methods = ["kl_entropy", "percentile", "minmax"]
        if calib_samples < 100 or calib_method not in valid_methods:
            findings.append(Finding(
                rule_id=rules[4].id,
                rule_name=rules[4].name,
                severity=rules[4].severity,
                weight=rules[4].weight,
                cwe=rules[4].cwe,
                message=f"calibration_samples={calib_samples} (< 100) or calibration_method='{calib_method}' not in {valid_methods}.",
                remediation=rules[4].remediation,
            ))

        # 6. LH-QUANT-006 (HIGH, 10)
        preserve_head = bool(policy.get("preserve_fp16_detection_head", False))
        if not preserve_head:
            findings.append(Finding(
                rule_id=rules[5].id,
                rule_name=rules[5].name,
                severity=rules[5].severity,
                weight=rules[5].weight,
                cwe=rules[5].cwe,
                message=f"preserve_fp16_detection_head={preserve_head} (detection/NMS head quantized).",
                remediation=rules[5].remediation,
            ))

        # 7. LH-QUANT-007 (HIGH, 10)
        dyn_bounded = bool(policy.get("dynamic_axes_bounded", False))
        max_batch = int(policy.get("max_batch_size", 64))
        max_ws = int(policy.get("max_workspace_mb", 2048))
        if not dyn_bounded or max_batch > 32 or max_ws > 1024:
            findings.append(Finding(
                rule_id=rules[6].id,
                rule_name=rules[6].name,
                severity=rules[6].severity,
                weight=rules[6].weight,
                cwe=rules[6].cwe,
                message=f"dynamic_axes_bounded={dyn_bounded}, max_batch_size={max_batch} (> 32), or max_workspace_mb={max_ws} (> 1024).",
                remediation=rules[6].remediation,
            ))

        # 8. LH-QUANT-008 (MEDIUM, 4)
        fold_bn = bool(policy.get("fold_conv_batchnorm", False))
        if not fold_bn:
            findings.append(Finding(
                rule_id=rules[7].id,
                rule_name=rules[7].name,
                severity=rules[7].severity,
                weight=rules[7].weight,
                cwe=rules[7].cwe,
                message=f"fold_conv_batchnorm={fold_bn} (redundant BatchNorm nodes unmerged).",
                remediation=rules[7].remediation,
            ))

        # 9. LH-QUANT-009 (MEDIUM, 4)
        opset = int(policy.get("opset_version", 0))
        saturate = bool(policy.get("saturate_int8_clipping", False))
        if opset < 13 or not saturate:
            findings.append(Finding(
                rule_id=rules[8].id,
                rule_name=rules[8].name,
                severity=rules[8].severity,
                weight=rules[8].weight,
                cwe=rules[8].cwe,
                message=f"opset_version={opset} (< 13) or saturate_int8_clipping={saturate}.",
                remediation=rules[8].remediation,
            ))

        # 10. LH-QUANT-010 (LOW, 1)
        chk_nan = bool(policy.get("check_nan_inf_weights", False))
        clear_cache = bool(policy.get("clear_calibration_cache_on_exit", False))
        if not chk_nan or not clear_cache:
            findings.append(Finding(
                rule_id=rules[9].id,
                rule_name=rules[9].name,
                severity=rules[9].severity,
                weight=rules[9].weight,
                cwe=rules[9].cwe,
                message=f"check_nan_inf_weights={chk_nan} or clear_calibration_cache_on_exit={clear_cache}.",
                remediation=rules[9].remediation,
            ))

        crit = sum(1 for f in findings if f.severity == "CRITICAL")
        high = sum(1 for f in findings if f.severity == "HIGH")
        med = sum(1 for f in findings if f.severity == "MEDIUM")
        low = sum(1 for f in findings if f.severity == "LOW")

        deduction = sum(f.weight for f in findings)
        score = max(0, 100 - deduction)
        status = "PASSED" if score == 100 else "FAILED"
        passed = len(rules) - len(findings)

        return AuditReport(
            manifest_path=manifest_path,
            score=score,
            status=status,
            passed_rules=passed,
            total_rules=len(rules),
            findings=findings,
            critical_count=crit,
            high_count=high,
            medium_count=med,
            low_count=low,
        )
