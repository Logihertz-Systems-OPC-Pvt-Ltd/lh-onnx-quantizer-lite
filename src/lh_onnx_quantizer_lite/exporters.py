# ==============================================================================
# Product:       Logihertz ONNX Quantizer Lite (lh-onnx-quantizer-lite)
# Module:        src/lh_onnx_quantizer_lite/exporters.py
# Role:          ANSI Terminal Table, Canonical JSON, and OASIS SARIF v2.1.0 Exporters
# Organization:  Logihertz Systems (OPC) Private Limited
# Developer:     Sameer Tawade (Creator, Founder, Director & Sole Developer)
# Contact:       ops@logihertz.com / security@logihertz.com
# License:       Apache-2.0
# ==============================================================================

"""Exporters for audit reports: ANSI table, Canonical JSON, and OASIS SARIF v2.1.0."""

from __future__ import annotations

import json
from typing import Any, Dict

from lh_onnx_quantizer_lite.auditor import AuditReport, QuantizationAuditor


class Exporters:
    """Formatters for quantization posture audit findings."""

    @staticmethod
    def format_table(report: AuditReport) -> str:
        """Format report into an ANSI terminal table with score banner."""
        lines = [
            "",
            "=" * 80,
            "   Logihertz ONNX Quantizer Lite - Edge Robotics Posture Auditor",
            f"   Policy Manifest: {report.manifest_path or 'In-Memory / Direct Input'}",
            f"   Posture Score:   {report.score}/100   [{report.status}]",
            f"   Summary:         {report.passed_rules} passed, {len(report.findings)} violations "
            f"(CRIT: {report.critical_count}, HIGH: {report.high_count}, MED: {report.medium_count}, LOW: {report.low_count})",
            "=" * 80,
            "",
        ]

        if not report.findings:
            lines.append(" [OK] All 10 Edge AI & Robotics quantization posture controls are satisfied.")
            lines.append("      Model deployment policy conforms to production-grade sovereign standards.\n")
            return "\n".join(lines)

        col_fmt = "+----------------+----------+--------+----------+--------------------------------------+"
        header  = "| Rule ID        | Severity | Weight | CWE      | Rule Title                           |"
        lines.append(col_fmt)
        lines.append(header)
        lines.append(col_fmt)

        for f in report.findings:
            short_name = f.rule_name if len(f.rule_name) <= 36 else (f.rule_name[:33] + "...")
            row = (
                f"| {f.rule_id:<14} | {f.severity:<8} | {f.weight:<6} | {f.cwe:<8} | {short_name:<36} |"
            )
            lines.append(row)

            msg_sub = f.message if len(f.message) <= 66 else (f.message[:63] + "...")
            lines.append(f"|   -> Finding:  {msg_sub:<67} |")

            rem_sub = f.remediation if len(f.remediation) <= 66 else (f.remediation[:63] + "...")
            lines.append(f"|   -> Action:   {rem_sub:<67} |")
            lines.append(col_fmt)

        lines.append("")
        return "\n".join(lines)

    @staticmethod
    def format_json(report: AuditReport) -> str:
        """Format report into structured Canonical JSON."""
        data: Dict[str, Any] = {
            "version": "0.1.0",
            "scanner": "lh-onnx-quant",
            "organization": "Logihertz Systems (OPC) Private Limited",
            "developer": "Sameer Tawade",
            "manifest_path": report.manifest_path,
            "score": report.score,
            "status": report.status,
            "total_rules": report.total_rules,
            "passed_rules": report.passed_rules,
            "severity_counts": {
                "critical": report.critical_count,
                "high": report.high_count,
                "medium": report.medium_count,
                "low": report.low_count,
            },
            "findings": [
                {
                    "rule_id": f.rule_id,
                    "rule_name": f.rule_name,
                    "severity": f.severity,
                    "weight": f.weight,
                    "cwe": f.cwe,
                    "message": f.message,
                    "remediation": f.remediation,
                }
                for f in report.findings
            ],
        }
        return json.dumps(data, indent=2, sort_keys=True)

    @staticmethod
    def format_sarif(report: AuditReport) -> str:
        """Format report into OASIS SARIF v2.1.0."""
        sarif_rules = []
        for r in QuantizationAuditor.get_rules():
            sarif_rules.append({
                "id": r.id,
                "name": r.name,
                "shortDescription": {"text": r.name},
                "fullDescription": {"text": r.description},
                "help": {"text": r.remediation},
                "properties": {
                    "tags": ["security", "ai-safety", "quantization", "robotics", r.cwe.lower()],
                    "precision": "high",
                    "problem.severity": "error" if r.severity in ["CRITICAL", "HIGH"] else "warning",
                    "security-severity": "9.0" if r.severity == "CRITICAL" else ("7.0" if r.severity == "HIGH" else "4.0"),
                },
            })

        results = []
        for f in report.findings:
            level = "error" if f.severity in ["CRITICAL", "HIGH"] else "warning"
            results.append({
                "ruleId": f.rule_id,
                "level": level,
                "message": {
                    "text": f"{f.rule_name}: {f.message} Remediation: {f.remediation}"
                },
                "locations": [
                    {
                        "physicalLocation": {
                            "artifactLocation": {
                                "uri": report.manifest_path or "manifest.json"
                            }
                        }
                    }
                ],
                "properties": {
                    "cwe": f.cwe,
                    "severity": f.severity,
                    "weight": f.weight,
                },
            })

        sarif: Dict[str, Any] = {
            "$schema": "https://raw.githubusercontent.com/oasis-tcs/sarif-spec/master/Schemata/sarif-schema-2.1.0.json",
            "version": "2.1.0",
            "runs": [
                {
                    "tool": {
                        "driver": {
                            "name": "lh-onnx-quant",
                            "semanticVersion": "0.1.0",
                            "informationUri": "https://github.com/Logihertz-Systems-OPC-Pvt-Ltd/lh-onnx-quantizer-lite",
                            "organization": "Logihertz Systems (OPC) Private Limited",
                            "rules": sarif_rules,
                        }
                    },
                    "results": results,
                }
            ],
        }
        return json.dumps(sarif, indent=2, sort_keys=True)
