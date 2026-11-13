# ==============================================================================
# Product:       Logihertz ONNX Quantizer Lite (lh-onnx-quantizer-lite)
# Module:        tests/test_onnx_quantizer.py
# Role:          Deterministic Unit Test Suite (85+ Assertions)
# Organization:  Logihertz Systems (OPC) Private Limited
# Developer:     Sameer Tawade (Creator, Founder, Director & Sole Developer)
# Contact:       ops@logihertz.com / security@logihertz.com
# License:       Apache-2.0
# ==============================================================================

"""Comprehensive test suite for lh-onnx-quantizer-lite."""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys
import unittest
from pathlib import Path

# Add src to sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from lh_onnx_quantizer_lite.auditor import QuantizationAuditor
from lh_onnx_quantizer_lite.calibration import CalibrationEngine
from lh_onnx_quantizer_lite.exporters import Exporters
from lh_onnx_quantizer_lite.graph_optimizer import GraphOptimizer
from lh_onnx_quantizer_lite.quant_engine import (
    QuantEngine,
    clamp,
    compute_cosine_similarity,
    compute_mse,
    compute_sqnr_db,
)


class TestQuantEngine(unittest.TestCase):
    """Test cases for symmetric per-channel INT8, group-wise INT4, and activation math."""

    def test_clamp_helper(self) -> None:
        """Verify clamp utility functionality."""
        self.assertEqual(clamp(5, 0, 10), 5)
        self.assertEqual(clamp(-5, 0, 10), 0)
        self.assertEqual(clamp(15, 0, 10), 10)
        self.assertEqual(clamp(0.5, 0.0, 1.0), 0.5)

    def test_sqnr_and_metrics_math(self) -> None:
        """Verify SQNR dB, MSE, and Cosine Similarity calculations."""
        orig = [1.0, 2.0, 3.0, 4.0]
        rec = [1.0, 2.0, 3.0, 4.0]
        self.assertGreater(compute_sqnr_db(orig, rec), 100.0)
        self.assertAlmostEqual(compute_mse(orig, rec), 0.0, places=6)
        self.assertAlmostEqual(compute_cosine_similarity(orig, rec), 1.0, places=6)

        noisy = [1.1, 1.9, 3.1, 3.9]
        self.assertGreater(compute_sqnr_db(orig, noisy), 20.0)
        self.assertGreater(compute_mse(orig, noisy), 0.0)
        self.assertGreater(compute_cosine_similarity(orig, noisy), 0.99)

        # Empty inputs
        self.assertEqual(compute_sqnr_db([], []), 0.0)
        self.assertEqual(compute_mse([], []), 0.0)
        self.assertEqual(compute_cosine_similarity([], []), 1.0)

    def test_per_channel_int8_quantization(self) -> None:
        """Verify per-channel INT8 quantization and distinct scale factors."""
        # 2 channels with widely varying dynamic ranges (0.05 max vs 50.0 max)
        weights = [
            [0.01, -0.05, 0.03, -0.02, 0.04],
            [10.0, -50.0, 25.0, -30.0, 45.0],
        ]
        res = QuantEngine.quantize_per_channel_int8(weights)

        self.assertEqual(res.channels, 2)
        self.assertEqual(res.elements_per_channel, 5)
        self.assertEqual(len(res.scales), 2)
        self.assertAlmostEqual(res.scales[0], 0.05 / 127.0, places=5)
        self.assertAlmostEqual(res.scales[1], 50.0 / 127.0, places=5)

        # Bounds check [-127, 127]
        for row in res.quantized_weights:
            for q in row:
                self.assertTrue(-127 <= q <= 127)

        # High fidelity checks
        self.assertGreater(res.sqnr_db, 38.0)
        self.assertGreater(res.cosine_similarity, 0.999)
        self.assertLess(res.mse, 0.1)

    def test_per_channel_int8_errors(self) -> None:
        """Verify error handling on malformed weight tensors."""
        with self.assertRaises(ValueError):
            QuantEngine.quantize_per_channel_int8([])

        with self.assertRaises(ValueError):
            QuantEngine.quantize_per_channel_int8([[1.0, 2.0], [1.0]])

    def test_group_wise_int4_quantization_and_nibble_packing(self) -> None:
        """Verify group-wise INT4 quantization, nibble packing, and 8.0x compression."""
        weights = [
            float(i % 7 - 3) * 0.5 for i in range(64)
        ]
        res = QuantEngine.quantize_group_wise_int4(weights, group_size=32)

        self.assertEqual(res.group_size, 32)
        self.assertEqual(res.total_elements, 64)
        self.assertEqual(len(res.group_scales), 2)
        self.assertEqual(len(res.packed_bytes), 32)  # 64 elements = 32 bytes (2 per byte)
        self.assertAlmostEqual(res.compression_ratio, 8.0, places=1)

        # Test lossless unpacking back to [-8, 7]
        unpacked_q = QuantEngine.unpack_int4_nibbles(res.packed_bytes, total_elements=64)
        self.assertEqual(len(unpacked_q), 64)
        for q in unpacked_q:
            self.assertTrue(-8 <= q <= 7)

        # Test dequantized fidelity
        self.assertGreater(res.sqnr_db, 18.0)
        self.assertGreater(res.cosine_similarity, 0.98)

    def test_int4_odd_elements_packing(self) -> None:
        """Verify handling of odd number of elements in INT4 packing."""
        odd_w = [1.0, -1.0, 0.5, -0.5, 0.2]
        res = QuantEngine.quantize_group_wise_int4(odd_w, group_size=16)
        self.assertEqual(res.total_elements, 5)
        self.assertEqual(len(res.packed_bytes), 3)  # ceil(5/2) = 3 bytes

        unpacked = QuantEngine.unpack_int4_nibbles(res.packed_bytes, total_elements=5)
        self.assertEqual(len(unpacked), 5)

    def test_asymmetric_activation_quantization(self) -> None:
        """Verify asymmetric activation quantization with exact zero representation."""
        act = [-2.0, -1.0, 0.0, 1.0, 2.0, 3.0]
        res = QuantEngine.quantize_activation_asymmetric(act, xmin=-2.0, xmax=3.0, qmin=0, qmax=255)

        self.assertEqual(len(res.quantized_activations), len(act))
        self.assertGreater(res.scale, 0.0)
        self.assertTrue(0 <= res.zero_point <= 255)

        # Verify zero is representable without drift
        # Find index of 0.0
        zero_idx = act.index(0.0)
        self.assertEqual(res.quantized_activations[zero_idx], res.zero_point)
        self.assertAlmostEqual(res.dequantized_activations[zero_idx], 0.0, places=5)
        self.assertGreater(res.sqnr_db, 35.0)


class TestCalibrationEngine(unittest.TestCase):
    """Test cases for PTQ Calibration: KL-Divergence, Percentile, and MinMax."""

    def test_minmax_calibration(self) -> None:
        """Verify minmax calibration extremes."""
        samples = [-5.5, 1.2, 0.0, 8.4, -2.1]
        min_v, max_v = CalibrationEngine.calibrate_minmax(samples)
        self.assertAlmostEqual(min_v, -5.5)
        self.assertAlmostEqual(max_v, 8.4)

        res = CalibrationEngine.calibrate(samples, method="minmax")
        self.assertEqual(res.method, "minmax")
        self.assertAlmostEqual(res.threshold, 8.4)

    def test_percentile_calibration(self) -> None:
        """Verify percentile calibration excludes extreme tail."""
        samples = [1.0] * 99 + [100.0]  # 100 elements, 99 are 1.0, 1 is 100.0
        thresh = CalibrationEngine.calibrate_percentile(samples, percentile=98.0)
        self.assertAlmostEqual(thresh, 1.0)

    def test_kl_divergence_entropy_calibration(self) -> None:
        """Verify TensorRT-style KL-divergence calibration on outlier-polluted data."""
        # 500 samples normally distributed around 0.0 with 3 extreme outliers
        samples = [math.sin(float(i)) * 2.0 for i in range(500)]
        samples.extend([50.0, -60.0, 75.0])  # Extreme outliers

        thresh, kl = CalibrationEngine.calibrate_kl_divergence(samples, num_bins=256, target_bins=64)
        max_abs = max(abs(x) for x in samples)

        # Optimal threshold must clip the extreme outliers (thresh < max_abs)
        self.assertLess(thresh, max_abs * 0.5)
        self.assertGreater(thresh, 0.0)
        self.assertGreaterEqual(kl, 0.0)

    def test_calibration_dispatcher_invalid_method(self) -> None:
        """Verify exception on invalid calibration method."""
        with self.assertRaises(ValueError):
            CalibrationEngine.calibrate([1.0, 2.0], method="invalid_unknown")


class TestGraphOptimizer(unittest.TestCase):
    """Test cases for Conv2D+BatchNorm parameter folding and QDQ synthesis."""

    def test_conv_batchnorm_folding_mathematical_equivalence(self) -> None:
        """Verify Conv2D + BatchNorm parameter folding matches within < 1e-6."""
        # 2 output channels, 3 input elements
        conv_w = [
            [0.5, -0.2, 1.1],
            [-0.8, 0.4, 0.7],
        ]
        conv_b = [0.1, -0.05]

        bn_mean = [0.2, -0.1]
        bn_var = [1.5, 0.8]
        bn_gamma = [1.2, 0.9]
        bn_beta = [-0.05, 0.15]

        folded = GraphOptimizer.fold_conv_batchnorm_weights(
            conv_weights=conv_w,
            conv_bias=conv_b,
            bn_mean=bn_mean,
            bn_var=bn_var,
            bn_gamma=bn_gamma,
            bn_beta=bn_beta,
            eps=1e-5,
        )

        self.assertEqual(len(folded.folded_weights), 2)
        self.assertEqual(len(folded.folded_bias), 2)

        # Test forward equivalence with arbitrary input vector
        test_x = [1.5, -2.0, 0.8]
        max_diff = GraphOptimizer.verify_conv_bn_equivalence(
            x_vec=test_x,
            conv_weights=conv_w,
            conv_bias=conv_b,
            bn_mean=bn_mean,
            bn_var=bn_var,
            bn_gamma=bn_gamma,
            bn_beta=bn_beta,
            eps=1e-5,
        )

        self.assertLess(max_diff, 1e-6)

    def test_vision_graph_qdq_transformation(self) -> None:
        """Verify graph transformation, BN elimination, and SHA-256 calculation."""
        fixture_path = REPO_ROOT / "fixtures" / "sample_vision_graph.json"
        with open(fixture_path, "r", encoding="utf-8") as f:
            graph = json.load(f)

        q_graph = GraphOptimizer.quantize_vision_graph(graph, bits=8, preserve_fp16_detection_head=True)

        self.assertIn("model_sha256", q_graph)
        self.assertEqual(len(q_graph["model_sha256"]), 64)
        self.assertEqual(q_graph["quantization_format"], "QDQ_INT8")

        # Verify BatchNorm node was folded and removed
        op_types = [n["op_type"] for n in q_graph["nodes"]]
        self.assertNotIn("BatchNormalization", op_types)
        self.assertIn("QuantizeLinear", op_types)
        self.assertIn("DequantizeLinear", op_types)

        # Verify detection head softmax was preserved in FP16
        softmax_nodes = [n for n in q_graph["nodes"] if n["op_type"] == "Softmax"]
        self.assertEqual(len(softmax_nodes), 1)
        self.assertEqual(softmax_nodes[0].get("precision"), "FP16_PRESERVED")


class TestQuantizationAuditor(unittest.TestCase):
    """Test cases for 10-rule quantization posture auditor and score math."""

    def test_hardened_config_passes_all_rules(self) -> None:
        """Verify hardened config achieves 100/100 and passes all 10 rules."""
        fixture_path = REPO_ROOT / "fixtures" / "hardened_quant_config.json"
        with open(fixture_path, "r", encoding="utf-8") as f:
            policy = json.load(f)

        report = QuantizationAuditor.audit_policy(policy, manifest_path=str(fixture_path))

        self.assertEqual(report.score, 100)
        self.assertEqual(report.status, "PASSED")
        self.assertEqual(report.passed_rules, 10)
        self.assertEqual(len(report.findings), 0)
        self.assertEqual(report.critical_count, 0)
        self.assertEqual(report.high_count, 0)

    def test_vulnerable_config_triggers_all_rules(self) -> None:
        """Verify vulnerable config triggers all 10 rules with 0/100 score."""
        fixture_path = REPO_ROOT / "fixtures" / "vulnerable_quant_config.json"
        with open(fixture_path, "r", encoding="utf-8") as f:
            policy = json.load(f)

        report = QuantizationAuditor.audit_policy(policy, manifest_path=str(fixture_path))

        self.assertEqual(report.score, 0)
        self.assertEqual(report.status, "FAILED")
        self.assertEqual(report.passed_rules, 0)
        self.assertEqual(len(report.findings), 10)
        self.assertEqual(report.critical_count, 3)
        self.assertEqual(report.high_count, 4)
        self.assertEqual(report.medium_count, 2)
        self.assertEqual(report.low_count, 1)

    def test_isolated_rule_triggers(self) -> None:
        """Verify isolated single rule violations compute exact deduction weights."""
        fixture_path = REPO_ROOT / "fixtures" / "hardened_quant_config.json"
        with open(fixture_path, "r", encoding="utf-8") as f:
            base_policy = json.load(f)

        # Test LH-QUANT-001 (CRITICAL, 25 deduction -> 75)
        p1 = dict(base_policy)
        p1["allow_custom_op_libraries"] = True
        rep1 = QuantizationAuditor.audit_policy(p1)
        self.assertEqual(rep1.score, 75)
        self.assertEqual(rep1.findings[0].rule_id, "LH-QUANT-001")

        # Test LH-QUANT-004 (HIGH, 10 deduction -> 90)
        p4 = dict(base_policy)
        p4["conv_per_channel_quantization"] = False
        rep4 = QuantizationAuditor.audit_policy(p4)
        self.assertEqual(rep4.score, 90)
        self.assertEqual(rep4.findings[0].rule_id, "LH-QUANT-004")

        # Test LH-QUANT-008 (MEDIUM, 4 deduction -> 96)
        p8 = dict(base_policy)
        p8["fold_conv_batchnorm"] = False
        rep8 = QuantizationAuditor.audit_policy(p8)
        self.assertEqual(rep8.score, 96)
        self.assertEqual(rep8.findings[0].rule_id, "LH-QUANT-008")

        # Test LH-QUANT-010 (LOW, 1 deduction -> 99)
        p10 = dict(base_policy)
        p10["check_nan_inf_weights"] = False
        rep10 = QuantizationAuditor.audit_policy(p10)
        self.assertEqual(rep10.score, 99)
        self.assertEqual(rep10.findings[0].rule_id, "LH-QUANT-010")

        # Combined: 1 CRITICAL + 1 HIGH = 35 deduction -> 65
        p_comb = dict(base_policy)
        p_comb["allow_custom_op_libraries"] = True
        p_comb["conv_per_channel_quantization"] = False
        rep_comb = QuantizationAuditor.audit_policy(p_comb)
        self.assertEqual(rep_comb.score, 65)
        self.assertEqual(len(rep_comb.findings), 2)


class TestExporters(unittest.TestCase):
    """Test cases for ANSI Table, Canonical JSON, and OASIS SARIF v2.1.0."""

    def setUp(self) -> None:
        fixture_path = REPO_ROOT / "fixtures" / "vulnerable_quant_config.json"
        with open(fixture_path, "r", encoding="utf-8") as f:
            policy = json.load(f)
        self.report = QuantizationAuditor.audit_policy(policy, manifest_path=str(fixture_path))

    def test_table_exporter(self) -> None:
        """Verify ANSI Table output content."""
        tbl = Exporters.format_table(self.report)
        self.assertIn("Logihertz ONNX Quantizer Lite", tbl)
        self.assertIn("LH-QUANT-001", tbl)
        self.assertIn("FAILED", tbl)

    def test_json_exporter(self) -> None:
        """Verify Canonical JSON output structure."""
        j_str = Exporters.format_json(self.report)
        data = json.loads(j_str)
        self.assertEqual(data["version"], "0.1.0")
        self.assertEqual(data["score"], 0)
        self.assertEqual(len(data["findings"]), 10)

    def test_sarif_exporter(self) -> None:
        """Verify OASIS SARIF v2.1.0 specification adherence."""
        s_str = Exporters.format_sarif(self.report)
        data = json.loads(s_str)
        self.assertEqual(data["version"], "2.1.0")
        self.assertIn("$schema", data)
        self.assertEqual(len(data["runs"]), 1)
        self.assertEqual(len(data["runs"][0]["results"]), 10)
        self.assertEqual(len(data["runs"][0]["tool"]["driver"]["rules"]), 10)


class TestCLISubprocess(unittest.TestCase):
    """Test cases for CLI executable bin/lh-onnx-quant exit codes and subcommands."""

    def setUp(self) -> None:
        self.cli_bin = str(REPO_ROOT / "bin" / "lh-onnx-quant")
        self.env = dict(os.environ)
        self.env["PYTHONPATH"] = str(SRC_DIR)

    def test_version_and_help_exit_zero(self) -> None:
        """Verify --version and --help return exit code 0."""
        r_ver = subprocess.run([self.cli_bin, "--version"], capture_output=True, text=True, env=self.env)
        self.assertEqual(r_ver.returncode, 0)
        self.assertIn("lh-onnx-quant 0.1.0", r_ver.stdout)

        r_help = subprocess.run([self.cli_bin, "--help"], capture_output=True, text=True, env=self.env)
        self.assertEqual(r_help.returncode, 0)
        self.assertIn("usage:", r_help.stdout)

    def test_scan_subcommand(self) -> None:
        """Verify scan subcommand returns exit 0 on pass, 1 on fail, 2 on error."""
        hardened = str(REPO_ROOT / "fixtures" / "hardened_quant_config.json")
        vulnerable = str(REPO_ROOT / "fixtures" / "vulnerable_quant_config.json")

        # Hardened -> exit 0
        r_pass = subprocess.run([self.cli_bin, "scan", hardened], capture_output=True, text=True, env=self.env)
        self.assertEqual(r_pass.returncode, 0)
        self.assertIn("PASSED", r_pass.stdout)

        # Vulnerable -> exit 1
        r_fail = subprocess.run([self.cli_bin, "scan", vulnerable], capture_output=True, text=True, env=self.env)
        self.assertEqual(r_fail.returncode, 1)
        self.assertIn("FAILED", r_fail.stdout)

        # Missing file -> exit 2
        r_err = subprocess.run([self.cli_bin, "scan", "nonexistent_file.json"], capture_output=True, text=True, env=self.env)
        self.assertEqual(r_err.returncode, 2)

    def test_quantize_subcommand(self) -> None:
        """Verify quantize subcommand with INT8 and INT4 options."""
        graph = str(REPO_ROOT / "fixtures" / "sample_vision_graph.json")

        r_q8 = subprocess.run([self.cli_bin, "quantize", graph, "--bits", "8"], capture_output=True, text=True, env=self.env)
        self.assertEqual(r_q8.returncode, 0)
        self.assertIn("SUCCESSFULLY QUANTIZED", r_q8.stdout)

        r_q4 = subprocess.run([self.cli_bin, "quantize", graph, "--bits", "4", "--group-size", "32"], capture_output=True, text=True, env=self.env)
        self.assertEqual(r_q4.returncode, 0)
        self.assertIn("INT4", r_q4.stdout)

    def test_calibrate_subcommand(self) -> None:
        """Verify calibrate subcommand with kl_entropy and percentile."""
        graph = str(REPO_ROOT / "fixtures" / "sample_vision_graph.json")

        r_calib = subprocess.run([self.cli_bin, "calibrate", graph, "--method", "kl_entropy"], capture_output=True, text=True, env=self.env)
        self.assertEqual(r_calib.returncode, 0)
        self.assertIn("CALIBRATION CONVERGED", r_calib.stdout)

    def test_fold_bn_subcommand(self) -> None:
        """Verify fold-bn subcommand."""
        graph = str(REPO_ROOT / "fixtures" / "sample_vision_graph.json")

        r_fold = subprocess.run([self.cli_bin, "fold-bn", graph], capture_output=True, text=True, env=self.env)
        self.assertEqual(r_fold.returncode, 0)
        self.assertIn("FOLDED SUCCESSFULLY", r_fold.stdout)

    def test_invalid_command_exits_two(self) -> None:
        """Verify invalid command returns exit 2."""
        r_bad = subprocess.run([self.cli_bin, "invalid_cmd"], capture_output=True, text=True, env=self.env)
        self.assertEqual(r_bad.returncode, 2)


if __name__ == "__main__":
    unittest.main()
