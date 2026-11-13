# ==============================================================================
# Product:       Logihertz ONNX Quantizer Lite (lh-onnx-quantizer-lite)
# Module:        src/lh_onnx_quantizer_lite/graph_optimizer.py
# Role:          Conv2D+BatchNorm Parameter Folding & ONNX QDQ Graph Synthesizer
# Organization:  Logihertz Systems (OPC) Private Limited
# Developer:     Sameer Tawade (Creator, Founder, Director & Sole Developer)
# Contact:       ops@logihertz.com / security@logihertz.com
# License:       Apache-2.0
# ==============================================================================

"""Analytical Conv2D+BatchNorm folding and ONNX QDQ transformation engine."""

from __future__ import annotations

import copy
import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Sequence, Tuple

from lh_onnx_quantizer_lite.calibration import CalibrationEngine
from lh_onnx_quantizer_lite.quant_engine import QuantEngine


@dataclass
class FoldedConvBNResult:
    """Result of analytical Conv2D + BatchNorm parameter fusion."""
    folded_weights: List[List[float]]      # [C_out, K] fused weights
    folded_bias: List[float]               # [C_out] fused bias
    alpha_scales: List[float]              # [C_out] gamma / sqrt(var + eps)
    max_divergence: float                  # Maximum divergence between fused & unfused


class GraphOptimizer:
    """Graph optimizer for neural network vision subgraphs and QDQ syntheses."""

    @staticmethod
    def fold_conv_batchnorm_weights(
        conv_weights: Sequence[Sequence[float]],
        conv_bias: Sequence[float] | None,
        bn_mean: Sequence[float],
        bn_var: Sequence[float],
        bn_gamma: Sequence[float],
        bn_beta: Sequence[float],
        eps: float = 1e-5,
    ) -> FoldedConvBNResult:
        """Mathematically folds BatchNorm parameters into Conv2D weights and biases.

        alpha_c = gamma_c / sqrt(var_c + eps)
        W_fold[c] = W[c] * alpha_c
        b_fold[c] = (b[c] - mean_c) * alpha_c + beta_c
        """
        c_out = len(conv_weights)
        if not (len(bn_mean) == len(bn_var) == len(bn_gamma) == len(bn_beta) == c_out):
            raise ValueError(
                f"Dimension mismatch: conv C_out={c_out}, bn_mean={len(bn_mean)}, "
                f"bn_var={len(bn_var)}, bn_gamma={len(bn_gamma)}, bn_beta={len(bn_beta)}"
            )

        conv_b = list(conv_bias) if conv_bias is not None else [0.0] * c_out
        if len(conv_b) != c_out:
            raise ValueError(f"conv_bias length ({len(conv_b)}) must match C_out ({c_out})")

        alpha_scales: List[float] = []
        folded_w: List[List[float]] = []
        folded_b: List[float] = []

        for c in range(c_out):
            alpha = bn_gamma[c] / math.sqrt(bn_var[c] + eps)
            alpha_scales.append(alpha)

            # Fold weights for channel c
            f_w_c = [w * alpha for w in conv_weights[c]]
            folded_w.append(f_w_c)

            # Fold bias for channel c
            f_b_c = (conv_b[c] - bn_mean[c]) * alpha + bn_beta[c]
            folded_b.append(f_b_c)

        return FoldedConvBNResult(
            folded_weights=folded_w,
            folded_bias=folded_b,
            alpha_scales=alpha_scales,
            max_divergence=0.0,
        )

    @classmethod
    def verify_conv_bn_equivalence(
        cls,
        x_vec: Sequence[float],
        conv_weights: Sequence[Sequence[float]],
        conv_bias: Sequence[float] | None,
        bn_mean: Sequence[float],
        bn_var: Sequence[float],
        bn_gamma: Sequence[float],
        bn_beta: Sequence[float],
        eps: float = 1e-5,
    ) -> float:
        """Verifies that fused Conv(x) equals unfused BatchNorm(Conv(x)).

        Returns maximum absolute difference across all output channels.
        """
        folded = cls.fold_conv_batchnorm_weights(
            conv_weights, conv_bias, bn_mean, bn_var, bn_gamma, bn_beta, eps
        )

        c_out = len(conv_weights)
        conv_b = list(conv_bias) if conv_bias is not None else [0.0] * c_out

        max_diff = 0.0
        for c in range(c_out):
            w_row = conv_weights[c]
            # 1. Unfused forward
            conv_out = sum(w * x for w, x in zip(w_row, x_vec)) + conv_b[c]
            bn_out = ((conv_out - bn_mean[c]) / math.sqrt(bn_var[c] + eps)) * bn_gamma[c] + bn_beta[c]

            # 2. Fused forward
            fused_w_row = folded.folded_weights[c]
            fused_out = sum(fw * x for fw, x in zip(fused_w_row, x_vec)) + folded.folded_bias[c]

            diff = abs(bn_out - fused_out)
            if diff > max_diff:
                max_diff = diff

        return max_diff

    @classmethod
    def quantize_vision_graph(
        cls,
        graph_dict: Dict[str, Any],
        bits: int = 8,
        group_size: int = 32,
        preserve_fp16_detection_head: bool = True,
        calibration_method: str = "kl_entropy",
    ) -> Dict[str, Any]:
        """Transform FP32 graph by folding Conv+BN pairs and inserting QDQ nodes."""
        out_graph = copy.deepcopy(graph_dict)
        nodes = out_graph.get("nodes", [])
        initializers = out_graph.get("initializers", {})

        # 1. Identify and fold Conv + BatchNormalization pairs
        nodes_to_remove: List[int] = []
        for i, node in enumerate(nodes):
            if node.get("op_type") == "Conv":
                conv_out = node.get("outputs", [""])[0]
                # Look for subsequent BatchNormalization node reading conv_out
                for j in range(i + 1, len(nodes)):
                    next_node = nodes[j]
                    if (
                        next_node.get("op_type") == "BatchNormalization"
                        and next_node.get("inputs", [""])[0] == conv_out
                    ):
                        # Perform parameter folding
                        w_name = node.get("inputs", ["", ""])[1]
                        b_name = node.get("inputs", ["", "", ""])[2] if len(node.get("inputs", [])) > 2 else ""

                        gamma_name = next_node.get("inputs", ["", ""])[1]
                        beta_name = next_node.get("inputs", ["", "", ""])[2]
                        mean_name = next_node.get("inputs", ["", "", "", ""])[3]
                        var_name = next_node.get("inputs", ["", "", "", "", ""])[4]

                        if (
                            w_name in initializers
                            and gamma_name in initializers
                            and beta_name in initializers
                            and mean_name in initializers
                            and var_name in initializers
                        ):
                            w_data = initializers[w_name]
                            b_data = initializers.get(b_name, [0.0] * len(w_data))
                            gamma_data = initializers[gamma_name]
                            beta_data = initializers[beta_name]
                            mean_data = initializers[mean_name]
                            var_data = initializers[var_name]

                            folded = cls.fold_conv_batchnorm_weights(
                                conv_weights=w_data,
                                conv_bias=b_data,
                                bn_mean=mean_data,
                                bn_var=var_data,
                                bn_gamma=gamma_data,
                                bn_beta=beta_data,
                            )

                            # Update initializers
                            folded_w_name = f"{w_name}_folded"
                            folded_b_name = f"{b_name or w_name}_bias_folded"

                            initializers[folded_w_name] = folded.folded_weights
                            initializers[folded_b_name] = folded.folded_bias

                            # Update Conv node inputs and outputs
                            node["inputs"] = [node["inputs"][0], folded_w_name, folded_b_name]
                            node["outputs"] = next_node.get("outputs", [conv_out])

                            # Mark BatchNormalization node for removal
                            nodes_to_remove.append(j)
                        break

        # Remove folded BN nodes in reverse order
        for idx in sorted(set(nodes_to_remove), reverse=True):
            del nodes[idx]

        # 2. Insert QuantizeLinear / DequantizeLinear (QDQ) nodes
        qdq_nodes: List[Dict[str, Any]] = []
        for node in nodes:
            op_type = node.get("op_type")
            node_name = node.get("name", "")

            # Check if this node is in the detection head (NMS, BBox regression, Softmax)
            is_head = any(k in node_name.lower() or k in op_type.lower() for k in ["detection", "nms", "softmax", "bbox", "head"])
            if is_head and preserve_fp16_detection_head:
                node["precision"] = "FP16_PRESERVED"
                qdq_nodes.append(node)
                continue

            if op_type in ["Conv", "MatMul", "Gemm"]:
                w_name = node.get("inputs", ["", ""])[1]
                if w_name in initializers:
                    raw_w = initializers[w_name]
                    # Flatten if nested
                    flat_w: List[float] = []
                    if isinstance(raw_w[0], list):
                        for row in raw_w:
                            flat_w.extend(row)
                    else:
                        flat_w = list(raw_w)

                    if bits == 4:
                        q_res = QuantEngine.quantize_group_wise_int4(flat_w, group_size=group_size)
                        scale_name = f"{w_name}_scale"
                        initializers[scale_name] = q_res.group_scales
                        initializers[f"{w_name}_packed_int4"] = list(q_res.packed_bytes)
                        node["weight_precision"] = "INT4_GROUPWISE"
                        node["group_size"] = group_size
                        node["sqnr_db"] = q_res.sqnr_db
                    else:
                        # Symmetric per-channel INT8
                        if isinstance(raw_w[0], list):
                            matrix_w = raw_w
                        else:
                            # 1 channel fallback
                            matrix_w = [flat_w]
                        q_res_8 = QuantEngine.quantize_per_channel_int8(matrix_w)
                        scale_name = f"{w_name}_per_channel_scale"
                        initializers[scale_name] = q_res_8.scales
                        initializers[f"{w_name}_quant_int8"] = q_res_8.quantized_weights
                        node["weight_precision"] = "INT8_PER_CHANNEL"
                        node["sqnr_db"] = q_res_8.sqnr_db

                # Insert QDQ on activation input
                act_in = node["inputs"][0]
                q_act_node = {
                    "name": f"{act_in}_QuantizeLinear",
                    "op_type": "QuantizeLinear",
                    "inputs": [act_in, f"{act_in}_scale", f"{act_in}_zero_point"],
                    "outputs": [f"{act_in}_quantized"],
                }
                dq_act_node = {
                    "name": f"{act_in}_DequantizeLinear",
                    "op_type": "DequantizeLinear",
                    "inputs": [f"{act_in}_quantized", f"{act_in}_scale", f"{act_in}_zero_point"],
                    "outputs": [f"{act_in}_dequantized"],
                }
                node["inputs"][0] = f"{act_in}_dequantized"
                qdq_nodes.extend([q_act_node, dq_act_node, node])
            else:
                qdq_nodes.append(node)

        out_graph["nodes"] = qdq_nodes
        out_graph["initializers"] = initializers
        out_graph["quantization_format"] = f"QDQ_INT{bits}"

        # 3. Compute deterministic model SHA-256 digest
        canon_json = json.dumps(out_graph, sort_keys=True, indent=2).encode("utf-8")
        out_graph["model_sha256"] = hashlib.sha256(canon_json).hexdigest()

        return out_graph
