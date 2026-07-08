"""Layer A: offline Quark/OCP MXFP4 checkpoint config parsing for
``amd/Kimi-K2.5-MXFP4`` (DeepSeekV3-style text config).

These are pure config/scheme-detection tests (no GPU, no weights): they assert
that SGLang's ``QuarkConfig`` recognizes the offline OCP MXFP4 quantization
config, routes routed experts / shared experts / the layer-0 dense MLP to the
MXFP4 execution path, and keeps the router ``mlp.gate``, attention projections,
``lm_head`` and vision modules out of MXFP4.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

import unittest

import torch

from sglang.srt.layers.quantization.quark.quark import QuarkConfig
from sglang.srt.layers.quantization.quark.schemes import (
    QuarkW4A4MXFP4,
    QuarkW4A4MXFp4MoE,
)
from sglang.srt.layers.quantization.quark.utils import should_ignore_layer
from sglang.test.test_utils import CustomTestCase

# Minimal Quark export config matching the amd/Kimi-K2.5-MXFP4 layout:
#   * routed experts, shared_experts and layers.0.mlp -> MXFP4 (global config)
#   * mlp.gate (router), self-attention, lm_head, vision/mm -> excluded (BF16)
_MXFP4_SPEC_WEIGHT = {
    "dtype": "fp4",
    "qscheme": "per_group",
    "group_size": 32,
    "is_dynamic": False,
    "scale_format": "e8m0",
    "ch_axis": -1,
    "round_method": "half_even",
    "observer_cls": "PerBlockMXObserver",
}
_MXFP4_SPEC_INPUT = {
    "dtype": "fp4",
    "qscheme": "per_group",
    "group_size": 32,
    "is_dynamic": True,
    "scale_format": "e8m0",
    "ch_axis": -1,
    "round_method": "half_even",
    "observer_cls": "PerBlockMXObserver",
}

KIMI_MXFP4_QUANT_CONFIG = {
    "quant_method": "quark",
    "packed_modules_mapping": {"gate_up_proj": ["gate_proj", "up_proj"]},
    "exclude": [
        "re:.*\\.gate$",  # MoE router gate (mlp.gate)
        "re:.*self_attn.*",  # attention projections
        "lm_head",
        "re:.*embed_tokens",
        "re:.*vision_tower.*",
        "re:.*mm_projector.*",
    ],
    "global_quant_config": {
        "weight": dict(_MXFP4_SPEC_WEIGHT),
        "input_tensors": dict(_MXFP4_SPEC_INPUT),
        "output_tensors": None,
        "bias": None,
    },
    "layer_quant_config": {},
    "layer_type_quant_config": {},
    "export": {"kv_cache_group": [], "pack_method": "reorder"},
}


class _FakeModule(torch.nn.Module):
    """Stand-in module for scheme lookup (only its type name is inspected)."""


class TestQuarkKimiMXFP4Config(CustomTestCase):
    def setUp(self):
        self.cfg = QuarkConfig.from_config(KIMI_MXFP4_QUANT_CONFIG)

    def test_recognized_as_quark_prequantized(self):
        self.assertEqual(self.cfg.get_name(), "quark")
        self.assertTrue(self.cfg.is_prequantized)

    def test_global_config_is_mxfp4(self):
        w = KIMI_MXFP4_QUANT_CONFIG["global_quant_config"]["weight"]
        i = KIMI_MXFP4_QUANT_CONFIG["global_quant_config"]["input_tensors"]
        self.assertTrue(self.cfg._is_mx_fp4(w, i))
        # It must NOT be misdetected as the W4A8 (fp8-activation) variant.
        self.assertFalse(self.cfg._is_mx_w4a8(w, i))

    def test_mxfp4_quant_params(self):
        w = KIMI_MXFP4_QUANT_CONFIG["global_quant_config"]["weight"]
        i = KIMI_MXFP4_QUANT_CONFIG["global_quant_config"]["input_tensors"]
        self.assertEqual(w["dtype"], "fp4")
        self.assertEqual(i["dtype"], "fp4")
        self.assertEqual(w["group_size"], 32)
        self.assertEqual(i["group_size"], 32)
        self.assertEqual(w["scale_format"], "e8m0")
        self.assertEqual(i["scale_format"], "e8m0")
        self.assertFalse(w["is_dynamic"])  # weights: static
        self.assertTrue(i["is_dynamic"])  # activations: dynamic

    def test_negative_mxfp4_detection(self):
        # A per-tensor fp8 spec must not be detected as MXFP4.
        fp8 = {
            "dtype": "fp8_e4m3",
            "qscheme": "per_tensor",
            "is_dynamic": False,
        }
        self.assertFalse(self.cfg._is_mx_fp4(fp8, fp8))
        # group_size != 32 must be rejected.
        bad = dict(_MXFP4_SPEC_WEIGHT)
        bad["group_size"] = 16
        self.assertFalse(self.cfg._is_mx_fp4(bad, _MXFP4_SPEC_INPUT))
        # non-e8m0 scale must be rejected.
        bad2 = dict(_MXFP4_SPEC_WEIGHT)
        bad2["scale_format"] = "e4m3"
        self.assertFalse(self.cfg._is_mx_fp4(bad2, _MXFP4_SPEC_INPUT))

    def test_linear_scheme_is_mxfp4(self):
        scheme = self.cfg._get_scheme_from_config(
            KIMI_MXFP4_QUANT_CONFIG["global_quant_config"]
        )
        self.assertIsInstance(scheme, QuarkW4A4MXFP4)
        self.assertTrue(scheme.is_checkpoint_mxfp4_serialized)

    def test_moe_scheme_is_mxfp4(self):
        scheme = self.cfg.get_moe_scheme(
            _FakeModule(), "model.layers.1.mlp.experts"
        )
        self.assertIsInstance(scheme, QuarkW4A4MXFp4MoE)
        self.assertTrue(scheme.is_checkpoint_mxfp4_serialized)

    def test_shared_experts_and_layer0_included(self):
        mapping = KIMI_MXFP4_QUANT_CONFIG["packed_modules_mapping"]
        exclude = self.cfg.exclude_layers
        included = [
            "model.layers.1.mlp.experts.3.down_proj",
            "model.layers.1.mlp.experts.3.gate_up_proj",
            "model.layers.1.mlp.shared_experts.gate_up_proj",
            "model.layers.1.mlp.shared_experts.down_proj",
            "model.layers.0.mlp.gate_up_proj",  # first dense MLP
            "model.layers.0.mlp.down_proj",
        ]
        for name in included:
            self.assertFalse(
                should_ignore_layer(name, ignore=exclude, fused_mapping=mapping),
                f"{name} should be MXFP4-quantized (not excluded)",
            )

    def test_router_attention_lmhead_excluded(self):
        mapping = KIMI_MXFP4_QUANT_CONFIG["packed_modules_mapping"]
        exclude = self.cfg.exclude_layers
        excluded = [
            "model.layers.1.mlp.gate",  # MoE router
            "model.layers.0.mlp.gate",
            "model.layers.1.self_attn.q_b_proj",
            "model.layers.1.self_attn.o_proj",
            "lm_head",
            "vision_tower.encoder.blocks.0.mlp.fc1",
            "mm_projector.linear_1",
        ]
        for name in excluded:
            self.assertTrue(
                should_ignore_layer(name, ignore=exclude, fused_mapping=mapping),
                f"{name} should be excluded from MXFP4",
            )


if __name__ == "__main__":
    unittest.main()
