"""Unit tests for QuarkConfig — CPU-only, no model loading."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import unittest
from unittest.mock import patch

from sglang.srt.layers.quantization.quark.quark import QuarkConfig
from sglang.test.test_utils import CustomTestCase

_GET_CAP = "sglang.srt.layers.quantization.quark.quark.get_device_capability"


def _bare_config() -> QuarkConfig:
    """Skip __init__ — _check_scheme_supported reads no instance attributes."""
    return QuarkConfig.__new__(QuarkConfig)


def _quark_config(
    *,
    global_quant_config: dict | None = None,
    layer_quant_config: dict | None = None,
    exclude: list[str] | None = None,
    is_prequantized: bool = True,
) -> QuarkConfig:
    return QuarkConfig(
        {
            "packed_modules_mapping": {},
            "exclude": exclude or [],
            "global_quant_config": global_quant_config or {},
            "layer_quant_config": layer_quant_config or {},
            "layer_type_quant_config": {},
        },
        is_prequantized=is_prequantized,
    )


def _mxfp4_spec() -> dict:
    return {
        "weight": {
            "dtype": "fp4",
            "qscheme": "per_group",
            "group_size": 32,
            "is_dynamic": False,
            "scale_format": "e8m0",
        },
        "input_tensors": {
            "dtype": "fp4",
            "qscheme": "per_group",
            "group_size": 32,
            "is_dynamic": True,
            "scale_format": "e8m0",
        },
    }


class TestCheckSchemeSupportedError(CustomTestCase):
    """Regression for `RuntimeError("a", "b", "c")` being passed three args.

    Bug: `_check_scheme_supported` raised `RuntimeError` with three positional
    string fragments. `RuntimeError.__str__` formats `self.args` as a tuple
    when `len(args) != 1`, so the user saw
        ('Quantization scheme is not supported for ', 'the current GPU…', 'Current capability: 70.')
    instead of a sentence. Fix: pass one already-joined message.
    """

    def test_error_is_single_argument(self):
        # The structural assertion that catches the bug regardless of wording.
        with patch(_GET_CAP, return_value=(7, 0)):  # capability = 70 < 200
            with self.assertRaises(RuntimeError) as ctx:
                _bare_config()._check_scheme_supported(min_capability=200)
        err = ctx.exception
        self.assertEqual(
            len(err.args),
            1,
            f"RuntimeError must carry a single joined message, got {err.args!r}",
        )

    def test_error_message_renders_as_sentence(self):
        with patch(_GET_CAP, return_value=(7, 0)):
            with self.assertRaises(RuntimeError) as ctx:
                _bare_config()._check_scheme_supported(min_capability=200)
        msg = str(ctx.exception)
        # Tuple-repr leakage shows up as a leading '(' and quote-comma joins.
        self.assertFalse(
            msg.startswith("("),
            f"error message starts with '(' (tuple repr leaked): {msg!r}",
        )
        self.assertNotIn(
            "', '",
            msg,
            f"error message contains tuple-style fragment join: {msg!r}",
        )

    def test_error_message_content(self):
        with patch(_GET_CAP, return_value=(7, 0)):
            with self.assertRaises(RuntimeError) as ctx:
                _bare_config()._check_scheme_supported(min_capability=200)
        msg = str(ctx.exception)
        self.assertIn("Quantization scheme is not supported", msg)
        self.assertIn("Min capability: 200", msg)
        self.assertIn("Current capability: 70", msg)

    # ---- Guardrails: unchanged code paths ---------------------------------

    def test_unsupported_returns_false_when_error_disabled(self):
        with patch(_GET_CAP, return_value=(7, 0)):
            ok = _bare_config()._check_scheme_supported(min_capability=200, error=False)
        self.assertFalse(ok)

    def test_supported_returns_true(self):
        with patch(_GET_CAP, return_value=(8, 0)):  # capability = 80 >= 70
            ok = _bare_config()._check_scheme_supported(min_capability=70)
        self.assertTrue(ok)

    def test_no_device_returns_false(self):
        with patch(_GET_CAP, return_value=None):
            ok = _bare_config()._check_scheme_supported(min_capability=70)
        self.assertFalse(ok)


class TestKimiSharedExpertQuantCompatibility(CustomTestCase):
    def test_recognizes_prequantized_mxfp4_checkpoint(self):
        config = _quark_config(global_quant_config=_mxfp4_spec())

        self.assertTrue(config.is_mxfp4_checkpoint)

    def test_non_prequantized_config_is_not_checkpoint_mxfp4(self):
        config = _quark_config(
            global_quant_config=_mxfp4_spec(),
            is_prequantized=False,
        )

        self.assertFalse(config.is_mxfp4_checkpoint)

    def test_kimi_plural_shared_expert_path_must_match_routed_spec(self):
        routed = _mxfp4_spec()
        shared = _mxfp4_spec()
        shared["weight"] = dict(shared["weight"], dtype="bf16")
        config = _quark_config(
            global_quant_config=routed,
            layer_quant_config={
                "language_model.model.layers.0.mlp.experts": routed,
                "language_model.model.layers.0.mlp.shared_experts.gate_proj": shared,
            },
        )

        self.assertFalse(config.can_fuse_shared_expert())

    def test_kimi_plural_shared_expert_path_allows_matching_spec(self):
        routed = _mxfp4_spec()
        config = _quark_config(
            global_quant_config=routed,
            layer_quant_config={
                "language_model.model.layers.0.mlp.experts": routed,
                "language_model.model.layers.0.mlp.shared_experts.gate_proj": routed,
            },
        )

        self.assertTrue(config.can_fuse_shared_expert())


if __name__ == "__main__":
    unittest.main()
