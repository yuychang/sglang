import unittest
from types import SimpleNamespace

from sglang.srt.models.deepseek_v2 import DeepseekV2ForCausalLM
from sglang.srt.runtime_context import get_context, reset_context
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=4, suite="base-a-test-cpu")


class TestDeepseekV2SharedExpertFusionPolicy(unittest.TestCase):
    def setUp(self):
        self._saved_server_args = get_context()._server_args

    def tearDown(self):
        if self._saved_server_args is None:
            reset_context()
        else:
            get_context().set_server_args(self._saved_server_args)

    def test_quantization_mismatch_disables_shared_expert_fusion(self):
        server_args = ServerArgs(model_path="dummy")
        server_args.enforce_shared_experts_fusion = False
        get_context().set_server_args(server_args)

        model = SimpleNamespace(
            config=SimpleNamespace(
                architectures=["DeepseekV3ForCausalLM"],
                hidden_size=7168,
                n_routed_experts=384,
                n_shared_experts=1,
            ),
            quant_config=SimpleNamespace(can_fuse_shared_expert=lambda: False),
        )

        DeepseekV2ForCausalLM.determine_num_fused_shared_experts(model)

        self.assertEqual(model.num_fused_shared_experts, 0)
        self.assertTrue(server_args.disable_shared_experts_fusion)


if __name__ == "__main__":
    unittest.main()
