import unittest
from types import SimpleNamespace

from sglang.srt.arg_groups.kimi_k3_rocm_hook import disable_kimi_k3_rocm_symm_mem
from sglang.srt.arg_groups.overrides import declare_resolution, resolution_result
from sglang.srt.model_executor.cuda_graph_config import Backend, CudaGraphConfig
from sglang.srt.runtime_context import override_platform
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _kimi_record(*, decode: str, prefill: str) -> ServerArgs:
    server_args = ServerArgs(model_path="dummy", enable_symm_mem=True)
    declare_resolution(
        server_args,
        "fixture",
        cuda_graph_config=CudaGraphConfig.from_dict(
            {"decode": {"backend": decode}, "prefill": {"backend": prefill}}
        ),
    )
    server_args._model_config = SimpleNamespace(
        hf_config=SimpleNamespace(architectures=["KimiK3ForConditionalGeneration"])
    )
    return server_args


class TestKimiK3RocmSymmMem(CustomTestCase):
    def test_graphs_on_turn_symm_mem_off(self):
        """--enable-symm-mem with a captured phase must resolve to off on ROCm
        instead of aborting resolution on the model-config lookup."""
        server_args = _kimi_record(decode=Backend.FULL, prefill=Backend.DISABLED)
        with override_platform(is_hip=True):
            disable_kimi_k3_rocm_symm_mem(server_args)
        self.assertFalse(resolution_result(server_args, "enable_symm_mem"))

    def test_fully_eager_keeps_symm_mem(self):
        server_args = _kimi_record(decode=Backend.DISABLED, prefill=Backend.DISABLED)
        with override_platform(is_hip=True):
            disable_kimi_k3_rocm_symm_mem(server_args)
        self.assertTrue(resolution_result(server_args, "enable_symm_mem"))


if __name__ == "__main__":
    unittest.main()
