import ast
from pathlib import Path

MODEL_PATH = (
    Path(__file__).resolve().parents[4]
    / "python"
    / "sglang"
    / "srt"
    / "models"
    / "kimi_k3.py"
)
SOURCE = MODEL_PATH.read_text(encoding="utf-8")
TREE = ast.parse(SOURCE)


def _method_source(class_name: str, method_name: str) -> str:
    for node in TREE.body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for child in node.body:
                if isinstance(child, ast.FunctionDef) and child.name == method_name:
                    source = ast.get_source_segment(SOURCE, child)
                    assert source is not None
                    return source
    raise AssertionError(f"{class_name}.{method_name} not found")


def test_rocm_stream_pool_is_opt_in_per_site():
    source = _method_source("KimiK3LinearModel", "__init__")

    assert "envs.SGLANG_ROCM_USE_MULTI_STREAM.get()" in source
    assert "envs.SGLANG_ROCM_K3_MULTI_STREAM_ATTN.get()" in source
    assert "envs.SGLANG_ROCM_K3_MULTI_STREAM_MOE.get()" in source
    # CUDA keeps all three slots; ROCm never takes slot [1], whose only
    # K3-reachable use splits AITER's fused qk-rmsnorm into two launches.
    assert "[torch.cuda.Stream() for _ in range(3)]" in source
    hip = source.index("if _is_hip:")
    hip_branch = source[hip : source.index("\n        else:", hip)]
    assert hip_branch.count("_hip_alt_stream(") == 2
    assert "None," in hip_branch


def test_rocm_overlaps_compute_but_keeps_one_flat_collective():
    source = _method_source("KimiK3MoE", "_forward_fused")

    fork = source.index("self.alt_stream.wait_stream(current_stream)")
    shared_compute = source.index("self._forward_shared(gate_up, shared_output)", fork)
    join = source.index("current_stream.wait_stream(self.alt_stream)", shared_compute)
    flat_ar = source.index("buf = tensor_model_parallel_all_reduce(buf)", join)

    assert "num_tokens in self._rocm_overlap_tokens" in source
    assert fork < shared_compute < join < flat_ar
    # One collective over the flat [latent | shared] pair; splitting it into a
    # shared and a latent collective regressed MI355X decode by 20-23%.
    assert (
        "shared_output = tensor_model_parallel_all_reduce(shared_output)" not in source
    )
    assert "latent = tensor_model_parallel_all_reduce(latent)" not in source


def test_moe_overlap_token_window_is_read_once():
    source = _method_source("KimiK3MoE", "__init__")

    assert "envs.SGLANG_ROCM_K3_MULTI_STREAM_MIN_TOKENS.get()" in source
    assert "envs.SGLANG_ROCM_K3_MULTI_STREAM_MAX_TOKENS.get()" in source
