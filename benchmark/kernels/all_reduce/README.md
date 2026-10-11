## MSCCL++ All-Reduce Benchmark

[MSCCL++](https://github.com/microsoft/mscclpp) is a GPU-driven communication library that can replace NCCL for all-reduce operations. It supports CUDA graph capture and is optimized for small-to-medium message sizes commonly seen in tensor-parallel inference.

Currently supported configurations: **TP=8** (single-node) and **TP=16** (two-node).

### Prerequisites

1. If you use the default SGLang Docker image build from `docker/Dockerfile`, [MSCCL++](https://github.com/microsoft/mscclpp) is already installed by default.
2. If you are not using that Docker image (or want to install manually), install [MSCCL++](https://github.com/microsoft/mscclpp) from source (requires CMake and a CUDA toolkit):
    ```bash
    git clone https://github.com/microsoft/mscclpp.git
    cd mscclpp && mkdir build && cd build
    cmake .. && make -j && pip install ..
    ```
3. Ensure `mscclpp` is importable in your Python environment before running the benchmark or using MSCCL++ for inference.

### Running the Benchmark

The benchmark compares all-reduce latency across torch/NCCL (eager), MSCCL++ (eager and graph), and PyNccl (graph) for power-of-two message sizes.

```bash
torchrun --nproc_per_node 8 \
    --nnodes 1 \
    --node_rank 0 \
    benchmark/kernels/all_reduce/benchmark_mscclpp.py
```

For multi-node (TP=16):
```bash
export WORLD_SIZE=2
export MASTER_ADDR=<master-ip>
export MASTER_PORT=12345

# Run on each node with the appropriate RANK (0 or 1):
torchrun --nproc_per_node 8 \
    --nnodes $WORLD_SIZE \
    --node_rank $RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT \
    benchmark/kernels/all_reduce/benchmark_mscclpp.py
```

### Inference with MSCCL++

Use the `--enable-mscclpp` flag to select MSCCL++ as the all-reduce backend during CUDA-graph-captured inference:

```bash
python -m sglang.launch_server \
    --model-path Qwen/Qwen3-8B \
    --tp-size 8 \
    --enable-mscclpp
```

> **Note:** MSCCL++ performs auto-tuning on first initialization, which may add a few seconds to startup time. The tuned configurations are cached for the lifetime of the process.

## Kimi-K3 prefill all-reduce (ROCm, TP8)

`benchmark_k3_prefill_ar.py` times RCCL, AITER custom AR (1-/2-stage and legacy),
sglang's 1-stage CA and quick-reduce (INT4/INT8/INT6/FP) at the bf16 shapes a
Kimi-K3 prefill chunk produces (`[N x 7168]` o_proj/dense/embedding and
`[N x 10752]` MoE latent|shared pair) and prints a latency table plus the backend
serving would pick today. Run it inside the serving env on 8 idle GPUs:

```bash
cd kimi_k3/agentx && bash -c 'source ./common.sh; export REPRO_CONC=bench DSPARK_BLOCK_SIZE=0; setup_server >/dev/null \
  && cd <worktree> && python3 -m torch.distributed.run --nproc_per_node 8 \
     benchmark/kernels/all_reduce/benchmark_k3_prefill_ar.py --rel-err --json-out /tmp/k3_prefill_ar.json'
```
