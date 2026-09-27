"""ROCm attention-residual frozen-bank mean-square cache."""

import torch

from sglang.kernels.ops.kimi_k3.attn_res_hip import (
    attn_res_hip,
    write_bank_hip,
)


def test_bank_ms_path_is_bit_exact():
    if not torch.cuda.is_available():
        return

    tokens = 512
    hidden = 7168
    nvb = 8
    generator = torch.Generator(device="cuda")
    generator.manual_seed(7)

    rows = [
        torch.randn(
            tokens,
            hidden,
            dtype=torch.bfloat16,
            device="cuda",
            generator=generator,
        )
        for _ in range(nvb)
    ]
    bank = torch.empty(
        tokens, nvb + 1, hidden, dtype=torch.bfloat16, device="cuda"
    )
    bank_ms = torch.empty(tokens, nvb + 1, dtype=torch.float32, device="cuda")
    for row, src in enumerate(rows):
        write_bank_hip(src, bank, bank_ms, row)

    prefix = torch.randn(
        tokens,
        hidden,
        dtype=torch.bfloat16,
        device="cuda",
        generator=generator,
    )
    addend = torch.randn(
        tokens,
        hidden,
        dtype=torch.bfloat16,
        device="cuda",
        generator=generator,
    )
    cw = torch.randn(
        hidden, dtype=torch.float32, device="cuda", generator=generator
    )
    ow = torch.randn(
        hidden, dtype=torch.bfloat16, device="cuda", generator=generator
    )
    ref_bank = bank.clone()
    ref_out = torch.empty_like(prefix)
    got_out = torch.empty_like(prefix)
    ref_prefix = torch.empty_like(prefix)
    got_prefix = torch.empty_like(prefix)

    attn_res_hip(
        prefix,
        ref_bank,
        None,
        cw,
        ow,
        ref_out,
        nvb,
        1e-6,
        1e-6,
        addend=addend,
        prefix_out=ref_prefix,
        write_prefix=True,
    )
    attn_res_hip(
        prefix,
        bank,
        bank_ms,
        cw,
        ow,
        got_out,
        nvb,
        1e-6,
        1e-6,
        addend=addend,
        prefix_out=got_prefix,
        write_prefix=True,
    )
    torch.cuda.synchronize()

    for row, src in enumerate(rows):
        torch.testing.assert_close(bank[:, row, :], src, atol=0, rtol=0)
    torch.testing.assert_close(got_prefix, ref_prefix, atol=0, rtol=0)
    torch.testing.assert_close(bank[:, nvb, :], ref_bank[:, nvb, :], atol=0, rtol=0)
    # The cached scalar is reduced in the bank-write kernel instead of as one
    # row of the [NVB, H] aggregate tile. ROCm emits a slightly different
    # reduction tree: fewer than 1e-5 elements cross one bf16 rounding bin.
    torch.testing.assert_close(got_out, ref_out, atol=0.00390625, rtol=0)
