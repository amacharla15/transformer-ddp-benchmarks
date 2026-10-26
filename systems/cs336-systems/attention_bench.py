import argparse
import timeit

import torch
import triton

from benchmark import append_csv
from cs336_basics.model import scaled_dot_product_attention
from cs336_systems.flash_attention import FlashAttentionTriton


def pytorch_attention(q, k, v):
    n = q.shape[-2]
    mask = torch.tril(torch.ones(n, n, dtype=torch.bool, device=q.device))
    return scaled_dot_product_attention(q, k, v, mask)


def triton_attention(q, k, v):
    return FlashAttentionTriton.apply(q, k, v, True)


IMPLS = {
    "pytorch": pytorch_attention,
    "compiled": torch.compile(pytorch_attention),
    "triton": triton_attention,
}


def time_ms(fn, grads):
    if torch.cuda.is_available():
        return triton.testing.do_bench(fn, grad_to_none=grads)
    # rough CPU timing, only so the script can be tried without a GPU
    start = timeit.default_timer()
    for _ in range(3):
        for t in grads:
            t.grad = None
        fn()
    return (timeit.default_timer() - start) / 3 * 1000


def bench(fn, q, k, v, do):
    # run once first so compiling is not timed
    fn(q, k, v).backward(do)
    forward = time_ms(lambda: fn(q, k, v), [q, k, v])
    both = time_ms(lambda: fn(q, k, v).backward(do), [q, k, v])
    # compiled graphs can't run backward twice on one forward, so backward = both - forward
    return forward, both - forward, both


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq-lens", default="512,1024,2048,4096,8192,16384,32768")
    parser.add_argument("--dims", default="64,128")
    parser.add_argument("--dtype", default="bf16", choices=["bf16", "fp32"])
    parser.add_argument("--impls", default="pytorch,compiled,triton")
    parser.add_argument("--csv", default="results/attention.csv")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32
    for d in [int(s) for s in args.dims.split(",")]:
        for n in [int(s) for s in args.seq_lens.split(",")]:
            for name in args.impls.split(","):
                torch.manual_seed(0)
                # batch 1, no head dimension, causal, like the handout
                q, k, v = (torch.randn(1, n, d, device=device, dtype=dtype, requires_grad=True) for _ in range(3))
                do = torch.randn(1, n, d, device=device, dtype=dtype)
                row = {"impl": name, "seq_len": n, "d": d, "dtype": args.dtype}
                try:
                    forward, backward, both = bench(IMPLS[name], q, k, v, do)
                    row["forward_ms"] = round(forward, 3)
                    row["backward_ms"] = round(backward, 3)
                    row["fwd_bwd_ms"] = round(both, 3)
                except torch.cuda.OutOfMemoryError:
                    row["forward_ms"] = row["backward_ms"] = row["fwd_bwd_ms"] = "OOM"
                del q, k, v, do
                if device == "cuda":
                    torch.cuda.empty_cache()
                append_csv(args.csv, row)
                print(row)


if __name__ == "__main__":
    main()
