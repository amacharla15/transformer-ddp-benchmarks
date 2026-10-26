import argparse
import csv
import os
import statistics
import timeit
from contextlib import nullcontext

import torch

from cs336_basics.model import BasicsTransformerLM
from cs336_basics.nn_utils import cross_entropy
from cs336_basics.optimizer import AdamW

MODEL_SIZES = {
    "tiny": dict(d_model=64, d_ff=256, num_layers=2, num_heads=4),  # for quick CPU tests
    "small": dict(d_model=768, d_ff=3072, num_layers=12, num_heads=12),
    "medium": dict(d_model=1024, d_ff=4096, num_layers=24, num_heads=16),
    "large": dict(d_model=1280, d_ff=5120, num_layers=36, num_heads=20),
    "xl": dict(d_model=2560, d_ff=10240, num_layers=32, num_heads=32),
}


def make_model(size, vocab_size, context_length, device):
    # build the weights directly on the device instead of on the CPU first
    with torch.device(device):
        return BasicsTransformerLM(vocab_size=vocab_size, context_length=context_length, rope_theta=10000.0, **MODEL_SIZES[size])


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def nvtx(name):
    # nvtx ranges only exist in CUDA builds of torch
    if torch.cuda.is_available():
        return torch.cuda.nvtx.range(name)
    return nullcontext()


def append_csv(path, row):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    new_file = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row))
        if new_file:
            writer.writeheader()
        writer.writerow(row)


def run(args, device):
    torch.manual_seed(0)
    model = make_model(args.size, args.vocab_size, args.context_length, device)
    optimizer = AdamW(model.parameters(), lr=1e-4)
    x = torch.randint(0, args.vocab_size, (args.batch_size, args.context_length), device=device)
    y = torch.randint(0, args.vocab_size, (args.batch_size, args.context_length), device=device)
    if args.bf16:
        autocast = torch.autocast(device_type=device, dtype=torch.bfloat16)
    else:
        autocast = nullcontext()

    times = {"forward": [], "backward": [], "optimizer": []}
    for step in range(args.warmup + args.steps):
        if step == args.warmup and device == "cuda":
            torch.cuda.reset_peak_memory_stats()
        with nvtx("warmup" if step < args.warmup else "step"):
            optimizer.zero_grad()
            sync()
            t0 = timeit.default_timer()
            with nvtx("forward"), autocast:
                logits = model(x)
                loss = cross_entropy(logits.reshape(-1, args.vocab_size), y.reshape(-1))
            sync()
            t1 = timeit.default_timer()
            with nvtx("backward"):
                loss.backward()
            sync()
            t2 = timeit.default_timer()
            with nvtx("optimizer"):
                optimizer.step()
            sync()
            t3 = timeit.default_timer()
        if step >= args.warmup:
            times["forward"].append((t1 - t0) * 1000)
            times["backward"].append((t2 - t1) * 1000)
            times["optimizer"].append((t3 - t2) * 1000)
    return times


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", default="small", choices=list(MODEL_SIZES))
    parser.add_argument("--vocab-size", type=int, default=10000)
    parser.add_argument("--context-length", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--csv", default="results/benchmark.csv")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    row = {"size": args.size, "dtype": "bf16" if args.bf16 else "fp32", "batch": args.batch_size, "context": args.context_length, "warmup": args.warmup}
    try:
        times = run(args, device)
        for name in ["forward", "backward", "optimizer"]:
            row[name + "_ms"] = round(statistics.mean(times[name]), 2)
            row[name + "_std"] = round(statistics.pstdev(times[name]), 2)
        row["peak_mem_gb"] = round(torch.cuda.max_memory_allocated() / 2**30, 2) if device == "cuda" else 0
    except torch.cuda.OutOfMemoryError:
        for name in ["forward_ms", "forward_std", "backward_ms", "backward_std", "optimizer_ms", "optimizer_std", "peak_mem_gb"]:
            row[name] = "OOM"
    append_csv(args.csv, row)
    print(row)


if __name__ == "__main__":
    main()
