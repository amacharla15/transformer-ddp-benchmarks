import argparse
import datetime
import gc
import os
import statistics
import timeit

import torch
import torch.distributed as dist

from benchmark import MODEL_SIZES, append_csv, make_model, nvtx, sync
from cs336_basics.nn_utils import cross_entropy
from cs336_basics.optimizer import AdamW
from cs336_systems.ddp import FlatDDP, NaiveDDP, OverlapDDP
from cs336_systems.fsdp import FSDP
from cs336_systems.sharded_optimizer import ShardedOptimizer

MODES = ["naive", "flat", "overlap", "sharded", "fsdp"]
METRICS = ["step_ms", "step_std", "comm_ms", "comm_pct", "mem_init_gb", "peak_before_step_gb", "peak_after_step_gb"]


def gb(n):
    return round(n / 2**30, 2)


def run(args, rank, world_size, device):
    cuda = torch.cuda.is_available()
    torch.manual_seed(0)
    model = make_model(args.size, args.vocab_size, args.context_length, device)
    if args.mode == "naive":
        model = NaiveDDP(model)
    elif args.mode == "flat":
        model = FlatDDP(model)
    elif args.mode == "fsdp":
        model = FSDP(model)
    else:
        model = OverlapDDP(model)
    if args.mode == "sharded":
        optimizer = ShardedOptimizer(model.parameters(), AdamW, lr=1e-4)
    else:
        optimizer = AdamW(model.parameters(), lr=1e-4)
    mem_init = gb(torch.cuda.memory_allocated()) if cuda else 0

    # every rank gets its own part of the global batch
    local_batch = args.batch_size // world_size
    torch.manual_seed(1 + rank)
    x = torch.randint(0, args.vocab_size, (local_batch, args.context_length), device=device)
    y = torch.randint(0, args.vocab_size, (local_batch, args.context_length), device=device)

    step_times, comm_times = [], []
    peak_before, peak_after = 0, 0
    total = args.warmup + args.steps
    for step in range(total):
        last = step == total - 1
        if last and cuda:
            torch.cuda.reset_peak_memory_stats()
        optimizer.zero_grad(set_to_none=True)
        sync()
        t0 = timeit.default_timer()
        with nvtx("forward"):
            loss = cross_entropy(model(x).reshape(-1, args.vocab_size), y.reshape(-1))
        with nvtx("backward"):
            loss.backward()
        # wait for the compute stream only, overlapped all-reduces may still be running
        if cuda:
            torch.cuda.current_stream().synchronize()
        t1 = timeit.default_timer()
        with nvtx("comm"):
            model.finish_gradient_synchronization()
            sync()
        t2 = timeit.default_timer()
        if last and cuda:
            peak_before = gb(torch.cuda.max_memory_allocated())
        with nvtx("optimizer"):
            optimizer.step()
        sync()
        t3 = timeit.default_timer()
        if last and cuda:
            peak_after = gb(torch.cuda.max_memory_allocated())
        if step >= args.warmup:
            step_times.append((t3 - t0) * 1000)
            comm_times.append((t2 - t1) * 1000)

    return {
        "step_ms": statistics.mean(step_times),
        "step_std": statistics.pstdev(step_times),
        "comm_ms": statistics.mean(comm_times),
        "mem_init_gb": mem_init,
        "peak_before_step_gb": peak_before,
        "peak_after_step_gb": peak_after,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="naive", choices=MODES)
    parser.add_argument("--size", default="xl", choices=list(MODEL_SIZES))
    parser.add_argument("--vocab-size", type=int, default=10000)
    parser.add_argument("--context-length", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=4, help="global batch, split across GPUs")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--csv", default="results/ddp.csv")
    args = parser.parse_args()

    # torchrun sets RANK, WORLD_SIZE, LOCAL_RANK, MASTER_ADDR and MASTER_PORT
    local_rank = int(os.environ["LOCAL_RANK"])
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = f"cuda:{local_rank}"
        backend = "nccl"
    else:
        device = "cpu"
        backend = "gloo"
    dist.init_process_group(backend, timeout=datetime.timedelta(minutes=5))
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    try:
        result = run(args, rank, world_size, device)
    except torch.cuda.OutOfMemoryError:
        result = None
    # free whatever the failed run left behind before talking to the other ranks
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    results = [None] * world_size
    dist.all_gather_object(results, result)
    if rank == 0:
        row = {"mode": args.mode, "size": args.size, "gpus": world_size, "global_batch": args.batch_size, "context": args.context_length}
        if None in results:
            for name in METRICS:
                row[name] = "OOM"
        else:
            # times are averaged over ranks, memory is the worst rank
            step = statistics.mean(r["step_ms"] for r in results)
            comm = statistics.mean(r["comm_ms"] for r in results)
            row["step_ms"] = round(step, 2)
            row["step_std"] = round(max(r["step_std"] for r in results), 2)
            row["comm_ms"] = round(comm, 2)
            row["comm_pct"] = round(100 * comm / step, 1)
            for name in ["mem_init_gb", "peak_before_step_gb", "peak_after_step_gb"]:
                row[name] = max(r[name] for r in results)
        append_csv(args.csv, row)
        print(row)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
