# CS336 Assignment 2 (Systems), self-study

My implementation of the systems part of Stanford CS336 (Spring 2026), Assignment 2:
data parallel training (naive, flat and overlapped DDP), optimizer state sharding,
FSDP, FlashAttention-2 in PyTorch and Triton, and benchmarks for all of it.
This is self-study, not a course submission.

Credit: the Transformer and AdamW in `cs336-basics/` and the tests in `tests/` are from
[stanford-cs336/assignment2-systems](https://github.com/stanford-cs336/assignment2-systems)
(MIT license, see `cs336-basics/LICENSE`). I only filled in `tests/adapters.py`.

## Files

| file | what it does |
|---|---|
| `cs336_systems/ddp.py` | `NaiveDDP` all-reduces each gradient after backward, `FlatDDP` all-reduces all gradients as one flat tensor, `OverlapDDP` starts an async all-reduce from a hook as soon as each gradient is ready and `finish_gradient_synchronization()` waits for them before the optimizer step. All three broadcast rank 0's weights at the start. |
| `cs336_systems/sharded_optimizer.py` | wraps any optimizer so each rank keeps optimizer state for only its share of the parameters, steps those, then broadcasts them to the other ranks |
| `cs336_systems/fsdp.py` | shards every Linear and Embedding weight across ranks, all-gathers a layer's weight right before it is needed (forward and backward) and frees it after, reduce-scatters its gradient. Norm weights stay replicated and are all-reduced. Optional `compute_dtype` for low precision weights. |
| `cs336_systems/flash_attention.py` | FlashAttention-2 as a tiled PyTorch `autograd.Function` and as a Triton forward kernel; both use a backward that recomputes P from Q, K and the logsumexp (compiled with `torch.compile` for the Triton version) |
| `benchmark.py` | forward, backward and optimizer step time for one model size on one GPU, fp32 or bf16 autocast, NVTX ranges, peak memory |
| `attention_bench.py` | PyTorch attention vs `torch.compile` vs Triton FlashAttention-2, forward and backward, causal, batch 1 |
| `ddp_bench.py` | run with torchrun: step time, communication time and memory for naive, flat and overlap DDP, overlap DDP + sharded optimizer, and FSDP |
| `make_tables.py` | prints every `results/*.csv` as a markdown table |
| `run_gpu.sh` | installs everything and runs all tests, benchmarks and profiles on a 2x H100 pod |
| `tests/` | Stanford's tests; `adapters.py` connects them to my code |

## Running

On a RunPod pod with 2x H100:

```
git clone https://github.com/amacharla15/transformer-ddp-benchmarks
cd transformer-ddp-benchmarks
git checkout cs336-systems
cd systems/cs336-systems
bash run_gpu.sh
```

Everything goes to `results/`: `benchmark.csv`, `attention.csv`, `ddp.csv`, the test logs,
`profiles/ddp_naive.nsys-rep` and `profiles/ddp_overlap.nsys-rep` (open them in Nsight Systems),
and `tables.md` with the tables below filled in. A config that runs out of memory gets `OOM`
in its row instead of numbers.

Tests on a machine without a GPU (Linux, DDP / sharded optimizer / FSDP run on CPU with gloo,
the Triton tests get skipped):

```
pip install torch -r requirements.txt
export PYTHONPATH=$PWD:$PWD/cs336-basics
pytest tests/
DDP_KIND=naive pytest tests/test_ddp.py   # or flat; the default is overlap
```

## How things are measured

- Single GPU: vocab 10000, batch 4, context 512, 5 warmup steps, then mean and std over 10 steps,
  with `torch.cuda.synchronize()` after every phase. Peak memory is `max_memory_allocated` over the measured steps.
- DDP: xl model, 2 GPUs, global batch 4 (2 per GPU), fp32, AdamW from `cs336-basics`.
  `step_ms` is the whole step. `comm_ms` is the time from the end of the backward compute until all
  gradient communication is done: for naive and flat that is all of it, for overlap, sharded and fsdp
  it is only the part not hidden behind backward. FSDP's all-gathers during forward count in `step_ms` only.
  Times are averaged over the two ranks, memory is the larger of the two.
- `mem_init_gb` is memory allocated after building the model and optimizer; `peak_before_step_gb` and
  `peak_after_step_gb` are the peak during the last step up to just before and just after `optimizer.step()`.
- Attention: `triton.testing.do_bench`. Backward is (forward + backward) minus forward, because a
  `torch.compile`d graph can't run backward twice on one forward.

## Results

Not run yet. `python make_tables.py` prints these from `results/`.

Single GPU (`benchmark.csv`, batch 4, context 512)

| size | dtype | forward ms | backward ms | optimizer ms | peak memory GB |
|---|---|---|---|---|---|
| small | fp32 | | | | |
| small | bf16 | | | | |
| medium | fp32 | | | | |
| medium | bf16 | | | | |
| large | fp32 | | | | |
| large | bf16 | | | | |
| xl | fp32 | | | | |
| xl | bf16 | | | | |

Attention (`attention.csv`, bf16, causal, batch 1)

| impl | seq_len | d | forward ms | backward ms | forward + backward ms |
|---|---|---|---|---|---|
| pytorch | | | | | |
| compiled | | | | | |
| triton | | | | | |

DDP on 2 GPUs (`ddp.csv`, xl, global batch 4)

| mode | step ms | comm ms | comm % | memory after init GB | peak before step GB | peak after step GB |
|---|---|---|---|---|---|---|
| naive | | | | | | |
| flat | | | | | | |
| overlap | | | | | | |
| sharded | | | | | | |
| fsdp | | | | | | |
