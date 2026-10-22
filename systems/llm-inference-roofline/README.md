# LLM inference roofline: predict, measure, compare

How fast should Qwen3-8B run on an H100? I predict it with roofline math,
measure it with vLLM, and look at the gap. A small agent uses the math to
decide which batch sizes are worth measuring for a latency target.

## The idea

Every forward pass moves bytes from GPU memory (weights, KV cache) and does
math. The slower of the two sets the time:

```
time = max(bytes / memory_bandwidth, flops / peak_flops)
```

Qwen3-8B on an H100 SXM (3.35 TB/s, 989 dense bf16 TFLOP/s), from `python roofline.py`:

- **Decode, batch 1.** Every new token reads all 15.1 GB of weights.
  15.1 GB / 3.35 TB/s = 4.5 ms per token, so at most ~220 tokens/s. Memory-bound.
- **Bigger batches.** The weights are read once per step no matter how many
  sequences are in the batch, so batch 64 costs 6.1 ms per step instead of 4.5 ms
  but produces 64 tokens. Decode stays memory-bound until the batch reaches
  989 / 3.35 ≈ **295** (the ridge point).
- **Prefill, 2048-token prompt.** About 30 trillion operations / 989 TFLOP/s
  ≈ **30 ms**. Compute-bound.

```
batch   TTFT ms  TPOT ms  tokens/s  decode   fits      (512-token prompts, 128 output tokens)
    1      7.27     4.54       220  memory   True
    8     58.17     4.72      1694  memory   True
   32    232.68     5.33      6003  memory   True
   64    465.35     6.14     10417  memory   True
  128    930.71     7.77     16475  memory   True
  256   1861.42    11.02     23230  memory   True
```

TTFT = time to first token (prefill of the whole batch).
TPOT = time per output token (one decode step).

## Files

| file | what it does |
|---|---|
| `roofline.py` | the math: bytes and FLOPs per forward pass, TTFT, TPOT, ridge point, memory fit |
| `measure.py` | runs Qwen3-8B in vLLM and measures TTFT and TPOT for a grid of batch sizes and prompt lengths |
| `compare.py` | predicted vs measured table and plot |
| `agent.py` | OpenAI tool-calling agent: predicts across batch sizes, measures only the ones near a latency target, writes a report |

## Running it (one GPU pod, I used RunPod)

```bash
pip install -r requirements.txt

python roofline.py --gpu H100-SXM              # predictions only, no GPU needed
python measure.py                              # ~10 min, saves results/<gpu>.json
python compare.py results/NVIDIA_H100_80GB_HBM3.json

export OPENAI_API_KEY=...
python agent.py                                # saves results/agent_report.md
```

How `measure.py` gets the two numbers: time a run that generates 1 token per
prompt (that is the prefill, TTFT), then a run that generates 128 tokens;
TPOT = (difference) / 127. Prefix caching is off so repeated runs really prefill.

## The agent

Default goal: *largest batch size where TPOT stays under 20 ms, with 2048-token
prompts and 128 output tokens.*

It has two tools, `predict` (the math, instant) and `measure` (a real vLLM run,
about a minute). It predicts a range of batch sizes, then measures only the 2-3
closest to 20 ms, since that is where the math can be wrong about which side of
the line a batch falls on. Measurements are capped at 4.

Every number in its final report has to come from a tool result. If it writes
a number it never got from a tool (for example, a percentage it worked out
itself), the report is sent back once to be fixed. `results/agent_log.json`
keeps every tool call.

## Results

Added after the H100 run: `results/compare.md`, `results/compare.png`,
`results/agent_report.md`.

## What the math leaves out

- Kernels don't reach datasheet peak bandwidth or FLOP/s.
- Fixed costs per step: kernel launches, scheduling, sampling.
- Attention, norms and activations are not modeled op by op; the whole
  forward pass is one roofline.
- When a batch's prompts don't fit in one prefill step (vLLM's
  `max_num_batched_tokens`), the last few decode steps run with a smaller batch.

The gap between the dashed (math) and solid (measured) lines in
`compare.png` is the cost of everything on this list.
