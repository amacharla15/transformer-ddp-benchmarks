Goal: Find the largest batch size for Qwen3-8B on this GPU where time per output token stays under 20 ms, with 2048-token prompts and 128 output tokens.

| batch | roofline TPOT (ms) | measured TPOT (ms) | efficiency | vs 20 ms target |
|---|---|---|---|---|
| 64 | 10.47 | 13.39 | 78% | under |
| 96 | 13.45 | 16.81 | 80% | under |
| 112 | 14.94 | 18.53 | 81% | under |
| 128 | 16.42 | 20.18 | 81% | borderline |

The largest batch size safely under the 20 ms TPOT target is 112 with a measured TPOT of 18.53 ms. There is also a borderline batch at 128 with 20.18 ms TPOT. Both batches are decode memory bound.
