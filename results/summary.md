# Results

Exact match (%) on sequences of length n, mean ± std across seeds.
Models were trained on n = 1–16.

| Task | Encoding | Seeds | n = 1–16 | n = 17 | n = 18 | n = 20 | n = 24 | Longest n ≥ 90% |
|---|---|---|---|---|---|---|---|---|
| copy | NoPE | 1 | 95.8 ± 0.0 | 69.9 ± 0.0 | 43.8 ± 0.0 | 2.3 ± 0.0 | 0.0 ± 0.0 | 15.0 |
| copy | Sinusoidal | 1 | 94.1 ± 0.0 | 56.2 ± 0.0 | 16.4 ± 0.0 | 0.0 ± 0.0 | 0.0 ± 0.0 | 12.0 |
| copy | RoPE | 1 | 100.0 ± 0.0 | 43.4 ± 0.0 | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.0 ± 0.0 | 16.0 |
| copy | ALiBi | 1 | 100.0 ± 0.0 | 98.8 ± 0.0 | 67.2 ± 0.0 | 0.0 ± 0.0 | 0.0 ± 0.0 | 17.0 |
| reverse | NoPE | 1 | 92.2 ± 0.0 | 55.9 ± 0.0 | 36.3 ± 0.0 | 5.9 ± 0.0 | 0.0 ± 0.0 | 11.0 |
| reverse | Sinusoidal | 1 | 97.8 ± 0.0 | 79.7 ± 0.0 | 52.7 ± 0.0 | 1.6 ± 0.0 | 0.0 ± 0.0 | 16.0 |
| reverse | RoPE | 1 | 100.0 ± 0.0 | 12.9 ± 0.0 | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.0 ± 0.0 | 16.0 |
| reverse | ALiBi | 1 | 100.0 ± 0.0 | 30.9 ± 0.0 | 0.4 ± 0.0 | 0.0 ± 0.0 | 0.0 ± 0.0 | 16.0 |

Runs below 95% in-distribution exact match: copy_sinusoidal_seed0, reverse_none_seed0.
