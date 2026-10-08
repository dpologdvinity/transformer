# Results

Exact match (%) on sequences of length n, mean ± std across seeds.
Models were trained on n = 1–16.

| Task | Encoding | Seeds | n = 1–16 | n = 16 | n = 17 | n = 18 | n = 20 | n = 24 | Longest n ≥ 90% |
|---|---|---|---|---|---|---|---|---|---|
| copy | NoPE | 3 | 96.5 ± 1.1 | 87.0 ± 5.6 | 74.0 ± 8.3 | 46.4 ± 8.2 | 3.0 ± 0.9 | 0.0 ± 0.0 | 15.0 |
| copy | Sinusoidal | 3 | 96.7 ± 1.9 | 85.0 ± 4.9 | 65.8 ± 7.5 | 29.2 ± 10.7 | 1.6 ± 1.7 | 0.0 ± 0.0 | 14.0 |
| copy | RoPE | 3 | 100.0 ± 0.0 | 100.0 ± 0.0 | 21.2 ± 17.7 | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.0 ± 0.0 | 16.0 |
| copy | ALiBi | 3 | 100.0 ± 0.0 | 100.0 ± 0.0 | 91.8 ± 5.2 | 31.9 ± 25.9 | 0.0 ± 0.0 | 0.0 ± 0.0 | 16.7 |
| reverse | NoPE | 3 | 94.7 ± 3.6 | 83.1 ± 11.9 | 76.6 ± 17.1 | 57.9 ± 16.3 | 10.0 ± 5.3 | 0.0 ± 0.0 | 12.3 |
| reverse | Sinusoidal | 3 | 99.3 ± 1.0 | 97.4 ± 3.7 | 93.1 ± 9.5 | 73.7 ± 15.8 | 3.9 ± 3.9 | 0.0 ± 0.0 | 17.0 |
| reverse | RoPE | 3 | 100.0 ± 0.0 | 100.0 ± 0.0 | 36.6 ± 43.0 | 1.0 ± 1.5 | 0.0 ± 0.0 | 0.0 ± 0.0 | 16.3 |
| reverse | ALiBi | 3 | 99.9 ± 0.1 | 99.1 ± 1.0 | 33.5 ± 22.7 | 0.4 ± 0.3 | 0.0 ± 0.0 | 0.0 ± 0.0 | 16.0 |

Runs below 95% in-distribution exact match: copy_sinusoidal_seed0, reverse_none_seed0, reverse_none_seed1.
