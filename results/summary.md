# Results

Exact match (%) on sequences of length n, mean ± population standard deviation across seeds.
Models were trained on n = 1–16.

| Task | Encoding | Seeds | n = 1–16 | n = 16 | n = 17 | n = 18 | n = 20 | n = 24 | Longest n ≥ 90% |
|---|---|---|---|---|---|---|---|---|---|
| copy | NoPE | 3 | 99.8 ± 0.2 | 99.0 ± 1.2 | 97.0 ± 2.6 | 81.1 ± 4.7 | 7.6 ± 4.6 | 0.0 ± 0.0 | 17.0 |
| copy | Sinusoidal | 3 | 99.5 ± 0.2 | 97.0 ± 0.4 | 93.1 ± 2.6 | 58.6 ± 9.3 | 3.5 ± 1.7 | 0.0 ± 0.0 | 16.7 |
| copy | RoPE | 3 | 100.0 ± 0.0 | 100.0 ± 0.0 | 25.0 ± 23.0 | 0.1 ± 0.2 | 0.0 ± 0.0 | 0.0 ± 0.0 | 16.0 |
| copy | ALiBi | 3 | 100.0 ± 0.0 | 100.0 ± 0.0 | 98.6 ± 1.0 | 53.4 ± 12.9 | 0.4 ± 0.3 | 0.0 ± 0.0 | 17.0 |
| reverse | NoPE | 3 | 97.3 ± 2.7 | 92.1 ± 6.7 | 91.3 ± 6.2 | 87.4 ± 5.5 | 36.8 ± 7.3 | 0.0 ± 0.0 | 15.3 |
| reverse | Sinusoidal | 3 | 99.3 ± 0.9 | 97.8 ± 2.9 | 95.2 ± 6.5 | 83.2 ± 16.8 | 7.7 ± 7.3 | 0.0 ± 0.0 | 17.3 |
| reverse | RoPE | 3 | 100.0 ± 0.0 | 100.0 ± 0.0 | 51.0 ± 40.7 | 11.2 ± 15.8 | 0.0 ± 0.0 | 0.0 ± 0.0 | 16.3 |
| reverse | ALiBi | 3 | 100.0 ± 0.0 | 100.0 ± 0.0 | 56.8 ± 28.7 | 0.3 ± 0.2 | 0.0 ± 0.0 | 0.0 ± 0.0 | 16.3 |

Runs below 95% in-distribution exact match: reverse_none_seed1.
