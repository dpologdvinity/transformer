# Positional Encodings and Length Generalization

A Transformer built from PyTorch primitives, used to run a small-scale replication of
[Kazemnejad et al., *The Impact of Positional Encoding on Length Generalization in Transformers* (NeurIPS 2023)](https://arxiv.org/abs/2305.19466)
on a laptop CPU.

**Question:** a decoder-only Transformer is trained on sequences of length ≤ 16. How well
does it handle longer sequences, depending on how it encodes position?

**Result:** no encoding generalizes far. Every model is at 0% exact match by length 23, less than 1.5× the longest training length. How they fail differs: **RoPE** collapses immediately (100% at n = 16, then 43% on copy and 13% on reverse at n = 17, 0% at n = 18); **ALiBi** holds best one step out on copy (99% at n = 17) but not on reverse (31%); **NoPE** starts slipping earliest (already 82% on copy and 70% on reverse at n = 16) but declines the most slowly, keeping a non-zero tail out to n = 22.

![Exact match vs. sequence length](results/length_generalization.png)

## Setup

| Setting | Value |
|---|---|
| Model | Decoder-only Transformer, d_model 64, 4 heads, 3 layers, d_ff 256, post-LN, no dropout |
| Positional encodings | none (NoPE), sinusoidal (absolute), RoPE, ALiBi |
| Tasks | **copy** `x → x` and **reverse** `x → reverse(x)` |
| Format | `[BOS, x1..xn, SEP, y1..yn, EOS]`, loss on `y1..yn, EOS` only |
| Vocabulary | 20 symbols + PAD, BOS, SEP, EOS |
| Training | n uniform in [1, 16], batch 64, 3,000 steps, AdamW lr 1e-3, warmup + cosine decay, grad clip 1.0 |
| Evaluation | every n in [1, 48], 256 sequences per length, same sequences for every model |
| Metric | exact match: all n symbols and EOS correct under greedy decoding |

Exact match is computed in one teacher-forced pass: a row is fully correct given the true
prefix exactly when greedy decoding would reproduce it. A unit test checks this against
step-by-step generation.

## Results

Exact match (%) at each length, trained on n = 1–16 (one seed, so ± is 0; from [`results/summary.md`](results/summary.md)). "Longest n ≥ 90%" is the largest n such that every length up to it scores at least 90%.

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

### What the model attends to

![Attention of the best pointer head on the reverse task](results/attention_reverse.png)

For each encoding, the head that puts the most attention on the input symbol it has to
output next, at the longest training length (n = 16) and at twice that (n = 32).
At n = 16, NoPE, RoPE and ALiBi show a clean anti-diagonal pointer; sinusoidal's is smeared over the middle of the sequence. At n = 32 the NoPE and sinusoidal heads largely keep that pointer for input positions inside the trained range (roughly the first 16) and smear beyond it; the RoPE pointer is gone entirely; and the ALiBi head still produces a diagonal, but at the wrong offset, stalling around input position 4–5. Each heatmap shows one example; the on-target score in each title averages 64.

## Comparison with the paper

- **Agrees: RoPE extrapolates poorly.** The paper finds Rotary behaves more like absolute encodings than other relative schemes. Here it has the sharpest cliff of the four on both tasks.
- **Partly agrees: NoPE.** The paper ranks NoPE best overall. Here NoPE has the most gradual decline and the longest tail, but that decline begins inside the training range, it trails just past training (sinusoidal leads on reverse and ALiBi on copy at n = 17–18), and it only leads or ties from n = 19 on, where every model is already at or below about 20%. NoPE also fit the training lengths less completely (see below).
- **Mixed: ALiBi.** Poor on reverse, as in the paper, but the strongest encoding one or two steps past the training length on copy, where every output's source token is the same distance back within a sequence, a pattern a distance-based bias can encode directly.
- **Scale.** The paper trains much larger models on many more tasks, so these runs test whether its qualitative ranking shows up at small scale, not its exact numbers.

## Limitations

- **Scale.** About 150k parameters and 3,000 steps, against the paper's much larger
  models. The machine was shared with other training jobs, so the model and step budget
  were cut to fit.
- **One seed.** Each configuration was trained once, so differences of a few points are
  within run-to-run noise.
- **Two tasks.** Copy and reverse only; the paper also covers arithmetic and other
  reasoning tasks.
- **In-distribution bar.** Two of the eight runs miss the 95% in-distribution bar: copy with sinusoidal (94.1%) and reverse with NoPE (92.2%). Their curves are shown anyway; some of their gradual decline past n = 16 may come from not fully fitting the training lengths rather than from better generalization. Copy with NoPE passes the bar on average (95.8%) but scores 82.4% at n = 16, so the same caveat applies to it.

## Reproduce

```bash
uv sync                                                    # Python 3.12, CPU PyTorch
uv run pytest                                              # 61 tests
uv run python -m scripts.sweep --seeds 0 --steps 3000      # results/runs/*.json
uv run python -m scripts.plot                              # figures + results/summary.md
```

A single run: `uv run python -m transformer.train --task reverse --pos-encoding rope --seed 0`.

## Repository layout

```
transformer/
  attention.py    scaled dot-product attention (+ additive bias), multi-head attention (+ RoPE hook)
  positional.py   sinusoidal table, RoPE, ALiBi
  blocks.py       feed-forward, encoder and decoder blocks, causal and padding masks
  seq2seq.py      the original encoder-decoder Transformer and greedy decoding
  decoder_lm.py   decoder-only model with a selectable positional encoding
  tasks.py        copy / reverse data
  train.py        training loop, per-length evaluation, single-run CLI
scripts/
  sweep.py        runs the grid in parallel, resumable
  plot.py         figures and summary table
tests/            pytest suite
results/          per-run JSON, figures, summary.md
```

## Background: the encoder-decoder

The project started as a from-scratch encoder-decoder Transformer ("Attention Is All You
Need"): multi-head attention, sinusoidal positional encoding, causal and padding masks,
cross-attention and greedy decoding, built only from PyTorch tensor operations. The test
suite checks the attention against `torch.nn.functional.scaled_dot_product_attention` and
`torch.nn.MultiheadAttention` numerically. The decoder-only model for this study reuses
the same blocks.
