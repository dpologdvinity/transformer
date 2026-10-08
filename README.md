# Positional Encodings and Length Generalization

[![tests](https://github.com/dpologdvinity/transformer/actions/workflows/tests.yml/badge.svg)](https://github.com/dpologdvinity/transformer/actions/workflows/tests.yml)

A Transformer built from PyTorch primitives, used to run a small-scale replication of
[Kazemnejad et al., *The Impact of Positional Encoding on Length Generalization in Transformers* (NeurIPS 2023)](https://arxiv.org/abs/2305.19466)
on a laptop CPU.

**Question:** a decoder-only Transformer is trained on sequences of length ≤ 16. How well
does it handle longer sequences, depending on how it encodes position?

**Result:** no encoding generalizes far. Across 3 seeds, every model is at 0% exact match by length 24, 1.5× the longest training length, but **NoPE** (no positional encoding) lasts longest. It is the best encoding on both tasks at n = 18–20 (81% on copy and 87% on reverse at n = 18), still at 37% on reverse at n = 20, and the only one above zero at n = 22, even though it is the weakest model on reverse inside the training range (97% averaged over n = 1–16, 92% at n = 16). **RoPE** breaks first: 100% at n = 16, then 25% on copy one token later and near zero by n = 19. **ALiBi** depends on the task: the best encoding on copy at n = 17 (99%) but at 57% on reverse and near 0% one token later. **Sinusoidal** holds up just past training on reverse (95% at n = 17, 83% at n = 18) before falling off.

![Exact match vs. sequence length](results/length_generalization.png)

## Setup

| Setting | Value |
|---|---|
| Model | Decoder-only Transformer, d_model 64, 4 heads, 3 layers, d_ff 256, post-LN, no dropout |
| Positional encodings | none (NoPE), sinusoidal (absolute), RoPE, ALiBi |
| Tasks | **copy** `x → x` and **reverse** `x → reverse(x)` |
| Format | `[BOS, x1..xn, SEP, y1..yn, EOS]`, loss on `y1..yn, EOS` only |
| Vocabulary | 20 symbols + PAD, BOS, SEP, EOS |
| Training | n uniform in [1, 16], batch 64, 5,000 steps, AdamW lr 1e-3, warmup + cosine decay, grad clip 1.0 |
| Evaluation | every n in [1, 48], 256 sequences per length, same sequences for every model |
| Metric | exact match: all n symbols and EOS correct under greedy decoding |

Exact match is computed in one teacher-forced pass: a row is fully correct given the true
prefix exactly when greedy decoding would reproduce it. A unit test checks this against
step-by-step generation.

## Results

Exact match (%) at each length, mean ± std over 3 seeds, trained on n = 1–16 (from [`results/summary.md`](results/summary.md)). "Longest n ≥ 90%" is the largest n such that every length up to it scores at least 90%, averaged over seeds.

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

### What the model attends to

![Attention of the best pointer head on the reverse task](results/attention_reverse.png)

For each encoding, the head that puts the most attention on the input symbol it has to
output next, at the longest training length (n = 16) and at twice that (n = 32), for the seed-0 models.
At n = 16, RoPE and ALiBi show a clean anti-diagonal pointer; NoPE's and sinusoidal's blur over the middle of the sequence. At n = 32 the NoPE head keeps a sharp pointer for input positions up to about 20 and smears beyond it, and the sinusoidal head keeps a weaker one up to about 16; the RoPE pointer is gone entirely; and the ALiBi head still produces a diagonal, but at the wrong offset, stalling around input position 3. Each heatmap shows one example; the on-target score in each title is averaged over 64 sequences.

## Comparison with the paper

- **Agrees: RoPE extrapolates poorly.** RoPE has the sharpest cliff of the four on copy, and on reverse it and ALiBi both fall to about half at n = 17 and to near zero by n = 19. The paper groups RoPE (Rotary) and ALiBi with absolute encodings as poorly suited to length generalization, and notes that Rotary "performs more similarly to APE than to other relative schemes."
- **Mostly agrees: NoPE.** The paper finds NoPE outperforms every explicit encoding, on par with or ahead of T5's relative bias. Here NoPE is the best encoding on both tasks at n = 18–20 and has the longest tail (87.4 ± 5.5% on reverse at n = 18, 36.8 ± 7.3% at n = 20). Two differences: the tail is short, with every model at 0% by n = 24, and on reverse NoPE is the weakest model inside the training range.
- **Not isolated by the paper: sinusoidal just past training.** The paper finds its absolute encoding (also sinusoidal) generalizes poorly over test lengths up to twice the training length. Here sinusoidal is the best encoding on reverse one token past training (95.2 ± 6.5% at n = 17, well clear of RoPE and ALiBi at about 50%) and close behind NoPE one token later, a narrow window that aggregate results do not separate out.
- **ALiBi.** Poor on reverse here; the paper reports that ALiBi "underperforms with respect to T5's Relative Bias in most cases." On copy, ALiBi is the best encoding just past training, where every output's source token is the same distance back within a sequence: a pattern a distance-based bias can encode directly.
- **Not tested here: T5's relative bias**, the strongest explicit encoding in the paper.
- **Scale and setup.** The paper trains models with about 107M weights on ten tasks (copy and reverse among them), with training length 20, test lengths up to 40 and three seeds. This study uses about 150k parameters, two tasks and training length 16, so it tests whether the qualitative ranking shows up at small scale, not the exact numbers.

## Limitations

- **Scale.** About 150k parameters and 5,000 steps, against the paper's much larger
  models. The machine was shared with other training jobs, so the model and step budget
  were cut to fit.
- **Training steps.** The grid was first run at 3,000 steps, where NoPE on reverse
  averaged 94.7% in-distribution exact match, under the 95% bar, so every run was
  retrained at 5,000 steps. The extra steps mostly helped NoPE past the training length
  (57.9% → 87.4% on reverse at n = 18); RoPE and ALiBi still collapse within two tokens
  of the training length. The 3,000-step numbers are in
  [`results/summary.md` at commit `e6c3307`](https://github.com/dpologdvinity/transformer/blob/e6c3307/results/summary.md).
- **Three seeds.** Variance is high right at the boundary: RoPE on reverse at n = 17 is
  51.0 ± 40.7%, so rankings at a single length can flip between seeds.
- **Two tasks.** Copy and reverse only; the paper also covers arithmetic and other
  reasoning tasks.
- **In-distribution bar.** One of the 24 runs misses the 95% in-distribution bar: reverse with NoPE, seed 1 (93.6%). Every configuration's average clears it. NoPE on reverse also scores below its average at n = 16 (92.1%), so it enters the untrained lengths from a lower starting point than the other encodings.

## Inference: KV cache

`DecoderOnlyLM.generate` keeps each layer's keys and values, so every decoding step feeds only
the newest token instead of re-running the whole sequence. RoPE, ALiBi and the sinusoidal table
are offset by the cache length, and a test checks that cached and uncached generation produce
identical tokens (and logits within 1e-5) for every encoding.

Greedy generation time for the study's model (d_model 64, 3 layers), 16-token prompt, one CPU
thread, median of 3 runs ([`results/kv_cache.json`](results/kv_cache.json)):

| Batch | New tokens | Cached | Uncached | Speedup |
|---|---|---|---|---|
| 1 | 32 | 0.40 s | 0.75 s | 1.9× |
| 1 | 64 | 0.92 s | 1.26 s | 1.4× |
| 1 | 128 | 1.77 s | 3.36 s | 1.9× |
| 64 | 32 | 1.04 s | 6.01 s | 5.8× |
| 64 | 64 | 2.21 s | 22.46 s | 10.2× |
| 64 | 128 | 4.91 s | 100.95 s | 20.6× |

Without the cache, total work grows with the square of the generated length, so the gap widens
as sequences get longer. At batch size 1 this small model is dominated by per-step overhead, so
the gain is modest. Timings were taken on a laptop shared with other jobs, so treat them as
approximate.

## Reproduce

```bash
uv sync                                                    # Python 3.12, CPU PyTorch
uv run pytest                                              # 79 tests
uv run python -m scripts.sweep --seeds 0 1 2 --steps 5000  # results/runs/*.json
uv run python -m scripts.plot                              # figures + results/summary.md
uv run python -m scripts.export_web                        # results/web/study.json for the website
uv run python -m scripts.bench_kv_cache                    # results/kv_cache.json
```

A single run: `uv run python -m transformer.train --task reverse --pos-encoding rope --seed 0`.

## Repository layout

```
transformer/
  attention.py    scaled dot-product attention (+ additive bias), multi-head attention (+ RoPE hook)
  positional.py   sinusoidal table, RoPE, ALiBi
  blocks.py       feed-forward, encoder and decoder blocks, causal and padding masks
  seq2seq.py      the original encoder-decoder Transformer and greedy decoding
  decoder_lm.py   decoder-only model with a selectable positional encoding and a KV cache
  tasks.py        copy / reverse data
  train.py        training loop, per-length evaluation, single-run CLI
scripts/
  sweep.py        runs the grid in parallel, resumable
  plot.py         figures and summary table
  export_web.py   JSON for the interactive results page
  bench_kv_cache.py  cached vs uncached generation timing
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

## License

[MIT](LICENSE)
