# Length Generalization Study — Design

## Goal

Turn the from-scratch encoder-decoder Transformer into a small research result:
a CPU-scale replication of Kazemnejad et al., *The Impact of Positional Encoding
on Length Generalization in Transformers* (NeurIPS 2023).

**Question:** In a decoder-only Transformer trained on sequences of length ≤ 16,
how does the choice of positional encoding affect accuracy on longer sequences?

**Success criteria**

- Every configuration reaches ≥ 95% exact-match on in-distribution lengths
  (otherwise the out-of-distribution numbers are meaningless). Configurations
  that miss this are reported, not hidden.
- Accuracy-vs-length curves for 4 encodings × 2 tasks × 3 seeds, committed as
  JSON and plots.
- A README that states the result, compares it to the paper, and lists
  limitations.
- A real pytest suite, including numerical checks against PyTorch's built-in
  attention.
- The whole pipeline reproduces on a laptop CPU in under an hour.

## Experimental setup

| Setting | Value |
|---|---|
| Architecture | Decoder-only, reusing `TransformerEncoderBlock` with a causal mask |
| Size | d_model 128, 4 heads, 4 layers, d_ff 512, dropout 0 |
| Encodings | none (NoPE), sinusoidal (absolute), RoPE, ALiBi |
| Tasks | copy, reverse |
| Symbol vocabulary | 20 symbols + PAD, BOS, SEP, EOS |
| Train lengths | n uniform in [1, 16] |
| Test lengths | every n in [1, 48], 256 samples per length |
| Seeds | 3 per (encoding, task) → 24 runs |
| Optimizer | AdamW, lr 1e-3 with linear warmup + cosine decay, batch 64 |
| Steps | fixed budget per run (initially 5,000; raised uniformly if any configuration misses the in-distribution bar) |

**Sequence format:** `[BOS, x1..xn, SEP, y1..yn, EOS]`, right-padded. Loss is
computed only on `y1..yn, EOS`. With causal attention and right padding, real
tokens never attend to padding, so no padding mask is needed.

**Metrics** (per length n, greedy decoding of n + 1 tokens after SEP):

- Exact match: all n output tokens and EOS correct.
- Token accuracy: fraction of the n output tokens correct.

Sinusoidal encodings are precomputed to 256 positions, so longer test inputs
work, but positions beyond ~35 are never seen in training. That gap is part
of what the study measures.

## Positional encodings

- **NoPE:** token embeddings only; position comes only from the causal mask.
- **Sinusoidal:** the existing `TokenAndPositionalEmbedding` formula, added
  to token embeddings.
- **RoPE:** rotate Q and K by position-dependent angles after splitting heads
  (base 10000, applied to the full head dimension).
- **ALiBi:** add `-m_h · (i − j)` to attention scores, with head slopes
  `m_h = 2^(−8h/H)`.

`scaled_dot_product_attention` gains an optional additive `bias`.
`MultiHeadAttention` gains an optional rotary module. Both default to off, so
the encoder-decoder model behaves exactly as before.

## Interpretability

For one seed per encoding, on the reverse task at n = 16 and n = 32, find the
head with the highest average attention weight on the correct source token
for each output position, and plot its attention map. The goal is to show
whether the "pointer" pattern survives past the training length.

## Repository layout

```
transformer/
  attention.py    scaled dot-product attention (+ bias), MultiHeadAttention (+ rotary)
  positional.py   sinusoidal embedding, RoPE, ALiBi bias
  blocks.py       feed-forward, encoder block, decoder block, causal mask
  seq2seq.py      original encoder-decoder Transformer + greedy decoding
  decoder_lm.py   decoder-only LM parameterized by encoding type
  tasks.py        copy / reverse batch generation and loss mask
  train.py        train one configuration, evaluate per length, write JSON
scripts/
  sweep.py        run the grid in parallel processes
  plot.py         accuracy-vs-length curves, attention heatmaps, summary table
tests/            pytest suite
results/          committed JSON, PNG and summary.md (checkpoints are gitignored)
```

`transformer.py` is split into the package, and its print-based shape checks
become pytest tests.

## Fixes to existing code

- `Transformer.forward` passes `tgt_mask` into the decoder's cross-attention
  slot, which expects the source padding mask. Pass `src_mask` instead, and
  add a padding-mask helper.
- Remove leftover assignment scaffolding (`TODO` comment, closing print).

## Tests

- Attention matches `F.scaled_dot_product_attention` with and without a causal
  mask; `MultiHeadAttention` matches `nn.MultiheadAttention` with copied
  weights.
- Causality: for each encoding, changing future tokens does not change logits
  at earlier positions.
- RoPE: preserves vector norms, and q·k depends only on relative offset.
- ALiBi: zero on the diagonal, more negative with distance, correct slopes.
- Tasks: format, reverse targets, and loss mask covers exactly `y, EOS`.
- Seq2seq: output shapes, and padded source tokens do not change outputs when
  masked.
- Smoke: a tiny model's loss decreases over a few dozen steps.

## Deliverables beyond the repo

Draft a replacement Transformer entry for the resume and portfolio, based on
the actual results, and present it for approval before editing either file.

## Out of scope

Learned absolute embeddings, T5 relative bias, beam search, addition and other
harder tasks, GPU support work. A KV cache with speedup measurements is a
stretch goal only if the core study finishes within the time budget.

## Risks

- **All encodings collapse past the training length.** That is consistent with
  follow-up work showing NoPE's advantage is small. Report where each one
  breaks down instead of a single headline number.
- **Post-LN training instability without dropout.** Mitigate with LR warmup.
  If runs still diverge, add a pre-norm option to the block and use it for all
  configurations.
- **CPU time.** Runs execute in parallel with 1–2 threads each. If the grid
  exceeds about 40 minutes, reduce test samples per length before reducing
  seeds.
