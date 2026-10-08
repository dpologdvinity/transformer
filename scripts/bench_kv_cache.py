"""
Time greedy generation with and without the KV cache.

    uv run python -m scripts.bench_kv_cache [--out results/kv_cache.json]

Cached and uncached runs alternate within each repetition so both see the same
background load; the reported time is the median over repetitions.
"""
import argparse
import json
import statistics
import time
from pathlib import Path

import torch

from transformer.tasks import VOCAB_SIZE
from transformer.train import RunConfig, build_model


def time_generation(model, prompt, num_new_tokens, use_cache):
    start = time.perf_counter()
    model.generate(prompt, num_new_tokens, use_cache=use_cache)
    return time.perf_counter() - start


def benchmark(model, prompt_len, new_tokens, batch_size, reps):
    """ :return: one row per generated length with median seconds and the speedup. """
    prompt = torch.randint(4, VOCAB_SIZE, (batch_size, prompt_len), generator=torch.Generator().manual_seed(0))
    rows = []
    for n in new_tokens:
        times = {True: [], False: []}
        for _ in range(reps):
            for use_cache in (True, False):
                times[use_cache].append(time_generation(model, prompt, n, use_cache))
        cached, uncached = statistics.median(times[True]), statistics.median(times[False])
        row = {"prompt_len": prompt_len, "new_tokens": n, "batch_size": batch_size,
               "cached_s": round(cached, 4), "uncached_s": round(uncached, 4),
               "speedup": round(uncached / cached, 2)}
        rows.append(row)
        print(f"batch {batch_size:>2}  {n:>3} new tokens: cached {row['cached_s']:.3f}s  "
              f"uncached {row['uncached_s']:.3f}s  speedup {row['speedup']:.1f}x", flush=True)
    return rows


def main():
    parser = argparse.ArgumentParser(description="Benchmark KV-cache generation.")
    parser.add_argument("--out", default="results/kv_cache.json")
    parser.add_argument("--reps", type=int, default=5)
    parser.add_argument("--new-tokens", type=int, nargs="+", default=[32, 64, 128, 224])
    args = parser.parse_args()

    torch.set_num_threads(1)
    config = RunConfig("copy", "rope", 0)  # the study's model size; weights do not affect timing
    torch.manual_seed(0)
    model = build_model(config).eval()
    rows = []
    for batch_size in (1, 64):
        rows += benchmark(model, prompt_len=16, new_tokens=tuple(args.new_tokens),
                          batch_size=batch_size, reps=args.reps)
    result = {"model": {k: getattr(config, k) for k in ("d_model", "num_heads", "num_layers", "d_ff")},
              "threads": 1, "reps": args.reps, "rows": rows}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
