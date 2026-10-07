"""
Run the experiment grid (seed x task x positional encoding) in parallel processes.
Finished runs are skipped, so an interrupted sweep resumes by running it again.

    uv run python -m scripts.sweep --workers 6 --threads 1
"""
import argparse
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import torch

from transformer.decoder_lm import POSITIONAL_ENCODINGS
from transformer.tasks import TASKS
from transformer.train import RunConfig, in_distribution_exact_match, run, run_name


def build_grid(seeds=(0, 1, 2), **overrides):
    """ Seed-major order, so a sweep cut short still covers every (task, encoding). """
    return [RunConfig(task, pe, seed, **overrides)
            for seed in seeds for task in TASKS for pe in POSITIONAL_ENCODINGS]


def pending(configs, out_dir):
    """ Configs without a finished <run_name>.json (a leftover .json.tmp does not count). """
    return [c for c in configs if not (Path(out_dir) / f"{run_name(c)}.json").exists()]


def run_one(config, out_dir, checkpoint_dir, threads):
    torch.set_num_threads(threads)
    start = time.perf_counter()
    result = run(config, out_dir, checkpoint_dir)
    return time.perf_counter() - start, in_distribution_exact_match(result)


def main():
    parser = argparse.ArgumentParser(description="Run the length-generalization grid.")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--threads", type=int, default=1, help="torch CPU threads per worker")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--steps", type=int, help="override RunConfig.steps")
    parser.add_argument("--eval-samples", type=int, help="override RunConfig.eval_samples")
    parser.add_argument("--out", default="results/runs")
    parser.add_argument("--checkpoints", default="checkpoints")
    args = parser.parse_args()

    overrides = {k: v for k, v in {"steps": args.steps, "eval_samples": args.eval_samples}.items() if v is not None}
    todo = pending(build_grid(tuple(args.seeds), **overrides), args.out)
    print(f"{len(todo)} runs pending", flush=True)

    failures = 0
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(run_one, c, args.out, args.checkpoints, args.threads): c for c in todo}
        for future in as_completed(futures):
            name = run_name(futures[future])
            try:
                seconds, in_dist = future.result()
                print(f"done {name}: {seconds:.0f}s, in-distribution exact match {in_dist:.3f}", flush=True)
            except Exception as error:
                failures += 1
                print(f"FAILED {name}: {error}", flush=True)
    if failures:
        raise SystemExit(f"{failures} run(s) failed")


if __name__ == "__main__":
    main()
