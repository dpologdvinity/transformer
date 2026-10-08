"""
Export the study's results as one JSON file for the website's interactive page.

    uv run python -m scripts.export_web [--runs results/runs] [--checkpoints checkpoints] [--out results/web/study.json]
"""
import argparse
import json
import statistics
from pathlib import Path

import torch

from scripts.plot import group_runs, load_model, load_runs, pointer_head
from transformer.tasks import answer_query_positions
from transformer.train import in_distribution_exact_match

META_KEYS = ("train_min_len", "train_max_len", "eval_max_len", "eval_samples", "steps", "batch_size",
             "d_model", "num_heads", "num_layers", "d_ff", "lr")


def _round(x):
    return round(x, 4)


def _curve(group):
    lengths = sorted(int(n) for n in group[0]["eval"])
    per_length = [[r["eval"][str(n)]["exact_match"] for r in group] for n in lengths]
    in_dist = [in_distribution_exact_match(r) for r in group]
    return {
        "seeds": len(group),
        "n": lengths,
        "mean": [_round(statistics.mean(v)) for v in per_length],
        "min": [_round(min(v)) for v in per_length],
        "max": [_round(max(v)) for v in per_length],
        "in_distribution": {"mean": _round(statistics.mean(in_dist)), "min": _round(min(in_dist))},
    }


@torch.no_grad()
def _attention(run, checkpoint_dir, lengths, task="reverse"):
    """ Best pointer head's attention (answer steps x input positions) for the first sample. """
    model = load_model(run, checkpoint_dir)
    maps = {}
    for n in lengths:
        layer, head, score = pointer_head(model, task, n)
        weights = model.attention_maps()[layer][0, head][answer_query_positions(n)][:, 1:n + 1]
        maps[str(n)] = {"layer": layer, "head": head, "score": _round(score),
                        "weights": [[round(float(w), 3) for w in row] for row in weights]}
    return maps


def build_study(runs, checkpoint_dir, lengths=(16, 32)):
    groups = group_runs(runs)
    curves = {}
    for (task, pe), group in groups.items():
        curves.setdefault(task, {})[pe] = _curve(group)
    first = runs[0]["config"]
    study = {
        "meta": {**{k: first[k] for k in META_KEYS}, "seeds": max(len(g) for g in groups.values())},
        "curves": curves,
        "attention": {},
    }
    if checkpoint_dir is not None:
        for r in runs:
            if r["config"]["task"] == "reverse" and r["config"]["seed"] == 0:
                study["attention"][r["config"]["pos_encoding"]] = _attention(r, checkpoint_dir, lengths)
    return study


def main():
    parser = argparse.ArgumentParser(description="Export study data for the website.")
    parser.add_argument("--runs", default="results/runs")
    parser.add_argument("--checkpoints", default="checkpoints")
    parser.add_argument("--out", default="results/web/study.json")
    args = parser.parse_args()

    torch.set_num_threads(1)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(build_study(load_runs(args.runs), args.checkpoints), separators=(",", ":")) + "\n")
    print(f"wrote {out} ({out.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
