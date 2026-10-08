"""
Turn sweep results into figures and a summary table.

    uv run python -m scripts.plot [--runs results/runs] [--checkpoints checkpoints] [--out results]
"""
import argparse
import json
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from transformer.decoder_lm import POSITIONAL_ENCODINGS
from transformer.tasks import TASKS, answer_query_positions, sample_fixed_length, source_positions
from transformer.train import RunConfig, build_model, in_distribution_exact_match, run_name

LABELS = {"none": "NoPE", "sinusoidal": "Sinusoidal", "rope": "RoPE", "alibi": "ALiBi"}
COLORS = {"none": "tab:blue", "sinusoidal": "tab:orange", "rope": "tab:green", "alibi": "tab:red"}
VALIDITY_BAR = 0.95


def load_runs(runs_dir):
    return [json.loads(p.read_text()) for p in sorted(Path(runs_dir).glob("*.json"))]


def group_runs(runs):
    """ {(task, pos_encoding): [run, ...]}. Runs in a group may differ only by seed. """
    groups = {}
    for r in runs:
        groups.setdefault((r["config"]["task"], r["config"]["pos_encoding"]), []).append(r)
    for (task, pe), group in groups.items():
        settings = [{k: v for k, v in r["config"].items() if k != "seed"} for r in group]
        if any(s != settings[0] for s in settings):
            raise ValueError(f"{task}/{pe} runs differ in more than the seed; refusing to average them")
    return groups


def exact_match(run, n):
    return run["eval"][str(n)]["exact_match"]


def longest_solved_length(run, threshold=0.9):
    """ Largest n such that every length 1..n reaches the threshold. """
    n = 0
    while str(n + 1) in run["eval"] and exact_match(run, n + 1) >= threshold:
        n += 1
    return n


def plot_length_curves(runs, out_path, train_max_len=16, max_len=32):
    groups = group_runs(runs)
    fig, axes = plt.subplots(1, len(TASKS), figsize=(11, 4), sharey=True)
    for ax, task in zip(axes, TASKS, strict=True):
        for pe in POSITIONAL_ENCODINGS:
            group = groups.get((task, pe))
            if not group:
                continue
            lengths = sorted(int(n) for n in group[0]["eval"])
            per_length = [[exact_match(r, n) for r in group] for n in lengths]
            ax.plot(lengths, [statistics.mean(v) for v in per_length], color=COLORS[pe], label=LABELS[pe])
            ax.fill_between(lengths, [min(v) for v in per_length], [max(v) for v in per_length],
                            color=COLORS[pe], alpha=0.15, linewidth=0)
        ax.axvline(train_max_len, color="gray", linestyle="--", linewidth=1)
        ax.text(train_max_len + 0.5, 0.03, "longest training length", color="gray", fontsize=8)
        ax.set_title(task.capitalize())
        ax.set_xlabel("Sequence length n")
        ax.set_ylim(-0.02, 1.02)
        ax.set_xlim(0, max_len)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("Exact match")
    axes[0].legend(loc="upper right")
    seeds = max(len(g) for g in groups.values())
    title = "Length generalization by positional encoding"
    if seeds > 1:
        title += f" (line: mean of {seeds} seeds, band: min–max)"
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _mean_std(values):
    return f"{100 * statistics.mean(values):.1f} ± {100 * statistics.pstdev(values):.1f}"


def summary_table(runs, report_lengths=(16, 17, 18, 20, 24)):
    """ Markdown table of exact match (%) as mean ± std across seeds. """
    groups = group_runs(runs)
    header = ["Task", "Encoding", "Seeds", "n = 1–16"] + [f"n = {n}" for n in report_lengths] + ["Longest n ≥ 90%"]
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for task in TASKS:
        for pe in POSITIONAL_ENCODINGS:
            group = groups.get((task, pe))
            if not group:
                continue
            cells = [task, LABELS[pe], str(len(group)), _mean_std([in_distribution_exact_match(r) for r in group])]
            cells += [_mean_std([exact_match(r, n) for r in group]) for n in report_lengths]
            cells.append(f"{statistics.mean(longest_solved_length(r) for r in group):.1f}")
            lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def below_validity_bar(runs):
    return [run_name(RunConfig(**r["config"])) for r in runs if in_distribution_exact_match(r) < VALIDITY_BAR]


@torch.no_grad()
def pointer_head(model, task, n, samples=64):
    """
    Teacher-forces full sequences and finds the head that puts the most attention,
    from each answer position, on the input symbol it has to output.
    :return: (layer, head, mean attention weight on the correct source)
    """
    prompt, answer = sample_fixed_length(task, samples, n, torch.Generator().manual_seed(0))
    model.eval()
    model(torch.cat([prompt, answer[:, :-1]], dim=1))
    queries, sources = answer_query_positions(n), source_positions(task, n)
    best = (0, 0, -1.0)
    for layer, weights in enumerate(model.attention_maps()):
        scores = weights[:, :, queries, sources].mean(dim=(0, 2))  # (num_heads,)
        head = int(scores.argmax())
        if scores[head] > best[2]:
            best = (layer, head, float(scores[head]))
    return best


def load_model(run, checkpoint_dir):
    config = RunConfig(**run["config"])
    model = build_model(config)
    model.load_state_dict(torch.load(Path(checkpoint_dir) / f"{run_name(config)}.pt"))
    return model.eval()


def plot_attention(runs, checkpoint_dir, out_path, task="reverse", lengths=(16, 32), seed=0):
    by_encoding = {r["config"]["pos_encoding"]: r for r in runs
                   if r["config"]["task"] == task and r["config"]["seed"] == seed}
    fig, axes = plt.subplots(len(POSITIONAL_ENCODINGS), len(lengths),
                             figsize=(4.2 * len(lengths), 3.6 * len(POSITIONAL_ENCODINGS)), squeeze=False)
    for row, pe in enumerate(POSITIONAL_ENCODINGS):
        if pe not in by_encoding:
            for ax in axes[row]:
                ax.axis("off")
            continue
        model = load_model(by_encoding[pe], checkpoint_dir)
        for col, n in enumerate(lengths):
            layer, head, score = pointer_head(model, task, n)
            weights = model.attention_maps()[layer][0, head]  # first sample of the batch above
            grid = weights[answer_query_positions(n)][:, 1:n + 1]
            ax = axes[row, col]
            ax.imshow(grid, cmap="viridis", vmin=0, vmax=1, aspect="auto")
            ax.set_title(f"{LABELS[pe]}, n = {n}\nlayer {layer}, head {head}, on-target {score:.2f}", fontsize=9)
            ax.set_xlabel("input position")
            ax.set_ylabel("output step")
    fig.suptitle(f"{task.capitalize()}: attention of the best 'pointer' head (trained on n ≤ 16)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Plot length-generalization results.")
    parser.add_argument("--runs", default="results/runs")
    parser.add_argument("--checkpoints", default="checkpoints")
    parser.add_argument("--out", default="results")
    args = parser.parse_args()

    torch.set_num_threads(1)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    runs = load_runs(args.runs)

    plot_length_curves(runs, out / "length_generalization.png")
    plot_attention(runs, args.checkpoints, out / "attention_reverse.png")

    summary = ["# Results", "", "Exact match (%) on sequences of length n, mean ± std across seeds.",
               "Models were trained on n = 1–16.", "", summary_table(runs), ""]
    failing = below_validity_bar(runs)
    summary.append(f"Runs below {VALIDITY_BAR:.0%} in-distribution exact match: "
                   + (", ".join(failing) if failing else "none") + ".")
    (out / "summary.md").write_text("\n".join(summary) + "\n")
    print("\n".join(summary))


if __name__ == "__main__":
    main()
