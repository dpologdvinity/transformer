"""
Train one decoder-only model on one task, then measure accuracy at every test length.

    uv run python -m transformer.train --task reverse --pos-encoding rope --seed 0
"""
import argparse
import json
import math
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn.functional as F

from transformer.decoder_lm import POSITIONAL_ENCODINGS, DecoderOnlyLM
from transformer.tasks import TASKS, VOCAB_SIZE, sample_batch, sample_fixed_length

# Every run is evaluated on the same prompts: one generator per length, seeded from this
EVAL_SEED = 1234


@dataclass
class RunConfig:
    task: str
    pos_encoding: str
    seed: int
    d_model: int = 64
    num_heads: int = 4
    num_layers: int = 3
    d_ff: int = 256
    dropout: float = 0.0
    batch_size: int = 64
    steps: int = 5000
    lr: float = 1e-3
    warmup_steps: int = 250
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    train_min_len: int = 1
    train_max_len: int = 16
    eval_max_len: int = 48
    eval_samples: int = 256
    max_seq_len: int = 256


def run_name(config):
    return f"{config.task}_{config.pos_encoding}_seed{config.seed}"


def build_model(config):
    return DecoderOnlyLM(VOCAB_SIZE, config.d_model, config.num_heads, config.num_layers,
                         config.d_ff, config.pos_encoding, config.max_seq_len, config.dropout)


def lr_factor(config, step):
    """ Linear warmup, then cosine decay to zero. """
    if step < config.warmup_steps:
        return (step + 1) / config.warmup_steps
    progress = (step - config.warmup_steps) / max(1, config.steps - config.warmup_steps)
    return 0.5 * (1 + math.cos(math.pi * progress))


def train(config, log_every=250, verbose=False):
    """
    :return: (trained model, mean training loss over each block of log_every steps)
    """
    torch.manual_seed(config.seed)
    generator = torch.Generator().manual_seed(config.seed)
    model = build_model(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: lr_factor(config, step))

    model.train()
    history, running = [], 0.0
    for step in range(config.steps):
        batch = sample_batch(config.task, config.batch_size, config.train_min_len, config.train_max_len, generator)
        logits = model(batch.inputs)

        # Only the answer tokens (y1..yn, EOS) are scored
        loss = F.cross_entropy(logits[batch.loss_mask], batch.targets[batch.loss_mask])
        if not torch.isfinite(loss):
            raise RuntimeError(f"non-finite loss in {run_name(config)} at step {step}")

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        optimizer.step()
        scheduler.step()

        running += loss.item()
        if (step + 1) % log_every == 0:
            history.append(running / log_every)
            running = 0.0
            if verbose:
                print(f"{run_name(config)} step {step + 1}/{config.steps} loss {history[-1]:.4f}", flush=True)
    return model, history


@torch.no_grad()
def teacher_forced_correct(model, prompt, answer):
    """
    Feeds [prompt, answer] in one pass and checks each answer token against the
    model's argmax prediction given the true prefix.
    A row is all True exactly when greedy decoding would reproduce the answer, so
    exact match from this single pass equals exact match from step-by-step decoding.
    :param prompt: (batch_size, n + 2) = [BOS, x, SEP]
    :param answer: (batch_size, n + 1) = [y, EOS]
    :return: BoolTensor (batch_size, n + 1)
    """
    tokens = torch.cat([prompt, answer[:, :-1]], dim=1)
    predictions = model(tokens)[:, prompt.size(1) - 1:].argmax(dim=-1)
    return predictions == answer


@torch.no_grad()
def evaluate_lengths(model, task, lengths, samples, batch_size=256):
    """
    exact_match:    all n symbols and the final EOS are correct (same as greedy decoding)
    token_accuracy: fraction of the n symbols predicted correctly given the true prefix
    :return: {n: {"exact_match": float, "token_accuracy": float}}
    """
    model.eval()
    results = {}
    for n in lengths:
        generator = torch.Generator().manual_seed(EVAL_SEED + n)
        exact = tokens = done = 0
        while done < samples:
            size = min(batch_size, samples - done)
            prompt, answer = sample_fixed_length(task, size, n, generator)
            correct = teacher_forced_correct(model, prompt, answer)
            exact += correct.all(dim=1).sum().item()
            tokens += correct[:, :n].sum().item()
            done += size
        results[n] = {"exact_match": exact / samples, "token_accuracy": tokens / (samples * n)}
    return results


def in_distribution_exact_match(result):
    """ Mean exact match over the training lengths, from a run's result dict. """
    config = result["config"]
    lengths = range(config["train_min_len"], config["train_max_len"] + 1)
    return sum(result["eval"][str(n)]["exact_match"] for n in lengths) / len(lengths)


def run(config, out_dir, checkpoint_dir, verbose=False):
    """ Train, save the weights, evaluate every length 1..eval_max_len, and write <run_name>.json. """
    out_dir, checkpoint_dir = Path(out_dir), Path(checkpoint_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    name = run_name(config)

    start = time.perf_counter()
    model, history = train(config, verbose=verbose)
    train_seconds = time.perf_counter() - start
    torch.save(model.state_dict(), checkpoint_dir / f"{name}.pt")

    evaluation = evaluate_lengths(model, config.task, range(1, config.eval_max_len + 1), config.eval_samples)
    result = {
        "config": asdict(config),
        "loss_history": history,
        "train_seconds": round(train_seconds, 1),
        "eval": {str(n): metrics for n, metrics in evaluation.items()},
    }

    # Write atomically so an interrupted run never leaves a JSON that looks finished
    tmp_path = out_dir / f"{name}.json.tmp"
    tmp_path.write_text(json.dumps(result, indent=2))
    os.replace(tmp_path, out_dir / f"{name}.json")
    return result


def main():
    parser = argparse.ArgumentParser(description="Train and evaluate one length-generalization run.")
    parser.add_argument("--task", required=True, choices=TASKS)
    parser.add_argument("--pos-encoding", required=True, choices=POSITIONAL_ENCODINGS)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, help="override RunConfig.steps")
    parser.add_argument("--eval-samples", type=int, help="override RunConfig.eval_samples")
    parser.add_argument("--threads", type=int, default=2, help="torch CPU threads")
    parser.add_argument("--out", default="results/runs")
    parser.add_argument("--checkpoints", default="checkpoints")
    args = parser.parse_args()

    torch.set_num_threads(args.threads)
    overrides = {"steps": args.steps, "eval_samples": args.eval_samples}
    config = RunConfig(args.task, args.pos_encoding, args.seed,
                       **{k: v for k, v in overrides.items() if v is not None})
    result = run(config, args.out, args.checkpoints, verbose=True)
    print(f"{run_name(config)}: trained in {result['train_seconds']}s, "
          f"in-distribution exact match {in_distribution_exact_match(result):.3f}")


if __name__ == "__main__":
    main()
