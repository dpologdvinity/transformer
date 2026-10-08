import json

import pytest
import torch

from transformer.decoder_lm import DecoderOnlyLM
from transformer.tasks import EOS, PAD, VOCAB_SIZE, make_target, sample_fixed_length
from transformer.train import RunConfig, evaluate_lengths, run, run_name, teacher_forced_correct, train

TINY = dict(d_model=32, num_heads=2, num_layers=2, d_ff=64)


class OracleModel:
    """ Stub whose next-token predictions are the correct answer, optionally corrupted. """
    def __init__(self, task, corrupt=None):
        self.task, self.corrupt = task, corrupt

    def eval(self):
        return self

    def __call__(self, tokens):
        n = (tokens.size(1) - 2) // 2  # tokens = [BOS, x1..xn, SEP, y1..yn]
        x = tokens[:, 1:n + 1]
        answer = torch.cat([make_target(self.task, x), torch.full((x.size(0), 1), EOS)], dim=1)
        if self.corrupt == "drop_eos":
            answer[:, -1] = PAD
        if self.corrupt == "early_eos":
            answer[:, 0] = EOS
        logits = torch.zeros(*tokens.shape, VOCAB_SIZE)
        logits[:, n + 1:].scatter_(-1, answer.unsqueeze(-1), 1.0)
        return logits


@pytest.mark.parametrize("task", ["copy", "reverse"])
def test_oracle_scores_perfectly(task):
    results = evaluate_lengths(OracleModel(task), task, range(1, 7), samples=8)
    assert all(r == {"exact_match": 1.0, "token_accuracy": 1.0} for r in results.values())


def test_missing_eos_fails_exact_match_only():
    r = evaluate_lengths(OracleModel("copy", "drop_eos"), "copy", [4], samples=8)
    assert r[4] == {"exact_match": 0.0, "token_accuracy": 1.0}


def test_early_eos_counts_wrong():
    r = evaluate_lengths(OracleModel("copy", "early_eos"), "copy", [4], samples=8)
    assert r[4]["exact_match"] == 0.0 and r[4]["token_accuracy"] == 0.75


def test_run_name():
    assert run_name(RunConfig("reverse", "rope", 2)) == "reverse_rope_seed2"


def test_smoke_training_reduces_loss():
    config = RunConfig("copy", "sinusoidal", 0, steps=60, warmup_steps=5, **TINY)
    model, history = train(config, log_every=10)
    assert isinstance(model, DecoderOnlyLM)
    assert len(history) == 6
    assert history[-1] < history[0]


def test_non_finite_loss_raises(monkeypatch):
    def nan_forward(self, tokens):
        return torch.full((*tokens.shape, 24), float("nan"), requires_grad=True)
    monkeypatch.setattr(DecoderOnlyLM, "forward", nan_forward)
    with pytest.raises(RuntimeError, match="non-finite loss in copy_none_seed0 at step"):
        train(RunConfig("copy", "none", 0, steps=5, **TINY))


def test_run_writes_json(tmp_path):
    config = RunConfig("reverse", "alibi", 1, steps=5, warmup_steps=1, eval_max_len=3, eval_samples=4, **TINY)
    run(config, tmp_path / "runs", tmp_path / "ckpt")
    data = json.loads((tmp_path / "runs" / "reverse_alibi_seed1.json").read_text())
    assert set(data) == {"config", "loss_history", "train_seconds", "eval"}
    assert data["config"]["pos_encoding"] == "alibi"
    assert set(data["eval"]) == {"1", "2", "3"}
    assert (tmp_path / "ckpt" / "reverse_alibi_seed1.pt").exists()
    assert not list((tmp_path / "runs").glob("*.tmp"))


def test_teacher_forced_scoring_agrees_with_greedy_decoding():
    # Exact match from one teacher-forced pass equals exact match from greedy decoding:
    # a model's own greedy output is always fully "correct" under teacher forcing,
    # and changing any answer token makes that row incorrect.
    torch.manual_seed(0)
    model = DecoderOnlyLM(VOCAB_SIZE, 32, 4, 2, 64, "rope").eval()
    prompt, _ = sample_fixed_length("copy", 16, 6, torch.Generator().manual_seed(0))
    greedy = model.generate(prompt, 7)
    assert teacher_forced_correct(model, prompt, greedy).all()
    altered = greedy.clone()
    altered[:8, 3] = (altered[:8, 3] + 1) % VOCAB_SIZE
    rows_correct = teacher_forced_correct(model, prompt, altered).all(dim=1)
    assert not rows_correct[:8].any() and rows_correct[8:].all()


def test_verbose_training_prints_progress(capsys):
    train(RunConfig("copy", "none", 0, steps=4, warmup_steps=1, **TINY), log_every=2, verbose=True)
    out = capsys.readouterr().out
    assert "copy_none_seed0 step 2/4 loss" in out and "copy_none_seed0 step 4/4 loss" in out
