import json

import pytest
import torch

from transformer.decoder_lm import DecoderOnlyLM
from transformer.tasks import EOS, PAD, make_target
from transformer.train import RunConfig, evaluate_lengths, run, run_name, train

TINY = dict(d_model=32, num_heads=2, num_layers=2, d_ff=64)


class OracleModel:
    """ Stub that answers from the prompt, optionally corrupted. """
    def __init__(self, task, corrupt=None):
        self.task, self.corrupt = task, corrupt

    def eval(self):
        return self

    def generate(self, prompt, num_new_tokens):
        x = prompt[:, 1:-1]
        out = torch.cat([make_target(self.task, x), torch.full((x.size(0), 1), EOS)], dim=1)
        if self.corrupt == "drop_eos":
            out[:, -1] = PAD
        if self.corrupt == "early_eos":
            out[:, 0] = EOS
        return out


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
