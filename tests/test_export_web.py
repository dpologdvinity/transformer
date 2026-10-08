from dataclasses import asdict

import torch

from scripts.export_web import build_study
from transformer.train import RunConfig, build_model, run_name

TINY = dict(d_model=32, num_heads=2, num_layers=2, d_ff=64)


def fake_run(task, pe, seed, scores, tmp_path=None):
    config = RunConfig(task, pe, seed, eval_max_len=len(scores), train_max_len=min(16, len(scores)), **TINY)
    if tmp_path is not None:
        torch.manual_seed(seed)
        torch.save(build_model(config).state_dict(), tmp_path / f"{run_name(config)}.pt")
    return {
        "config": asdict(config),
        "eval": {str(n): {"exact_match": s, "token_accuracy": s} for n, s in enumerate(scores, start=1)},
    }


def test_curves_aggregate_seeds_per_length():
    runs = [fake_run("copy", "rope", 0, [1.0, 0.5]), fake_run("copy", "rope", 1, [1.0, 0.1])]
    curve = build_study(runs, checkpoint_dir=None)["curves"]["copy"]["rope"]
    assert curve["n"] == [1, 2]
    assert curve["mean"] == [1.0, 0.3]
    assert curve["min"] == [1.0, 0.1] and curve["max"] == [1.0, 0.5]
    assert curve["seeds"] == 2


def test_meta_describes_training_setup():
    meta = build_study([fake_run("copy", "none", 0, [1.0] * 20)], checkpoint_dir=None)["meta"]
    assert meta["train_max_len"] == 16 and meta["steps"] == 5000 and meta["d_model"] == 32
    assert meta["seeds"] == 1


def test_attention_maps_for_reverse_seed0(tmp_path):
    scores = [1.0] * 32
    runs = [fake_run("reverse", pe, 0, scores, tmp_path) for pe in ("none", "rope")]
    attention = build_study(runs, checkpoint_dir=tmp_path, lengths=(4, 8))["attention"]
    assert set(attention) == {"none", "rope"}
    cell = attention["rope"]["8"]
    assert len(cell["weights"]) == 8 and all(len(row) == 8 for row in cell["weights"])
    assert all(0.0 <= w <= 1.0 for row in cell["weights"] for w in row)
    assert {"layer", "head", "score"} <= set(cell)


def test_kv_cache_benchmark_is_passed_through():
    runs = [fake_run("copy", "none", 0, [1.0] * 20)]
    bench = {"threads": 1, "reps": 3, "rows": [{"batch_size": 64, "new_tokens": 128, "speedup": 20.6}]}
    assert build_study(runs, checkpoint_dir=None, kv_cache=bench)["kv_cache"] == bench
    assert "kv_cache" not in build_study(runs, checkpoint_dir=None)
