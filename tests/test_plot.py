import pytest

from scripts.plot import group_runs


def fake_run(seed, steps):
    return {"config": {"task": "copy", "pos_encoding": "rope", "seed": seed, "steps": steps}, "eval": {}}


def test_group_runs_rejects_mixed_configs():
    with pytest.raises(ValueError, match="copy/rope"):
        group_runs([fake_run(0, 3000), fake_run(1, 5000)])


def test_group_runs_accepts_seed_only_differences():
    assert len(group_runs([fake_run(0, 3000), fake_run(1, 3000)])[("copy", "rope")]) == 2
