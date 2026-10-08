import json
from dataclasses import asdict

import pytest

from scripts.sweep import build_grid, pending
from transformer.train import run_name


def test_grid_covers_all_combinations():
    grid = build_grid(seeds=(0, 1, 2))
    assert len(grid) == 24
    assert len({run_name(c) for c in grid}) == 24


def test_pending_skips_completed(tmp_path):
    grid = build_grid(seeds=(0,))
    (tmp_path / f"{run_name(grid[0])}.json").write_text(json.dumps({"config": asdict(grid[0])}))
    (tmp_path / f"{run_name(grid[1])}.json.tmp").write_text("{")  # killed mid-write
    remaining = pending(grid, tmp_path)
    assert grid[0] not in remaining
    assert grid[1] in remaining
    assert len(remaining) == len(grid) - 1


def test_overrides_apply_to_every_config():
    assert all(c.steps == 10 and c.eval_samples == 7 for c in build_grid(steps=10, eval_samples=7))


def test_pending_rejects_finished_run_with_different_config(tmp_path):
    old = build_grid(seeds=(0,), steps=3000)[0]
    (tmp_path / f"{run_name(old)}.json").write_text(json.dumps({"config": asdict(old)}))
    assert old not in pending(build_grid(seeds=(0,), steps=3000), tmp_path)
    with pytest.raises(ValueError, match="different config"):
        pending(build_grid(seeds=(0,), steps=5000), tmp_path)
