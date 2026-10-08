import pytest
import torch

from transformer.tasks import (
    BOS,
    EOS,
    NUM_SYMBOLS,
    PAD,
    SEP,
    VOCAB_SIZE,
    answer_query_positions,
    make_target,
    sample_batch,
    sample_fixed_length,
    source_positions,
)


def gen(seed=0):
    return torch.Generator().manual_seed(seed)


def test_vocab_layout():
    assert (PAD, BOS, SEP, EOS) == (0, 1, 2, 3)
    assert NUM_SYMBOLS == 20 and VOCAB_SIZE == 24


def test_copy_and_reverse_targets():
    x = torch.tensor([[4, 5, 6]])
    assert make_target("copy", x).tolist() == [[4, 5, 6]]
    assert make_target("reverse", x).tolist() == [[6, 5, 4]]


def test_unknown_task_raises():
    with pytest.raises(ValueError):
        make_target("sort", torch.tensor([[4, 5]]))


@pytest.mark.parametrize("task", ["copy", "reverse"])
def test_batch_format_and_loss_mask(task):
    batch = sample_batch(task, 32, 1, 16, gen())
    seq = torch.cat([batch.inputs[:, :1], batch.targets], dim=1)  # rebuild full sequences
    for row, targets, mask in zip(seq, batch.targets, batch.loss_mask, strict=True):
        n = int((row >= 4).sum()) // 2
        x, y = row[1:n + 1], row[n + 2:2 * n + 2]
        assert row[0] == BOS and row[n + 1] == SEP and row[2 * n + 2] == EOS
        assert torch.all(row[2 * n + 3:] == PAD)
        assert torch.equal(y, make_target(task, x.unsqueeze(0))[0])
        assert mask.sum() == n + 1
        assert torch.equal(targets[mask], row[n + 2:2 * n + 3])  # loss covers exactly y, EOS


def test_batch_lengths_cover_range():
    batch = sample_batch("copy", 512, 1, 16, gen())
    lengths = batch.loss_mask.sum(1) - 1
    assert lengths.min() == 1 and lengths.max() == 16


def test_symbols_in_range():
    prompt, _ = sample_fixed_length("copy", 64, 10, gen())
    x = prompt[:, 1:-1]
    assert x.min() >= 4 and x.max() <= VOCAB_SIZE - 1


def test_fixed_length_prompt_and_answer():
    prompt, answer = sample_fixed_length("reverse", 4, 5, gen())
    assert prompt.shape == (4, 7) and answer.shape == (4, 6)
    assert torch.all(prompt[:, 0] == BOS) and torch.all(prompt[:, -1] == SEP)
    assert torch.equal(answer[:, :-1], prompt[:, 1:-1].flip(1))
    assert torch.all(answer[:, -1] == EOS)


def test_sampling_is_deterministic_given_generator():
    a, _ = sample_fixed_length("copy", 8, 12, gen(7))
    b, _ = sample_fixed_length("copy", 8, 12, gen(7))
    assert torch.equal(a, b)


def test_attention_positions():
    assert source_positions("copy", 4).tolist() == [1, 2, 3, 4]
    assert source_positions("reverse", 4).tolist() == [4, 3, 2, 1]
    assert answer_query_positions(4).tolist() == [5, 6, 7, 8]
