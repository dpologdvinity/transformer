"""
Algorithmic sequence tasks for a decoder-only model.

Every example is one sequence:  [BOS, x1..xn, SEP, y1..yn, EOS]
where y is the task applied to x. The model is trained to predict y and EOS.
"""
from dataclasses import dataclass

import torch

# Special tokens, then NUM_SYMBOLS ordinary symbols
PAD, BOS, SEP, EOS = 0, 1, 2, 3
FIRST_SYMBOL = 4
NUM_SYMBOLS = 20
VOCAB_SIZE = FIRST_SYMBOL + NUM_SYMBOLS

TASKS = ("copy", "reverse")


def make_target(task, x):
    """
    :param x: Symbol IDs, shape (batch_size, n)
    :return: The task's answer for each row, shape (batch_size, n)
    """
    if task == "copy":
        return x.clone()
    if task == "reverse":
        return x.flip(-1)
    raise ValueError(f"unknown task {task!r}, expected one of {TASKS}")


@dataclass
class Batch:
    inputs: torch.Tensor     # (batch_size, L) tokens fed to the model
    targets: torch.Tensor    # (batch_size, L) next-token labels
    loss_mask: torch.Tensor  # (batch_size, L) True where the label is y1..yn or EOS


def _sample_symbols(batch_size, n, generator):
    return torch.randint(FIRST_SYMBOL, VOCAB_SIZE, (batch_size, n), generator=generator)


def sample_batch(task, batch_size, min_len, max_len, generator):
    """
    Training batch with a separate length n ~ U[min_len, max_len] per row, right-padded.
    Under causal attention, real tokens never see the padding to their right.
    """
    lengths = torch.randint(min_len, max_len + 1, (batch_size,), generator=generator).tolist()
    total_len = 2 * max(lengths) + 3
    seq = torch.full((batch_size, total_len), PAD)
    is_answer = torch.zeros(batch_size, total_len, dtype=torch.bool)

    for i, n in enumerate(lengths):
        x = _sample_symbols(1, n, generator)[0]
        y = make_target(task, x.unsqueeze(0))[0]
        row = torch.cat([torch.tensor([BOS]), x, torch.tensor([SEP]), y, torch.tensor([EOS])])
        seq[i, :len(row)] = row
        is_answer[i, n + 2:2 * n + 3] = True  # positions of y1..yn, EOS

    # Shift by one: the token at position t is predicted from positions < t
    return Batch(inputs=seq[:, :-1], targets=seq[:, 1:], loss_mask=is_answer[:, 1:])


def sample_fixed_length(task, batch_size, n, generator):
    """
    Evaluation examples of a single length n.
    :return: (prompt [BOS, x, SEP] of shape (batch_size, n + 2),
              answer [y, EOS] of shape (batch_size, n + 1))
    """
    x = _sample_symbols(batch_size, n, generator)
    prompt = torch.cat([torch.full((batch_size, 1), BOS), x, torch.full((batch_size, 1), SEP)], dim=1)
    answer = torch.cat([make_target(task, x), torch.full((batch_size, 1), EOS)], dim=1)
    return prompt, answer


def source_positions(task, n):
    """ Sequence index of the input symbol that each output y_k should copy. """
    positions = torch.arange(1, n + 1)
    if task == "copy":
        return positions
    if task == "reverse":
        return positions.flip(0)
    raise ValueError(f"unknown task {task!r}, expected one of {TASKS}")


def answer_query_positions(n):
    """ Sequence index whose next-token prediction is y_k (SEP for y_1, then y_1..y_{n-1}). """
    return torch.arange(n + 1, 2 * n + 1)
