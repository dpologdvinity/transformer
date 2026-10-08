import pytest
import torch

from transformer.blocks import TransformerEncoderBlock, create_causal_mask
from transformer.positional import RotaryEmbedding, alibi_bias, alibi_slopes


def test_rope_position_zero_is_identity():
    rope = RotaryEmbedding(head_dim=8)
    x = torch.randn(2, 4, 5, 8)
    assert torch.allclose(rope(x)[:, :, 0], x[:, :, 0])


def test_rope_preserves_norm():
    rope = RotaryEmbedding(head_dim=8)
    x = torch.randn(2, 4, 20, 8)
    assert torch.allclose(rope(x).norm(dim=-1), x.norm(dim=-1), atol=1e-5)


def test_rope_scores_depend_only_on_offset():
    torch.manual_seed(0)
    rope = RotaryEmbedding(head_dim=16, max_seq_len=64)
    q, k = torch.randn(16), torch.randn(16)
    rq, rk = rope(q.expand(1, 1, 64, 16)), rope(k.expand(1, 1, 64, 16))

    def score(m, n):
        return (rq[0, 0, m] * rk[0, 0, n]).sum()

    assert torch.allclose(score(10, 3), score(30, 23), atol=1e-4)
    assert not torch.allclose(score(10, 3), score(10, 9), atol=1e-4)


def test_rope_rejects_sequences_longer_than_max():
    rope = RotaryEmbedding(head_dim=8, max_seq_len=16)
    with pytest.raises(ValueError):
        rope(torch.randn(1, 1, 17, 8))


def test_alibi_slopes():
    assert torch.allclose(alibi_slopes(4), torch.tensor([2**-2, 2**-4, 2**-6, 2**-8]))


def test_alibi_bias_zero_diagonal_and_decreasing_with_distance():
    bias = alibi_bias(num_heads=4, seq_len=8)
    assert bias.shape == (1, 4, 8, 8)
    assert torch.all(torch.diagonal(bias[0], dim1=-2, dim2=-1) == 0)
    row = bias[0, :, 5, :6]  # keys 0..5 for query 5
    assert torch.all(row[:, 1:] > row[:, :-1])


def test_encoder_block_accepts_rotary_and_bias():
    block = TransformerEncoderBlock(32, 4, 64, dropout=0.0, rotary=RotaryEmbedding(head_dim=8))
    x = torch.randn(2, 6, 32)
    out = block(x, create_causal_mask(6), bias=alibi_bias(4, 6))
    assert out.shape == (2, 6, 32)
