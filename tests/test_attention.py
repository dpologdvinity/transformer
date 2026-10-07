import torch
import torch.nn as nn
import torch.nn.functional as F

from transformer.attention import MultiHeadAttention, scaled_dot_product_attention
from transformer.blocks import create_causal_mask

B, H, T, D_K = 2, 4, 7, 8


def random_qkv():
    torch.manual_seed(0)
    return (torch.randn(B, H, T, D_K) for _ in range(3))


def test_sdpa_matches_torch_unmasked():
    q, k, v = random_qkv()
    ours, _ = scaled_dot_product_attention(q, k, v)
    assert torch.allclose(ours, F.scaled_dot_product_attention(q, k, v), atol=1e-5)


def test_sdpa_matches_torch_causal():
    q, k, v = random_qkv()
    mask = create_causal_mask(T)
    ours, _ = scaled_dot_product_attention(q, k, v, mask)
    assert torch.allclose(ours, F.scaled_dot_product_attention(q, k, v, attn_mask=mask), atol=1e-5)


def test_sdpa_bias_matches_torch_float_mask():
    q, k, v = random_qkv()
    bias = torch.randn(1, H, T, T)
    ours, _ = scaled_dot_product_attention(q, k, v, bias=bias)
    assert torch.allclose(ours, F.scaled_dot_product_attention(q, k, v, attn_mask=bias), atol=1e-5)


def test_mha_matches_nn_multihead_attention():
    torch.manual_seed(0)
    d_model = H * D_K
    ours = MultiHeadAttention(d_model, H)
    reference = nn.MultiheadAttention(d_model, H, bias=False, batch_first=True)
    with torch.no_grad():
        reference.in_proj_weight.copy_(torch.cat([ours.W_q.weight, ours.W_k.weight, ours.W_v.weight]))
        reference.out_proj.weight.copy_(ours.W_o.weight)
    x = torch.randn(B, T, d_model)
    causal = create_causal_mask(T)
    expected, _ = reference(x, x, x, attn_mask=~causal[0, 0], need_weights=False)
    assert torch.allclose(ours(x, x, x, causal), expected, atol=1e-5)


def test_mha_stores_attention_weights():
    mha = MultiHeadAttention(H * D_K, H)
    assert mha.attention_weights is None
    x = torch.randn(B, T, H * D_K, requires_grad=True)
    mha(x, x, x)
    assert mha.attention_weights.shape == (B, H, T, T)
    assert not mha.attention_weights.requires_grad
    assert torch.allclose(mha.attention_weights.sum(-1), torch.ones(B, H, T), atol=1e-5)
