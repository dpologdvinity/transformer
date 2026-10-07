import torch

from transformer.attention import MultiHeadAttention, scaled_dot_product_attention
from transformer.blocks import (
    PositionwiseFeedForward,
    TransformerDecoderBlock,
    TransformerEncoderBlock,
    create_causal_mask,
)
from transformer.positional import TokenAndPositionalEmbedding
from transformer.seq2seq import SOS_IDX, Transformer, greedy_decode

B, T, D_MODEL, H, V, D_FF = 2, 10, 64, 4, 100, 128
D_K = D_MODEL // H


def test_positional_encoding_buffer_shape():
    emb = TokenAndPositionalEmbedding(V, D_MODEL, max_seq_len=20)
    assert emb.pe.shape == (1, 20, D_MODEL)


def test_embedding_output_shape():
    emb = TokenAndPositionalEmbedding(V, D_MODEL, max_seq_len=20)
    assert emb(torch.randint(0, V, (B, T))).shape == (B, T, D_MODEL)


def test_attention_context_and_weight_shapes():
    q, k, v = (torch.randn(B, H, T, D_K) for _ in range(3))
    context, weights = scaled_dot_product_attention(q, k, v)
    assert context.shape == (B, H, T, D_K)
    assert weights.shape == (B, H, T, T)


def test_multi_head_attention_output_shape():
    mha = MultiHeadAttention(D_MODEL, H)
    x = torch.randn(B, T, D_MODEL)
    assert mha(x, x, x).shape == (B, T, D_MODEL)


def test_feed_forward_output_shape():
    ffn = PositionwiseFeedForward(D_MODEL, D_FF)
    assert ffn(torch.randn(B, T, D_MODEL)).shape == (B, T, D_MODEL)


def test_encoder_and_decoder_block_shapes():
    x = torch.randn(B, T, D_MODEL)
    memory = torch.randn(B, T + 3, D_MODEL)
    encoder = TransformerEncoderBlock(D_MODEL, H, D_FF)
    decoder = TransformerDecoderBlock(D_MODEL, H, D_FF)
    assert encoder(x).shape == (B, T, D_MODEL)
    assert decoder(x, memory, create_causal_mask(T)).shape == (B, T, D_MODEL)


def test_seq2seq_logits_shape():
    model = Transformer(100, 100, 64, 4, 2, 128, 20, 0.1)
    logits = model(torch.randint(3, 100, (2, 10)), torch.randint(3, 100, (2, 7)))
    assert logits.shape == (2, 7, 100)


def test_greedy_decode_starts_with_sos_and_respects_max_len():
    torch.manual_seed(0)
    model = Transformer(100, 100, 64, 4, 2, 128, 20, 0.1)
    out = greedy_decode(model, torch.randint(3, 100, (1, 10)), max_len=8)
    assert out[0, 0].item() == SOS_IDX
    assert out.shape[0] == 1 and out.shape[1] <= 8
