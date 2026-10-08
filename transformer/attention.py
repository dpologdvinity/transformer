import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def scaled_dot_product_attention(Q, K, V, mask=None, bias=None):
    """
    Implements the core attention formula:
    Attention(Q, K, V) = softmax( (QK^T) / sqrt(d_k) ) * V

    :param Q: Queries, shape (batch_size, num_heads, seq_len, d_k)
    :param K: Keys, shape (batch_size, num_heads, seq_len, d_k)
    :param V: Values, shape (batch_size, num_heads, seq_len_v, d_v) (Note: seq_len_k == seq_len_v)
    :param mask: Optional mask, shape (batch_size, 1, seq_len, seq_len)
    :param bias: Optional additive score bias, broadcastable to (batch_size, num_heads, seq_len, seq_len)
    :return: A tuple of (context_vector, attention_weights)
    """
    # 1. Get d_k
    d_k = Q.size(-1)

    # 2. Compute scores (QK^T) -> shape (B, num_heads, seq_len, seq_len)
    scores = torch.matmul(Q, K.transpose(-2, -1))

    # 3. Scale scores
    scores = scores / math.sqrt(d_k)

    # 4. Add positional bias (e.g. ALiBi), if provided
    if bias is not None:
        scores = scores + bias

    # 5. Apply mask (if provided)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))

    # 6. Apply softmax to get weights
    weights = F.softmax(scores, dim=-1)

    # 7. Compute context vector (weights * V)
    context_vector = torch.matmul(weights, V)

    return context_vector, weights


class MultiHeadAttention(nn.Module):
    """
    Implements a flexible Multi-Head Attention module.
    """
    def __init__(self, d_model, num_heads, rotary=None):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)

        # Optional rotary position embedding applied to Q and K
        self.rotary = rotary

        # Weights from the most recent forward pass, kept for visualization
        self.attention_weights = None

    def split_heads(self, x):
        """
        Splits the last dimension d_model into (num_heads, d_k).
        :param x: Tensor of shape (batch_size, seq_len, d_model)
        :return: Tensor of shape (batch_size, num_heads, seq_len, d_k)
        """
        batch_size, seq_len, _ = x.size()
        x = x.view(batch_size, seq_len, self.num_heads, self.d_k)
        return x.transpose(1, 2)  # (B, num_heads, seq_len, d_k)

    def forward(self, x_q, x_k, x_v, mask=None, bias=None, kv_cache=None, position=0):
        """
        A flexible forward pass.
        :param x_q: Input for Queries, shape (batch_size, seq_len_q, d_model)
        :param x_k: Input for Keys, shape (batch_size, seq_len_k, d_model)
        :param x_v: Input for Values, shape (batch_size, seq_len_v, d_model)
        :param mask: Optional mask
        :param bias: Optional additive score bias
        :param kv_cache: Optional dict holding this layer's past keys/values; extended in place
        :param position: Position of the first query token (for RoPE with a KV cache)
        :return: Output, shape (batch_size, seq_len_q, d_model)

        - For Self-Attention: x_q, x_k, x_v will be the SAME tensor.
        - For Cross-Attention: x_q will be the decoder state,
                               x_k and x_v will be the encoder output.
        """
        batch_size = x_q.size(0)

        # 1. Project: Create Q, K, V from their respective inputs
        Q = self.W_q(x_q)
        K = self.W_k(x_k)
        V = self.W_v(x_v)

        # 2. Split Heads -> (B, num_heads, seq_len, d_k)
        Q = self.split_heads(Q)
        K = self.split_heads(K)
        V = self.split_heads(V)

        # 2b. Rotate Q and K by position (RoPE), if enabled
        if self.rotary is not None:
            Q = self.rotary(Q, offset=position)
            K = self.rotary(K, offset=position)

        # 2c. Prepend cached keys/values from earlier decoding steps
        if kv_cache is not None:
            if "k" in kv_cache:
                K = torch.cat([kv_cache["k"], K], dim=2)
                V = torch.cat([kv_cache["v"], V], dim=2)
            kv_cache["k"], kv_cache["v"] = K, V

        # 3. Attention
        context_vector, weights = scaled_dot_product_attention(Q, K, V, mask, bias)
        self.attention_weights = weights.detach()

        # 4. Combine Heads
        seq_len_q = x_q.size(1)
        context_vector = context_vector.transpose(1, 2).contiguous()
        context_vector = context_vector.view(batch_size, seq_len_q, self.d_model)

        # 5. Final Linear Layer (W_o)
        output = self.W_o(context_vector)

        return output
