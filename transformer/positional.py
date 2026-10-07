import math

import torch
import torch.nn as nn


def sinusoidal_table(max_len, d_model):
    """
    Fixed sinusoidal position encodings from "Attention Is All You Need".
    :return: Tensor of shape (max_len, d_model)
    """
    pe = torch.zeros(max_len, d_model)
    position = torch.arange(0, max_len).unsqueeze(1).float()
    div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe


class TokenAndPositionalEmbedding(nn.Module):
    def __init__(self, vocab_size, d_model, max_seq_len):
        """
        :param vocab_size: Total size of the vocabulary.
        :param d_model: The embedding dimension (e.g., 512).
        :param max_seq_len: The maximum length of a sequence.
        """
        super(TokenAndPositionalEmbedding, self).__init__()
        self.d_model = d_model

        # 1. Token Embedding Layer
        self.token_embed = nn.Embedding(vocab_size, d_model)

        # 2. Positional Encoding
        # We 'register_buffer' so 'pe' is part of the model's state,
        # but not a trainable parameter.
        self.register_buffer('pe', sinusoidal_table(max_seq_len, d_model).unsqueeze(0)) # Shape (1, max_seq_len, d_model)

    def forward(self, x):
        """
        :param x: Input token IDs, shape (batch_size, seq_len)
        :return: Embeddings with position info, shape (batch_size, seq_len, d_model)
        """

        # Get the sequence length from the input
        seq_len = x.size(1)

        # 1. Get token embeddings from self.token_embed
        token_embeddings = self.token_embed(x) # Shape (batch_size, seq_len, d_model)

        # 2. Scale token embeddings (common practice)
        token_embeddings = token_embeddings * math.sqrt(self.d_model)

        # 3. Add the positional encodings
        #    self.pe[:, :seq_len] selects the positions for this batch.
        #    This is the X_PE = X + PE step .
        final_embeddings = token_embeddings + self.pe[:, :seq_len]

        return final_embeddings


class RotaryEmbedding(nn.Module):
    """
    Rotary position embedding (RoPE, Su et al. 2021).
    Rotates pairs of query/key features by an angle proportional to the token's
    position, so the score between a query and a key depends only on their offset.
    Applied to Q and K after the heads are split; it has no learnable parameters.
    """
    def __init__(self, head_dim, max_seq_len=256, base=10000.0):
        super(RotaryEmbedding, self).__init__()
        assert head_dim % 2 == 0, "head_dim must be even for rotary embeddings"
        self.max_seq_len = max_seq_len

        # One frequency per feature pair: base^(-2i / head_dim)
        inv_freq = base ** (-torch.arange(0, head_dim, 2).float() / head_dim)
        angles = torch.arange(max_seq_len).float().unsqueeze(1) * inv_freq  # (max_seq_len, head_dim/2)
        angles = torch.cat([angles, angles], dim=-1)  # (max_seq_len, head_dim)
        self.register_buffer('cos', angles.cos(), persistent=False)
        self.register_buffer('sin', angles.sin(), persistent=False)

    def forward(self, x):
        """
        :param x: Queries or keys, shape (batch_size, num_heads, seq_len, head_dim)
        :return: Rotated tensor, same shape
        """
        seq_len = x.size(-2)
        if seq_len > self.max_seq_len:
            raise ValueError(f"sequence length {seq_len} exceeds rotary max_seq_len {self.max_seq_len}")

        # "Rotate half": feature i is paired with feature i + head_dim/2
        x1, x2 = x.chunk(2, dim=-1)
        rotated = torch.cat([-x2, x1], dim=-1)
        return x * self.cos[:seq_len] + rotated * self.sin[:seq_len]


def alibi_slopes(num_heads):
    """
    Per-head ALiBi slopes (Press et al. 2022): 2^(-8h/H) for h = 1..H.
    :return: Tensor of shape (num_heads,)
    """
    h = torch.arange(1, num_heads + 1).float()
    return 2 ** (-8 * h / num_heads)


def alibi_bias(num_heads, seq_len):
    """
    ALiBi attention bias: -slope_h * |i - j|, a linear penalty on distance.
    Entries above the diagonal are always removed by the causal mask.
    :return: Tensor of shape (1, num_heads, seq_len, seq_len)
    """
    pos = torch.arange(seq_len)
    distance = (pos.view(-1, 1) - pos.view(1, -1)).abs().float()  # (seq_len, seq_len)
    return -alibi_slopes(num_heads).view(1, -1, 1, 1) * distance
