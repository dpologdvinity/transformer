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
