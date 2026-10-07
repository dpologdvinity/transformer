import math

import torch
import torch.nn as nn

from transformer.blocks import TransformerEncoderBlock, create_causal_mask
from transformer.positional import RotaryEmbedding, alibi_bias, sinusoidal_table

POSITIONAL_ENCODINGS = ("none", "sinusoidal", "rope", "alibi")


class DecoderOnlyLM(nn.Module):
    """
    GPT-style decoder-only language model: the encoder block plus a causal mask.
    The positional encoding is selectable so encodings can be compared on an
    otherwise identical architecture.
    """
    def __init__(self, vocab_size, d_model, num_heads, num_layers, d_ff, pos_encoding,
                 max_seq_len=256, dropout=0.0):
        super(DecoderOnlyLM, self).__init__()
        if pos_encoding not in POSITIONAL_ENCODINGS:
            raise ValueError(f"unknown positional encoding {pos_encoding!r}, expected one of {POSITIONAL_ENCODINGS}")

        self.d_model = d_model
        self.pos_encoding = pos_encoding
        self.max_seq_len = max_seq_len

        # 1. Token embeddings, plus whichever position signal is selected
        #    none:       position comes only from the causal mask
        #    sinusoidal: added to the token embeddings
        #    rope:       Q and K rotated inside every attention layer
        #    alibi:      distance penalty added to every attention score
        self.token_embed = nn.Embedding(vocab_size, d_model)
        if pos_encoding == "sinusoidal":
            self.register_buffer('pe', sinusoidal_table(max_seq_len, d_model), persistent=False)
        if pos_encoding == "alibi":
            self.register_buffer('alibi', alibi_bias(num_heads, max_seq_len), persistent=False)
        rotary = RotaryEmbedding(d_model // num_heads, max_seq_len) if pos_encoding == "rope" else None
        self.register_buffer('causal_mask', create_causal_mask(max_seq_len), persistent=False)

        # 2. Decoder stack (self-attention only, so the encoder block is reused)
        self.layers = nn.ModuleList([
            TransformerEncoderBlock(d_model, num_heads, d_ff, dropout, rotary)
            for _ in range(num_layers)
        ])
        self.dropout = nn.Dropout(dropout)

        # 3. Project to vocabulary
        self.lm_head = nn.Linear(d_model, vocab_size)

    def forward(self, tokens):
        """
        :param tokens: Token IDs, shape (batch_size, seq_len)
        :return: Logits, shape (batch_size, seq_len, vocab_size)
        """
        seq_len = tokens.size(1)
        if seq_len > self.max_seq_len:
            raise ValueError(f"sequence length {seq_len} exceeds max_seq_len {self.max_seq_len}")

        x = self.token_embed(tokens) * math.sqrt(self.d_model)
        if self.pos_encoding == "sinusoidal":
            x = x + self.pe[:seq_len]
        x = self.dropout(x)

        mask = self.causal_mask[:, :, :seq_len, :seq_len]
        bias = self.alibi[:, :, :seq_len, :seq_len] if self.pos_encoding == "alibi" else None
        for layer in self.layers:
            x = layer(x, mask, bias)
        return self.lm_head(x)

    @torch.no_grad()
    def generate(self, prompt, num_new_tokens):
        """
        Greedy decoding. Recomputes the full sequence each step (no KV cache).
        :param prompt: Token IDs, shape (batch_size, prompt_len)
        :return: Generated token IDs, shape (batch_size, num_new_tokens)
        """
        tokens = prompt
        for _ in range(num_new_tokens):
            next_token = self(tokens)[:, -1].argmax(dim=-1, keepdim=True)
            tokens = torch.cat([tokens, next_token], dim=1)
        return tokens[:, prompt.size(1):]

    def attention_maps(self):
        """ Attention weights from the last forward pass, one (B, H, T, T) tensor per layer. """
        return [layer.mha.attention_weights for layer in self.layers]
