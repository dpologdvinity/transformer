import torch
import torch.nn as nn
import torch.nn.functional as F

from transformer.attention import MultiHeadAttention


class PositionwiseFeedForward(nn.Module):
    """ Implements the FFN(x) = ReLU(xW1 + b1)W2 + b2 """
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.linear2(self.dropout(F.relu(self.linear1(x))))


def create_causal_mask(seq_len):
    """
    Creates a "look-ahead" mask for causal attention.
    :param seq_len: The length of the sequence.
    :return: A mask tensor of shape (1, 1, seq_len, seq_len)
    """
    mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
    return (mask == 0).unsqueeze(0).unsqueeze(0) # Shape (1, 1, seq_len, seq_len)


def create_padding_mask(seq, pad_idx=0):
    """
    Masks out padding tokens so attention never reads them.
    :param seq: Token IDs, shape (batch_size, seq_len)
    :return: A mask tensor of shape (batch_size, 1, 1, seq_len), True for real tokens
    """
    return (seq != pad_idx).unsqueeze(1).unsqueeze(2)


class TransformerEncoderBlock(nn.Module):
    """ Implements a single Transformer Encoder Block """
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1, rotary=None):
        super(TransformerEncoderBlock, self).__init__()

        self.mha = MultiHeadAttention(d_model, num_heads, rotary)
        self.ffn = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.layernorm1 = nn.LayerNorm(d_model)
        self.layernorm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, mask=None, bias=None, kv_cache=None, position=0):
        """
        :param x: Input, shape (batch_size, seq_len, d_model)
        :param mask: Optional mask (e.g., for padding)
        :param bias: Optional additive attention bias (e.g., ALiBi)
        :param kv_cache: Optional per-layer key/value cache (see DecoderOnlyLM)
        :param position: Position of the first token in x
        :return: Output, shape (batch_size, seq_len, d_model)
        """

        # 1. MHA + "Add & Norm"
        mha_output = self.mha(x, x, x, mask, bias, kv_cache, position)
        x_with_mha = self.layernorm1(x + self.dropout1(mha_output))

        # 2. FFN + "Add & Norm"
        ffn_output = self.ffn(x_with_mha)
        block_output = self.layernorm2(x_with_mha + self.dropout2(ffn_output))

        return block_output


class TransformerDecoderBlock(nn.Module):
    """ Implements a single Transformer Decoder Block """
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(TransformerDecoderBlock, self).__init__()

        self.masked_mha = MultiHeadAttention(d_model, num_heads)
        self.cross_mha = MultiHeadAttention(d_model, num_heads)
        self.ffn = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.layernorm1 = nn.LayerNorm(d_model)
        self.layernorm2 = nn.LayerNorm(d_model)
        self.layernorm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, x, encoder_output, causal_mask, src_padding_mask=None):
        """
        :param x: The decoder's input (e.g., target sequence embeddings)
                Shape: (batch_size, target_seq_len, d_model)
        :param encoder_output: The output from the final Encoder block.
                            Shape: (batch_size, src_seq_len, d_model)
        :param causal_mask: Mask for decoder self-attention (prevents looking ahead)
                        Shape: (1, 1, target_seq_len, target_seq_len)
        :param src_padding_mask: Optional mask for cross-attention (masks padded source tokens)
                                Shape: (batch_size, 1, 1, src_seq_len)
        :return: Output, shape (batch_size, target_seq_len, d_model)
        """

        # 1. Masked Self-Attention + "Add & Norm"
        mha_output = self.masked_mha(x, x, x, causal_mask)
        x_with_mha = self.layernorm1(x + self.dropout1(mha_output))

        # 2. Cross-Attention + "Add & Norm"
        #    Q = decoder state (x_with_mha), K/V = encoder output
        cross_output = self.cross_mha(x_with_mha, encoder_output, encoder_output, src_padding_mask)
        x_with_cross = self.layernorm2(x_with_mha + self.dropout2(cross_output))

        # 3. FFN + "Add & Norm"
        ffn_output = self.ffn(x_with_cross)
        block_output = self.layernorm3(x_with_cross + self.dropout3(ffn_output))

        return block_output
