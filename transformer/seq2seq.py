import torch
import torch.nn as nn

from transformer.blocks import TransformerDecoderBlock, TransformerEncoderBlock, create_causal_mask
from transformer.positional import TokenAndPositionalEmbedding

# Special Tokens
PAD_IDX = 0
SOS_IDX = 1
EOS_IDX = 2


class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model, num_heads, num_layers, d_ff, max_seq_len, dropout):
        super(Transformer, self).__init__()

        # 1. Embeddings
        self.src_embedding = TokenAndPositionalEmbedding(src_vocab_size, d_model, max_seq_len)
        self.tgt_embedding = TokenAndPositionalEmbedding(tgt_vocab_size, d_model, max_seq_len)

        # 2. Encoder Stack
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderBlock(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])

        # 3. Decoder Stack
        self.decoder_layers = nn.ModuleList([
            TransformerDecoderBlock(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])

        # 4. Final Linear Layer (Project to Target Vocab)
        self.final_linear = nn.Linear(d_model, tgt_vocab_size)
        self.dropout = nn.Dropout(dropout)

    def encode(self, src, src_mask):
        # Embed and add position info
        x = self.src_embedding(src)
        x = self.dropout(x)

        # Pass through all encoder layers
        for layer in self.encoder_layers:
            x = layer(x, src_mask)
        return x

    def decode(self, tgt, memory, causal_mask, tgt_mask):
        # Embed and add position info
        x = self.tgt_embedding(tgt)
        x = self.dropout(x)

        # Pass through all decoder layers
        for layer in self.decoder_layers:
            x = layer(x, memory, causal_mask, tgt_mask)
        return x

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        """
        :param src: Source Sequence (batch_size, src_len)
        :param tgt: Target Sequence (batch_size, tgt_len)
        """
        # 1. Create Causal Mask for the Decoder (Look-ahead mask)
        tgt_seq_len = tgt.size(1)
        causal_mask = create_causal_mask(tgt_seq_len).to(tgt.device)

        # 2. Run Encoder
        memory = self.encode(src, src_mask)

        # 3. Run Decoder
        output = self.decode(tgt, memory, causal_mask, tgt_mask)

        # 4. Final Projection
        logits = self.final_linear(output)
        return logits


def greedy_decode(model, src_seq, max_len, start_symbol=SOS_IDX):
    """
    Performs inference using greedy decoding.
    """
    model.eval()
    device = src_seq.device

    # 1. Encode the source
    memory = model.encode(src_seq, src_mask=None)

    # 2. Initialize the decoder input with <SOS>
    ys = torch.ones(1, 1).fill_(start_symbol).type(torch.long).to(device)

    # 3. Autoregressive generation
    for i in range(max_len-1):
        # Create causal mask for current sequence length
        tgt_mask = create_causal_mask(ys.size(1)).to(device)

        # Decode
        out = model.decode(ys, memory, tgt_mask, tgt_mask=None)

        # Get projection of the last token
        prob = model.final_linear(out[:, -1])

        # Get token with max probability
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.item()

        # Append to sequence
        ys = torch.cat([ys, torch.ones(1, 1).type_as(src_seq.data).fill_(next_word)], dim=1)

        if next_word == EOS_IDX:
            break

    return ys
