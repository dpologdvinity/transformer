import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
import math

# Part 1: Token and Positional Embedding
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
        pe = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0, max_seq_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # We 'register_buffer' so 'pe' is part of the model's state,
        # but not a trainable parameter.
        self.register_buffer('pe', pe.unsqueeze(0)) # Shape (1, max_seq_len, d_model)


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


#Part 2: The Attention Mechanism
def scaled_dot_product_attention(Q, K, V, mask=None):
    """
    Implements the core attention formula:
    Attention(Q, K, V) = softmax( (QK^T) / sqrt(d_k) ) * V

    :param Q: Queries, shape (batch_size, num_heads, seq_len, d_k)
    :param K: Keys, shape (batch_size, num_heads, seq_len, d_k)
    :param V: Values, shape (batch_size, num_heads, seq_len_v, d_v) (Note: seq_len_k == seq_len_v)
    :param mask: Optional mask, shape (batch_size, 1, seq_len, seq_len)
    :return: A tuple of (context_vector, attention_weights)
    """
    # 1. Get d_k
    d_k = Q.size(-1)

    # 2. Compute scores (QK^T) -> shape (B, num_heads, seq_len, seq_len)
    scores = torch.matmul(Q, K.transpose(-2, -1))

    # 3. Scale scores
    scores = scores / math.sqrt(d_k)

    # 4. Apply mask (if provided)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))

    # 5. Apply softmax to get weights
    weights = F.softmax(scores, dim=-1)

    # 6. Compute context vector (weights * V)
    context_vector = torch.matmul(weights, V)

    return context_vector, weights


class MultiHeadAttention(nn.Module):
    """
    Implements a flexible Multi-Head Attention module.
    """
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        # TODO: Initialize dimensions
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)

    def split_heads(self, x):
        """
        Splits the last dimension d_model into (num_heads, d_k).
        :param x: Tensor of shape (batch_size, seq_len, d_model)
        :return: Tensor of shape (batch_size, num_heads, seq_len, d_k)
        """
        batch_size, seq_len, _ = x.size()
        x = x.view(batch_size, seq_len, self.num_heads, self.d_k)
        return x.transpose(1, 2)  # (B, num_heads, seq_len, d_k)

    def forward(self, x_q, x_k, x_v, mask=None):
        """
        A flexible forward pass.
        :param x_q: Input for Queries, shape (batch_size, seq_len_q, d_model)
        :param x_k: Input for Keys, shape (batch_size, seq_len_k, d_model)
        :param x_v: Input for Values, shape (batch_size, seq_len_v, d_model)
        :param mask: Optional mask
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

        # 3. Attention
        context_vector, _ = scaled_dot_product_attention(Q, K, V, mask)

        # 4. Combine Heads
        seq_len_q = x_q.size(1)
        context_vector = context_vector.transpose(1, 2).contiguous()
        context_vector = context_vector.view(batch_size, seq_len_q, self.d_model)

        # 5. Final Linear Layer (W_o)
        output = self.W_o(context_vector)

        return output

#Part 3: Transformer Encoder Block
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

class TransformerEncoderBlock(nn.Module):
    """ Implements a single Transformer Encoder Block """
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(TransformerEncoderBlock, self).__init__()

        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ffn = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.layernorm1 = nn.LayerNorm(d_model)
        self.layernorm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        """
        :param x: Input, shape (batch_size, seq_len, d_model)
        :param mask: Optional mask (e.g., for padding)
        :return: Output, shape (batch_size, seq_len, d_model)
        """

        # 1. MHA + "Add & Norm"
        mha_output = self.mha(x, x, x, mask)
        x_with_mha = self.layernorm1(x + self.dropout1(mha_output))

        # 2. FFN + "Add & Norm"
        ffn_output = self.ffn(x_with_mha)
        block_output = self.layernorm2(x_with_mha + self.dropout2(ffn_output))

        return block_output


#Part 4: Transformer Decoder Block
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

################################################################################
# Train the model and testing inference

# --- Configuration ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters (Small for testing)
SRC_VOCAB_SIZE = 100
TGT_VOCAB_SIZE = 100
D_MODEL = 64
NUM_HEADS = 2
NUM_LAYERS = 2
D_FF = 128
MAX_SEQ_LEN = 20
DROPOUT = 0.1
BATCH_SIZE = 16
EPOCHS = 100

# Special Tokens
PAD_IDX = 0
SOS_IDX = 1
EOS_IDX = 2

def generate_random_data(batch_size, seq_len=10):
    """
    Generates a 'Copy Task' dataset.
    Input: [SOS, 5, 8, 2, ..., EOS]
    Target: [SOS, 5, 8, 2, ..., EOS]
    """
    data = torch.randint(3, SRC_VOCAB_SIZE, (batch_size, seq_len - 2))

    # Add SOS and EOS
    sos_col = torch.full((batch_size, 1), SOS_IDX)
    eos_col = torch.full((batch_size, 1), EOS_IDX)

    seq = torch.cat([sos_col, data, eos_col], dim=1)
    return seq.to(device), seq.to(device)

def train_model():
    print(f"--- Initializing Model on {device} ---")
    model = Transformer(
        SRC_VOCAB_SIZE, TGT_VOCAB_SIZE, D_MODEL, NUM_HEADS,
        NUM_LAYERS, D_FF, MAX_SEQ_LEN, DROPOUT
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)

    model.train()

    print("--- Starting Training (Task: Copy Input to Output) ---")
    for epoch in range(EPOCHS):
        # Generate random batch
        src_data, tgt_data = generate_random_data(BATCH_SIZE)

        # Teacher Forcing:
        # Decoder Input: <SOS> A B C
        # Target Label:  A B C <EOS>
        decoder_input = tgt_data[:, :-1]
        target_output = tgt_data[:, 1:]

        optimizer.zero_grad()

        # Forward pass
        logits = model(src_data, decoder_input)

        # Reshape for Loss: (batch * seq_len, vocab_size)
        loss = criterion(logits.reshape(-1, TGT_VOCAB_SIZE), target_output.reshape(-1))

        loss.backward()
        optimizer.step()

        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {loss.item():.4f}")

    return model

def greedy_decode(model, src_seq, max_len, start_symbol=SOS_IDX):
    """
    Performs inference using greedy decoding.
    """
    model.eval()
    src_seq = src_seq.to(device)

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

def run_shape_tests():
    print("--- Running Shape Sanity Checks ---")

    # 1. Define Hyperparameters for Testing
    B = 2   # Batch size
    T = 10  # Sequence length
    d_model = 64
    h = 4   # Number of heads
    d_k = d_model // h
    V = 100 # Vocab size
    d_ff = 128

    device = torch.device("cpu") # Keep it simple for shape tests

    # ==========================================
    # TEST 1: Embedding Shapes
    # ==========================================
    print("\n[1] Testing TokenAndPositionalEmbedding...")
    try:
        emb_layer = TokenAndPositionalEmbedding(V, d_model, max_seq_len=20).to(device)
        dummy_input = torch.randint(0, V, (B, T)).to(device)

        # Check 1: Positional Encoding Buffer Shape
        # Expected: (1, Max_Len, d_model)
        expected_pe_dims = (1, 20, d_model)
        if emb_layer.pe.shape != expected_pe_dims:
             print(f"FAIL: self.pe shape is {emb_layer.pe.shape}, expected {expected_pe_dims}")
        else:
             print(f"PASS: self.pe shape is correct {emb_layer.pe.shape}")

        # Check 2: Output Shape
        # Expected: (B, T, d_model)
        output = emb_layer(dummy_input)
        if output.shape != (B, T, d_model):
            print(f"FAIL: Embedding output shape is {output.shape}, expected {(B, T, d_model)}")
        else:
            print(f"PASS: Embedding output shape is correct {(B, T, d_model)}")

    except Exception as e:
        print(f"CRITICAL ERROR in Embedding: {e}")

    # ==========================================
    # TEST 2: Attention Internal Shapes
    # ==========================================
    print("\n[2] Testing scaled_dot_product_attention (Internal Shapes)...")
    try:
        # Create dummy Q, K, V with split heads
        # Shape: (B, h, T, d_k)
        q_test = torch.randn(B, h, T, d_k)
        k_test = torch.randn(B, h, T, d_k)
        v_test = torch.randn(B, h, T, d_k)

        context, weights = scaled_dot_product_attention(q_test, k_test, v_test)

        # Check 1: Context Vector (Head Output)
        # Expected: (B, h, T, d_k)
        if context.shape != (B, h, T, d_k):
            print(f"FAIL: Attention Context shape is {context.shape}, expected {(B, h, T, d_k)}")
        else:
            print(f"PASS: Context vector shape is correct {(B, h, T, d_k)}")

        # Check 2: Attention Weights (Softmax Output)
        # Expected: (B, h, T, T)
        if weights.shape != (B, h, T, T):
            print(f"FAIL: Attention Weights shape is {weights.shape}, expected {(B, h, T, T)}")
        else:
            print(f"PASS: Attention Weights shape is correct {(B, h, T, T)}")

    except Exception as e:
        print(f"CRITICAL ERROR in scaled_dot_product_attention: {e}")

    # ==========================================
    # TEST 3: MultiHeadAttention Module
    # ==========================================
    print("\n[3] Testing MultiHeadAttention (Full Module)...")
    try:
        mha = MultiHeadAttention(d_model, h).to(device)
        # Input is (B, T, d_model)
        x_test = torch.randn(B, T, d_model).to(device)

        # Self-attention: Q=K=V=x
        output = mha(x_test, x_test, x_test)

        # Check: Final Concatenated Output
        # Expected: (B, T, d_model)
        if output.shape != (B, T, d_model):
            print(f"FAIL: MHA Output shape is {output.shape}, expected {(B, T, d_model)}")
        else:
            print(f"PASS: MHA Output shape is correct {(B, T, d_model)}")

    except Exception as e:
         print(f"CRITICAL ERROR in MultiHeadAttention: {e}")

    # ==========================================
    # TEST 4: Feed-Forward Network
    # ==========================================
    print("\n[4] Testing PositionwiseFeedForward...")
    try:
        ffn = PositionwiseFeedForward(d_model, d_ff).to(device)
        x_test = torch.randn(B, T, d_model).to(device)

        output = ffn(x_test)

        # Check: Output Shape
        # Expected: (B, T, d_model)
        if output.shape != (B, T, d_model):
            print(f"FAIL: FFN Output shape is {output.shape}, expected {(B, T, d_model)}")
        else:
            print(f"PASS: FFN Output shape is correct {(B, T, d_model)}")

    except Exception as e:
        print(f"CRITICAL ERROR in FeedForward: {e}")

if __name__ == "__main__":
    run_shape_tests()

    ################################################################################
    # Train the model and testing inference

    # 1. Train the model
    trained_model = train_model()

    # 2. Test Inference (Translation)
    print("\n--- Testing Inference ---")
    test_src, _ = generate_random_data(1) # Get 1 random sequence
    print(f"Input Sequence:  {test_src[0].tolist()}")

    # Run translation
    generated_seq = greedy_decode(trained_model, test_src, max_len=20)
    print(f"Model Predicted: {generated_seq[0].tolist()}")

    # Check correctness
    print(f"You are not expect to get the same sequence, but do you see the loss decreasing? If so, you are on the right track!")
