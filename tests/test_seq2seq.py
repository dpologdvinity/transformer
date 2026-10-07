import torch

from transformer.blocks import create_padding_mask
from transformer.seq2seq import PAD_IDX, Transformer


def test_padding_mask_marks_real_tokens():
    seq = torch.tensor([[5, 6, PAD_IDX], [7, PAD_IDX, PAD_IDX]])
    mask = create_padding_mask(seq)
    assert mask.shape == (2, 1, 1, 3)
    assert mask[:, 0, 0].tolist() == [[True, True, False], [True, False, False]]


def test_masked_source_padding_does_not_change_output():
    torch.manual_seed(0)
    model = Transformer(50, 50, 32, 4, 2, 64, 32, 0.0).eval()
    src, tgt = torch.randint(3, 50, (2, 8)), torch.randint(3, 50, (2, 6))
    padded = torch.cat([src, torch.full((2, 4), PAD_IDX)], dim=1)
    out = model(src, tgt, src_mask=create_padding_mask(src))
    out_padded = model(padded, tgt, src_mask=create_padding_mask(padded))
    assert torch.allclose(out, out_padded, atol=1e-5)
