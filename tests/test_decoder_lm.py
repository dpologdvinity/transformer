import pytest
import torch

from transformer.decoder_lm import POSITIONAL_ENCODINGS, DecoderOnlyLM


def small_model(pe, **kwargs):
    torch.manual_seed(0)
    return DecoderOnlyLM(24, 32, 4, 2, 64, pe, **kwargs).eval()


@pytest.mark.parametrize("pe", POSITIONAL_ENCODINGS)
def test_future_tokens_do_not_change_past_logits(pe):
    model = small_model(pe)
    a = torch.randint(4, 24, (2, 12))
    b = a.clone()
    b[:, 6:] = torch.randint(4, 24, (2, 6))
    assert torch.allclose(model(a)[:, :6], model(b)[:, :6], atol=1e-5)


@pytest.mark.parametrize("pe", POSITIONAL_ENCODINGS)
def test_logits_shape(pe):
    assert small_model(pe)(torch.randint(4, 24, (2, 12))).shape == (2, 12, 24)


def test_encodings_differ_in_position_sensitivity():
    # With identical tokens, NoPE cannot tell positions apart (attention averages
    # identical values); an absolute encoding can.
    tokens = torch.full((1, 6), 5)
    nope = small_model("none")(tokens)
    sinusoidal = small_model("sinusoidal")(tokens)
    assert torch.allclose(nope[0, 0], nope[0, 5], atol=1e-5)
    assert not torch.allclose(sinusoidal[0, 0], sinusoidal[0, 5], atol=1e-5)


def test_unknown_encoding_raises():
    with pytest.raises(ValueError):
        DecoderOnlyLM(24, 32, 4, 2, 64, "learned")


def test_too_long_input_raises():
    model = small_model("sinusoidal", max_seq_len=16)
    with pytest.raises(ValueError):
        model(torch.randint(4, 24, (1, 17)))


def test_generate_shape_and_determinism():
    model = small_model("rope")
    prompt = torch.randint(4, 24, (2, 7))
    first = model.generate(prompt, 5)
    assert first.shape == (2, 5)
    assert torch.equal(first, model.generate(prompt, 5))


def test_attention_maps_one_per_layer():
    model = small_model("alibi")
    model(torch.randint(4, 24, (3, 9)))
    maps = model.attention_maps()
    assert len(maps) == 2
    for m in maps:
        assert m.shape == (3, 4, 9, 9)
        assert torch.allclose(m.sum(-1), torch.ones(3, 4, 9), atol=1e-5)
