import pytest
import torch

from transformer.decoder_lm import POSITIONAL_ENCODINGS, DecoderOnlyLM, KVCache


def small_model(pe):
    torch.manual_seed(0)
    return DecoderOnlyLM(24, 32, 4, 2, 64, pe, max_seq_len=64).eval()


@pytest.mark.parametrize("pe", POSITIONAL_ENCODINGS)
def test_incremental_logits_match_full_forward(pe):
    model = small_model(pe)
    tokens = torch.randint(4, 24, (2, 12))
    full = model(tokens)
    cache = KVCache()
    first = model(tokens[:, :5], cache=cache)
    steps = [model(tokens[:, t:t + 1], cache=cache) for t in range(5, 12)]
    incremental = torch.cat([first, *steps], dim=1)
    assert cache.length == 12
    assert torch.allclose(full, incremental, atol=1e-5)


@pytest.mark.parametrize("pe", POSITIONAL_ENCODINGS)
def test_cached_generation_matches_uncached(pe):
    model = small_model(pe)
    prompt = torch.randint(4, 24, (3, 9))
    assert torch.equal(model.generate(prompt, 20, use_cache=True), model.generate(prompt, 20, use_cache=False))


def test_cache_respects_max_seq_len():
    model = small_model("rope")
    cache = KVCache()
    model(torch.randint(4, 24, (1, 60)), cache=cache)
    with pytest.raises(ValueError):
        model(torch.randint(4, 24, (1, 5)), cache=cache)


def test_generating_zero_tokens_returns_empty():
    out = small_model("none").generate(torch.randint(4, 24, (2, 5)), 0)
    assert out.shape == (2, 0)
