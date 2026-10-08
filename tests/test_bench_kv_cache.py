import torch

from scripts.bench_kv_cache import benchmark
from transformer.decoder_lm import DecoderOnlyLM


def test_benchmark_reports_one_row_per_length():
    torch.manual_seed(0)
    model = DecoderOnlyLM(24, 16, 2, 1, 32, "rope", max_seq_len=32).eval()
    rows = benchmark(model, prompt_len=4, new_tokens=(2, 5), batch_size=2, reps=1)
    assert [r["new_tokens"] for r in rows] == [2, 5]
    assert all(r["cached_s"] > 0 and r["uncached_s"] > 0 and r["speedup"] > 0 for r in rows)
