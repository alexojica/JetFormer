from importlib import import_module
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

attention_module = import_module("jetformer.model.attention")


def metadata(device="mps", dtype=torch.float32, heads=4, batch=2):
    return SimpleNamespace(device=torch.device(device), dtype=dtype, shape=(batch, heads, 7, 8))


@pytest.mark.parametrize(
    ("device", "dtype", "heads", "grad", "autocast", "expected"),
    [
        ("mps", torch.float32, 4, False, False, True),
        ("mps", torch.float32, 4, True, False, False),
        ("mps", torch.float32, 4, False, True, False),
        ("mps", torch.bfloat16, 4, False, False, False),
        ("mps", torch.float16, 4, False, False, False),
        ("mps", torch.float32, 1, False, False, False),
        ("cpu", torch.float32, 4, False, False, False),
        ("cuda", torch.float32, 4, False, False, False),
    ],
)
def test_mps_inference_bmm_scope(device, dtype, heads, grad, autocast, expected):
    queries = metadata(device, dtype, heads)
    keys, values = metadata(device, dtype, 1), metadata(device, dtype, 1)
    with torch.set_grad_enabled(grad), patch.object(torch, "is_autocast_enabled", return_value=autocast):
        assert attention_module._use_mps_inference_bmm(queries, keys, values) is expected


@pytest.mark.parametrize(
    "changed", ["key_dtype", "value_dtype", "key_heads", "value_heads", "key_batch", "value_batch"]
)
def test_mps_inference_bmm_requires_matching_fp32_single_kv_head_batches(changed):
    q, k, v = metadata(), metadata(heads=1), metadata(heads=1)
    target = k if changed.startswith("key") else v
    if changed.endswith("dtype"):
        target.dtype = torch.bfloat16
    elif changed.endswith("heads"):
        target.shape = (2, 2, 7, 8)
    else:
        target.shape = (1, 1, 7, 8)
    with torch.no_grad(), patch.object(torch, "is_autocast_enabled", return_value=False):
        assert not attention_module._use_mps_inference_bmm(q, k, v)


@pytest.mark.parametrize(("causal", "length"), [(True, 7), (False, 7), (False, 1)])
def test_folded_layout_preserves_attention_on_strided_offset_views(monkeypatch, causal, length):
    generator = torch.Generator().manual_seed(8301)
    q = torch.randn(2, length, 4, 10, generator=generator)[..., 1:9].transpose(1, 2)
    k = torch.randn(2, 1, 13, 10, generator=generator)[:, :, 2:9, 1:9]
    v = torch.randn(2, 1, 13, 7, generator=generator)[:, :, 2:9, 1:6]
    assert not q.is_contiguous() and not k.is_contiguous() and not v.is_contiguous()
    assert all(value.storage_offset() > 0 for value in (q, k, v))
    before = tuple(value.clone() for value in (q, k, v))
    expected = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=causal, enable_gqa=True)
    # Exercise only the layout branch on CPU; the separate scope tests retain the real device gate.
    monkeypatch.setattr(attention_module, "_use_mps_inference_bmm", lambda *_: True)
    with torch.no_grad(), patch.object(torch, "bmm", wraps=torch.bmm) as bmm:
        actual = attention_module.explicit_attention(q, k, v, causal=causal)
    assert bmm.call_count == 2
    assert actual.shape == (2, 4, length, 5) and actual.dtype == torch.float32
    torch.testing.assert_close(actual, expected)
    assert all(torch.equal(value, original) for value, original in zip((q, k, v), before, strict=True))


def test_folded_causal_length_error_still_precedes_kernel_dispatch(monkeypatch):
    monkeypatch.setattr(attention_module, "_use_mps_inference_bmm", lambda *_: True)
    with patch.object(torch, "bmm", side_effect=AssertionError("No kernel should run")):
        with pytest.raises(ValueError, match="query and key lengths"):
            attention_module.explicit_attention(
                torch.empty(2, 4, 1, 8), torch.empty(2, 1, 7, 8), torch.empty(2, 1, 7, 8), causal=True
            )


@pytest.mark.parametrize(("batch", "length", "source_length"), [(0, 7, 7), (2, 0, 0), (2, 1, 0)])
def test_folded_empty_axes_preserve_explicit_output_shape(monkeypatch, batch, length, source_length):
    monkeypatch.setattr(attention_module, "_use_mps_inference_bmm", lambda *_: True)
    with torch.no_grad():
        actual = attention_module.explicit_attention(
            torch.empty(batch, 4, length, 8),
            torch.empty(batch, 1, source_length, 8),
            torch.empty(batch, 1, source_length, 5),
            causal=length == source_length,
        )
    assert actual.shape == (batch, 4, length, 5)
    assert torch.count_nonzero(actual) == 0


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_public_attention_keeps_sdpa_dispatch(device):
    q, k, v = metadata(device), metadata(device, heads=1), metadata(device, heads=1)
    with patch.object(attention_module.F, "scaled_dot_product_attention", return_value="sdpa") as sdpa:
        assert attention_module.attention(q, k, v, causal=True) == "sdpa"
    sdpa.assert_called_once_with(q, k, v, is_causal=True, enable_gqa=True)
