import math

import pytest
import torch

import jetformer.model.flow as flow_module
from jetformer.model.attention import explicit_attention
from jetformer.model.flow import JetFlow, spatial_permutation
from jetformer.model.gmm import CFGDensity, DiagonalGMM, gmm_log_prob, gmm_params, standard_normal_nll
from jetformer.model.jetformer import JetFormer, count_parameters
from jetformer.model.patches import patchify, unpatchify
from jetformer.model.transformer import (
    GatedMLP,
    GemmaBlock,
    GemmaRMSNorm,
    KVCache,
    MultiQueryAttention,
    apply_rotary,
    rotary_tables,
)
from tests.conftest import tiny_config

# ---- patches ----------------------------------------------------------------------------------


def test_patchify_round_trips_and_orders_row_column_channel():
    images = torch.arange(2 * 3 * 4 * 4, dtype=torch.float32).reshape(2, 3, 4, 4)
    tokens = patchify(images, 2)
    assert tokens.shape == (2, 4, 12)
    assert tokens[0, 0, :3].tolist() == [images[0, c, 0, 0].item() for c in range(3)]
    torch.testing.assert_close(unpatchify(tokens, (4, 4), 2), images)
    with pytest.raises(ValueError, match="do not tile"):
        unpatchify(tokens, (8, 4), 4)
    with pytest.raises(ValueError, match="divisible"):
        unpatchify(tokens, (6, 4), 4)
    with pytest.raises(ValueError, match="divisible"):
        patchify(images, 3)


# ---- transformer components ---------------------------------------------------------------------


def test_rms_norm_matches_gemma_parameterization():
    norm = GemmaRMSNorm(8)
    x = torch.randn(3, 8)
    expected = x * torch.rsqrt(x.square().mean(-1, keepdim=True) + 1e-6)
    torch.testing.assert_close(norm(x), expected)
    with torch.no_grad():
        norm.scale.fill_(1.0)
    torch.testing.assert_close(norm(x), 2.0 * expected)


def test_gated_mlp_uses_tanh_gelu_gate():
    mlp = GatedMLP(4, 6)
    x = torch.randn(2, 4)
    gate, up = mlp.gate_up(x).split(6, dim=-1)
    torch.testing.assert_close(mlp(x), mlp.down(torch.nn.functional.gelu(gate, approximate="tanh") * up))


def test_rotary_is_relative():
    cos, sin = rotary_tables(8, 12)
    torch.testing.assert_close(cos[0], torch.ones(8))
    torch.testing.assert_close(sin[0], torch.zeros(8))
    torch.manual_seed(0)
    q, k = torch.randn(1, 1, 1, 8), torch.randn(1, 1, 1, 8)
    for m, n in ((3, 7), (0, 5), (9, 2)):
        rotated = (apply_rotary(q, cos[m], sin[m]) * apply_rotary(k, cos[n], sin[n])).sum()
        if n >= m:
            shifted = (apply_rotary(q, cos[0], sin[0]) * apply_rotary(k, cos[n - m], sin[n - m])).sum()
        else:
            shifted = (apply_rotary(q, cos[m - n], sin[m - n]) * apply_rotary(k, cos[0], sin[0])).sum()
        torch.testing.assert_close(rotated, shifted, atol=1e-5, rtol=1e-5)
    with pytest.raises(ValueError, match="even"):
        rotary_tables(7, 4)


@pytest.mark.parametrize("n_kv_heads", [1, 2, 4])
def test_grouped_attention_matches_expanded_kv_heads(n_kv_heads):
    torch.manual_seed(0)
    attention = MultiQueryAttention(16, 4, n_kv_heads)
    x = torch.randn(2, 5, 16)
    cos, sin = rotary_tables(4, 5)
    out = attention(x, cos, sin, cache=None, layer=0)
    q, k, v = attention.qkv(x).split((16, 4 * n_kv_heads, 4 * n_kv_heads), dim=-1)
    q = q.view(2, 5, 4, 4).transpose(1, 2)
    k = k.view(2, 5, n_kv_heads, 4).transpose(1, 2).repeat_interleave(4 // n_kv_heads, dim=1)
    v = v.view(2, 5, n_kv_heads, 4).transpose(1, 2).repeat_interleave(4 // n_kv_heads, dim=1)
    reference = torch.nn.functional.scaled_dot_product_attention(
        apply_rotary(q, cos, sin), apply_rotary(k, cos, sin), v, is_causal=True
    )
    torch.testing.assert_close(out, attention.out(reference.transpose(1, 2).reshape(2, 5, 16)))
    with pytest.raises(ValueError, match="divisible"):
        MultiQueryAttention(16, 4, 3)


@pytest.mark.parametrize("causal", [True, False])
@pytest.mark.parametrize("n_kv_heads", [1, 2, 4])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_explicit_attention_matches_the_fused_kernel(causal, n_kv_heads, dtype):
    torch.manual_seed(0)
    queries = torch.randn(2, 4, 7, 8).to(dtype)
    keys = torch.randn(2, n_kv_heads, 7, 8).to(dtype)
    values = torch.randn(2, n_kv_heads, 7, 8).to(dtype)
    expected = torch.nn.functional.scaled_dot_product_attention(
        queries, keys, values, is_causal=causal, enable_gqa=n_kv_heads != 4
    )
    out = explicit_attention(queries, keys, values, causal=causal)
    assert out.dtype == dtype
    tolerance = {"atol": 2e-2, "rtol": 2e-2} if dtype == torch.bfloat16 else {}
    torch.testing.assert_close(out, expected, **tolerance)
    if causal:
        with pytest.raises(ValueError, match="query and key lengths"):
            explicit_attention(queries[:, :, -1:], keys, values, causal=True)
    else:  # a decode step attends from one query to every key
        step = explicit_attention(queries[:, :, -1:], keys, values, causal=False)
        torch.testing.assert_close(step, expected[:, :, -1:], **tolerance)


def test_kv_cache_slices_grow_per_layer_and_reject_overflow():
    cache = KVCache(num_layers=2, capacity=3)
    first = torch.arange(2 * 4, dtype=torch.float32).reshape(1, 1, 2, 4)
    for layer in range(2):
        keys, values = cache.append(layer, first + layer, first)
        torch.testing.assert_close(keys, first + layer)
        assert values.shape == (1, 1, 2, 4)
    cache.advance(2)
    third = torch.full((1, 1, 1, 4), 9.0)
    keys, _ = cache.append(1, third, third)
    torch.testing.assert_close(keys, torch.cat((first + 1, third), dim=2))
    cache.advance(1)
    with pytest.raises(ValueError, match="capacity"):
        cache.append(0, third, third)
    attention = MultiQueryAttention(8, 2, 1)
    cos, sin = rotary_tables(4, 4)
    with pytest.raises(ValueError, match="empty cache"):
        attention(torch.zeros(1, 2, 8), cos[:2], sin[:2], cache=cache, layer=0)


def test_decoder_block_checkpointing_matches_the_plain_path(config):
    torch.manual_seed(0)
    plain = JetFormer.from_config(config, "cpu")
    torch.manual_seed(0)
    checkpointed = JetFormer.from_config(tiny_config(model={"grad_checkpoint": True}), "cpu")
    labels, tokens = torch.tensor([1, 2]), torch.randn(2, plain.image_seq_len, plain.image_ar_dim)
    torch.manual_seed(3)
    out_plain = plain(labels, tokens)
    out_plain.sum().backward()
    torch.manual_seed(3)
    out_checkpointed = checkpointed(labels, tokens)
    out_checkpointed.sum().backward()
    torch.testing.assert_close(out_checkpointed, out_plain)
    for a, b in zip(plain.blocks.parameters(), checkpointed.blocks.parameters(), strict=True):
        torch.testing.assert_close(a.grad, b.grad)


# ---- flow ---------------------------------------------------------------------------------------


@pytest.mark.parametrize("kind", ["checkerboard", "checkerboard-inv", "hstripes", "vstripes", "vstripes-inv"])
def test_spatial_permutations_partition_the_grid(kind):
    permutation = spatial_permutation(kind, 4, 6)
    assert sorted(permutation.tolist()) == list(range(24))
    first = set(permutation[:12].tolist())
    if kind.startswith("checkerboard"):
        parity = {index for index in range(24) if (index // 6 + index % 6) % 2 == 0}
        assert first == (parity if not kind.endswith("-inv") else set(range(24)) - parity)
    elif kind == "hstripes":
        assert first == {index for index in range(24) if (index // 6) % 2 == 0}
    else:
        columns = {index for index in range(24) if (index % 6) % 2 == 0}
        assert first == (columns if not kind.endswith("-inv") else set(range(24)) - columns)


def test_stripes_require_even_grid_sides():
    assert set(spatial_permutation("vstripes", 3, 4)[:6].tolist()) == {0, 2, 4, 6, 8, 10}  # column stripes on 3x4
    with pytest.raises(ValueError, match="even patch-grid width"):
        spatial_permutation("vstripes", 4, 3)
    with pytest.raises(ValueError, match="even patch-grid height"):
        spatial_permutation("hstripes-inv", 3, 4)
    with pytest.raises(ValueError, match="Unknown spatial"):
        spatial_permutation("diagonal", 4, 4)


def _random_flow(**overrides):
    kwargs = dict(
        grid_size=(2, 2),
        token_dim=6,
        depth=3,
        block_depth=1,
        emb_dim=16,
        num_heads=2,
        kinds=("channels", "spatial"),
        seed=1,
    )
    kwargs.update(overrides)
    torch.manual_seed(0)
    flow = JetFlow(**kwargs)
    for coupling in flow.couplings:  # make the couplings non-trivial
        torch.nn.init.normal_(coupling.net.final_proj.weight, std=0.5)
    return flow


@pytest.mark.parametrize("kinds", [("channels",), ("channels", "spatial"), ("spatial", "channels", "channels")])
def test_flow_is_invertible_with_consistent_logdet(kinds):
    flow = _random_flow(kinds=kinds)
    tokens = torch.randn(3, 4, 6)
    latents, logdet = flow(tokens)
    recovered, inverse_logdet = flow.inverse(latents)
    torch.testing.assert_close(recovered, tokens, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(inverse_logdet, -logdet)
    jacobian = torch.autograd.functional.jacobian(lambda x: flow(x[None])[0].reshape(-1), tokens[0])
    torch.testing.assert_close(torch.linalg.slogdet(jacobian.reshape(24, 24))[1], logdet[0], atol=1e-4, rtol=1e-4)
    with pytest.raises(ValueError, match="Expected tokens"):
        flow(tokens[:, :3])


def test_composed_permutations_match_explicit_reordering():
    """Each channel coupling sees its own permutation of the canonical channels, exactly as if the
    tensor were permuted before it and un-permuted after it."""
    flow = _random_flow(kinds=("channels",), depth=3)
    tokens = torch.randn(2, 4, 6)
    latents, _ = flow(tokens)
    current = tokens
    for coupling in flow.couplings:
        ordered = current.index_select(-1, coupling.permutation)
        first, second = ordered.chunk(2, dim=-1)
        bias, scale, _ = coupling.net(first)
        transformed = torch.cat((first, (second + bias) * scale), dim=-1)
        current = transformed.index_select(-1, torch.argsort(coupling.permutation))
    torch.testing.assert_close(latents, current)


def test_flow_starts_as_the_identity():
    flow = JetFlow(grid_size=(2, 2), token_dim=6, depth=2, block_depth=1, emb_dim=16, num_heads=2, seed=0)
    tokens = torch.randn(2, 4, 6)
    latents, logdet = flow(tokens)
    torch.testing.assert_close(latents, tokens)
    torch.testing.assert_close(logdet, torch.zeros(2))


def test_scale_is_bounded_by_two_and_matches_the_logdet():
    flow = JetFlow(grid_size=(2, 2), token_dim=6, depth=1, block_depth=1, emb_dim=16, num_heads=2, seed=0)
    net = flow.couplings[0].net
    torch.manual_seed(0)
    torch.nn.init.normal_(net.final_proj.weight, std=1.0)
    _, scale, logdet = net(torch.randn(2, 4, 3) * 100)
    assert scale.max() <= 2.0 and scale.min() >= 0.0 and scale.max() > 1.9
    torch.testing.assert_close(logdet, scale.log().flatten(1).sum(1))


def test_flow_head_stays_fp32_under_autocast():
    flow = _random_flow()
    tokens = torch.randn(3, 4, 6)
    _, logdet_fp32 = flow(tokens)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        bias, scale, logdet = flow.couplings[0].net(tokens[..., :3])
        assert bias.dtype == scale.dtype == logdet.dtype == torch.float32
        latents, logdet_all = flow(tokens)
        recovered, _ = flow.inverse(latents)
    assert latents.dtype == torch.float32
    torch.testing.assert_close(recovered, tokens, atol=1e-4, rtol=1e-4)  # the same bf16 network is inverted exactly
    torch.testing.assert_close(logdet_all, logdet_fp32, atol=1.0, rtol=5e-2)


def test_flow_checkpointing_matches_the_plain_path(monkeypatch):
    kwargs = dict(grid_size=(2, 2), token_dim=6, depth=2, block_depth=2, emb_dim=16, num_heads=2, seed=1)
    torch.manual_seed(0)
    plain = JetFlow(**kwargs)
    torch.manual_seed(0)
    checkpointed = JetFlow(**kwargs, grad_checkpoint=True)
    for a, b in zip(plain.couplings, checkpointed.couplings, strict=True):
        torch.nn.init.normal_(a.net.final_proj.weight, std=0.5)
        b.net.final_proj.weight.data.copy_(a.net.final_proj.weight)
    calls = []
    original = flow_module.checkpoint.checkpoint
    monkeypatch.setattr(flow_module.checkpoint, "checkpoint", lambda *a, **k: (calls.append(1), original(*a, **k))[1])
    tokens = torch.randn(2, 4, 6, requires_grad=True)
    plain.train()
    checkpointed.train()
    latents, logdet = plain(tokens)
    (latents.sum() + logdet.sum()).backward()
    grads = [p.grad.clone() for p in plain.parameters()]
    latents, logdet = checkpointed(tokens)
    (latents.sum() + logdet.sum()).backward()
    assert len(calls) == 2  # one segment per coupling
    for g, p in zip(grads, checkpointed.parameters(), strict=True):
        torch.testing.assert_close(p.grad, g)
    checkpointed.eval()
    calls.clear()
    checkpointed(tokens)
    assert calls == []


def test_channel_permutations_are_seeded():
    def perms(seed):
        flow = JetFlow(grid_size=(2, 2), token_dim=6, depth=2, block_depth=1, emb_dim=16, num_heads=2, seed=seed)
        return [c.permutation.tolist() for c in flow.couplings]

    assert perms(3) == perms(3) and perms(3) != perms(4)
    from_flow = [
        c.permutation.tolist() for c in JetFormer.from_config(tiny_config(flow={"seed": 7}), "cpu").flow.couplings
    ]
    from_seed = [c.permutation.tolist() for c in JetFormer.from_config(tiny_config(seed=7), "cpu").flow.couplings]
    assert from_flow == from_seed
    with pytest.raises(ValueError, match="even token dimension"):
        JetFlow(grid_size=(2, 2), token_dim=5, depth=1, block_depth=1, emb_dim=16, num_heads=2)


# ---- mixture --------------------------------------------------------------------------------------


def test_gmm_log_prob_matches_torch_distributions():
    torch.manual_seed(0)
    logits = torch.randn(2, 3, 4 * (1 + 2 * 5))
    mix, means, log_scales = gmm_params(logits, 4, 5, scale_tol=1e-6)
    values = torch.randn(2, 3, 5)
    mixture = torch.distributions.MixtureSameFamily(
        torch.distributions.Categorical(logits=mix),
        torch.distributions.Independent(torch.distributions.Normal(means, log_scales.exp()), 1),
    )
    torch.testing.assert_close(gmm_log_prob(mix, means, log_scales, values), mixture.log_prob(values))
    torch.testing.assert_close(DiagonalGMM(mix, means, log_scales).log_prob(values), mixture.log_prob(values))


def test_gmm_parameter_layout_is_component_major():
    logits = torch.arange(2 * (1 + 2 * 3), dtype=torch.float32).reshape(1, 1, -1)
    mix, means, log_scales = gmm_params(logits, 2, 3, scale_tol=1e-6)
    assert mix.tolist() == [[[0.0, 1.0]]]
    assert means[0, 0].tolist() == [[2.0, 3.0, 4.0], [8.0, 9.0, 10.0]]
    raw = torch.tensor([[5.0, 6.0, 7.0], [11.0, 12.0, 13.0]])
    torch.testing.assert_close(
        log_scales[0, 0].exp(), (raw + (raw.square() + 4).sqrt()) / 2
    )  # log square_plus = asinh(r/2)
    assert gmm_params(torch.full((1, 1, 14), -1e4), 2, 3, scale_tol=1e-3)[2].min() == pytest.approx(math.log(1e-3))
    with pytest.raises(ValueError, match="mixture parameters"):
        gmm_params(logits, 3, 3, scale_tol=1e-6)


def test_standard_normal_nll_sums_every_value():
    values = torch.zeros(2, 3, 4)
    torch.testing.assert_close(standard_normal_nll(values), torch.full((2,), 12 * 0.5 * math.log(2 * math.pi)))
    assert standard_normal_nll(torch.zeros(2, 3, 0)).tolist() == [0.0, 0.0]


def test_component_draws_match_categorical_and_multinomial():
    torch.manual_seed(0)
    logits = torch.randn(7, 3, 16) * 5.0
    pdf = DiagonalGMM(logits, torch.zeros(7, 3, 16, 2), torch.zeros(7, 3, 16, 2))
    for seed in range(20):
        torch.manual_seed(seed)
        ours = pdf.sample_component()
        torch.manual_seed(seed)
        categorical = torch.distributions.Categorical(logits=logits).sample()
        torch.manual_seed(seed)
        multinomial = torch.multinomial(torch.softmax(logits.reshape(-1, 16), -1), 1, replacement=True).reshape(7, 3)
        assert torch.equal(ours, categorical) and torch.equal(ours, multinomial)
    assert pdf.sample().shape == (7, 3, 2) and pdf.mean().shape == (7, 3, 2) and pdf.mode().shape == (7, 3, 2)
    assert torch.equal(pdf.mode(), pdf.component(logits.argmax(-1))[0])
    assert torch.equal(pdf[0::2].mix_logits, logits[0::2])


def test_cfg_density_is_the_analytic_guided_gaussian():
    cond = DiagonalGMM(
        torch.tensor([[[0.0, 1.0]]]), torch.tensor([[[[1.0], [3.0]]]]), torch.tensor([[[[1.0], [0.5]]]]).log()
    )
    uncond = DiagonalGMM(torch.zeros(1, 1, 2), torch.tensor([[[[0.0], [2.0]]]]), torch.tensor([[[[2.0], [1.0]]]]).log())
    guided = CFGDensity(cond, uncond, weight=1.0)
    mean, scale = guided.guided(torch.tensor([[1]]))
    precision = 2.0 / 0.5**2 - 1.0 / 1.0**2
    torch.testing.assert_close(mean.reshape(()), torch.tensor((2.0 * 3.0 / 0.5**2 - 2.0 / 1.0) / precision))
    torch.testing.assert_close(scale.reshape(()), torch.tensor(precision**-0.5))
    torch.testing.assert_close(guided.mean(), mean)  # the highest-weight conditional component is 1
    torch.testing.assert_close(guided.mode(), mean)
    torch.manual_seed(0)
    assert guided.sample().shape == (1, 1, 1)
    # A non-positive guided precision falls back to the conditional component.
    narrow = DiagonalGMM(torch.zeros(1, 1, 2), uncond.means, torch.full((1, 1, 2, 1), 0.1).log())
    fallback_mean, fallback_scale = CFGDensity(cond, narrow, weight=1.0).guided(torch.tensor([[1]]))
    torch.testing.assert_close(fallback_mean.reshape(()), torch.tensor(3.0))
    torch.testing.assert_close(fallback_scale.reshape(()), torch.tensor(0.5))
    with pytest.raises(ValueError, match="non-negative"):
        CFGDensity(cond, uncond, weight=-1.0)
    with pytest.raises(ValueError, match="identical shapes"):
        CFGDensity(cond, DiagonalGMM(torch.zeros(1, 1, 2), torch.zeros(1, 1, 2, 2), torch.zeros(1, 1, 2, 2)), 1.0)


# ---- JetFormer ----------------------------------------------------------------------------------


def test_special_tokens_follow_the_classes(model):
    assert (model.bos_id, model.boi_id, model.nolabel_id, model.vocab_size) == (10, 11, 12, 13)
    assert model.max_seq_len == model.prefix_len + model.image_seq_len - 1
    labels = torch.tensor([3, 5])
    tokens = model.embed_prefix(labels)
    assert tokens.shape == (2, model.prefix_len, model.d_model)
    torch.testing.assert_close(tokens[0, 0], model.token_emb.weight[model.bos_id])
    torch.testing.assert_close(tokens[0, 1], model.token_emb.weight[3])
    torch.testing.assert_close(tokens[0, 2], model.token_emb.weight[3 + model.vocab_size])
    torch.testing.assert_close(tokens[0, -1], model.token_emb.weight[model.boi_id])
    dropped = model.embed_prefix(labels, torch.tensor([False, True]))
    torch.testing.assert_close(dropped[0], tokens[0])
    torch.testing.assert_close(dropped[1, 1], model.token_emb.weight[model.nolabel_id])
    with pytest.raises(ValueError, match=r"\[B\]"):
        model.embed_prefix(labels[None])
    with pytest.raises(ValueError, match="image_tokens must have shape"):
        model(labels, torch.zeros(2, 3, model.image_ar_dim))


@pytest.mark.parametrize("autocast", [False, True])
def test_teacher_forcing_matches_incremental_decoding(model, autocast):
    model.eval()
    torch.manual_seed(0)
    torch.nn.init.normal_(model.image_head.weight, std=0.05)
    labels = torch.tensor([1, 4, 8])
    tokens = torch.randn(3, model.image_seq_len, model.image_ar_dim)
    context = torch.autocast("cpu", dtype=torch.bfloat16) if autocast else torch.no_grad()
    with context:
        full = model(labels, tokens)
        hidden, cache = model.prefill(labels)
        steps = [model.head_logits(hidden)]
        for position in range(model.image_seq_len - 1):
            hidden = model.decode_step(tokens[:, position : position + 1], cache)
            assert hidden.dtype == torch.float32  # the residual stream stays fp32 while decoding
            steps.append(model.head_logits(hidden))
    tolerance = {"atol": 1e-1, "rtol": 1e-1} if autocast else {"atol": 1e-5, "rtol": 1e-5}
    torch.testing.assert_close(torch.cat(steps, dim=1).float(), full.float(), **tolerance)
    assert cache.length == model.max_seq_len
    assert cache.keys.dtype == (torch.bfloat16 if autocast else torch.float32)


def test_gmm_head_initialization_spreads_component_means(model):
    means = model.image_head.bias[model.num_mixtures :].view(model.num_mixtures, 2, model.image_ar_dim)[:, 0]
    assert means.std() > 0.0
    torch.testing.assert_close(means.mean(dim=0), torch.zeros(model.image_ar_dim), atol=1e-6, rtol=0)
    assert model.image_head.weight.abs().max() == 0.0
    counts = count_parameters(model)
    assert counts["total"] == counts["flow"] + counts["transformer"] > 0


def test_encode_and_decode_invert_each_other(model):
    for coupling in model.flow.couplings:
        torch.nn.init.normal_(coupling.net.final_proj.weight, std=0.5)
    images = torch.randint(0, 256, (2, 3, 32, 32)).float() / 255
    latents, logdet = model.encode_images(images * 2 - 1)
    assert logdet.shape == (2,) and latents.shape == (2, model.image_seq_len, model.image_token_dim)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        decoded = model.decode_tokens_to_images(latents)
    torch.testing.assert_close(decoded, images, atol=2e-4, rtol=0)
    extreme = model.decode_tokens_to_images(latents * 1e3)
    assert extreme.min() >= 0.0 and extreme.max() <= 1.0


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"num_classes": 0}, "positive"),
        ({"image_ar_dim": 200}, "image_ar_dim"),
        ({"patch_size": 5}, "divisible"),
        ({"patch_size": 4}, "patch-token grid"),
    ],
)
def test_model_constructor_validates_inputs(config, overrides, message):
    flow = JetFormer.from_config(config, "cpu").flow
    kwargs = dict(
        num_classes=10, input_size=(32, 32), patch_size=8, image_ar_dim=6, d_model=32, n_layers=2, n_heads=2, n_kv_heads=1,
        d_ff=64, num_mixtures=4, gmm_mean_init_std=0.02, scale_tol=1e-6, dropout=0.0, num_class_repeats=2,
        grad_checkpoint=False, flow=flow,
    )  # fmt: skip
    kwargs.update(overrides)
    with pytest.raises(ValueError, match=message):
        JetFormer(**kwargs)


def test_gemma_block_default_layer_only_matters_with_a_cache():
    block = GemmaBlock(8, 2, 1, 16, 0.0)
    cos, sin = rotary_tables(4, 3)
    x = torch.randn(1, 3, 8)
    torch.testing.assert_close(block(x, cos, sin), block(x, cos, sin, cache=None, layer=5))


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16, torch.float64])
def test_unit_floor_retains_original_half_derivative_at_signed_zero(dtype):
    logits = torch.zeros(4, 3, dtype=dtype)
    logits[:, 2] = torch.tensor([-2.0, -0.0, 0.0, 2.0], dtype=dtype)
    logits.requires_grad_()
    logs = gmm_params(logits, 1, 1, scale_tol=1.0)[2]
    gradient = torch.autograd.grad(logs, logits, torch.ones_like(logs))[0][:, 2]
    assert torch.equal(gradient[1:3], torch.full((2,), 0.5, dtype=dtype))
    assert gradient[0].item() == 0.0
    assert torch.equal(gradient[3], (torch.tensor(0.5) / torch.sqrt(torch.tensor(2.0))).to(dtype))


@pytest.mark.parametrize("upstream", [1.0, -1.0, 0.0, -0.0, math.inf, -math.inf, math.nan])
def test_nan_is_preserved_through_the_actual_raw_scale_gradient_chain(upstream):
    logits = torch.tensor([[0.0, 0.5, math.nan]], requires_grad=True)
    logs = gmm_params(logits, 1, 1, scale_tol=1.0)[2]
    gradient = torch.autograd.grad(logs, logits, torch.full_like(logs, upstream))[0]
    assert logs.isnan().all() and gradient[0, 2].isnan()


def test_signed_zero_consumers_have_identical_outputs_and_input_gradients():
    """Log-scale sign is exposed by component(), but exp/density/CFG arithmetic is checked here."""
    mix = torch.tensor([[[0.25, -0.5]]])
    means = torch.tensor([[[[0.5, -0.5], [1.0, -1.0]]]])
    values = torch.tensor([[[0.125, -0.25]]])
    calls = []
    for negative in (False, True):
        logs = torch.full_like(means, -0.0 if negative else 0.0).requires_grad_()
        pdf = DiagonalGMM(mix, means, logs)
        conditional = pdf
        unconditional = DiagonalGMM(mix, means * 0.75, logs)
        density = gmm_log_prob(mix, means, logs, values)
        guided = CFGDensity(conditional, unconditional, 2.0)
        location, scale = guided.guided(torch.zeros(1, 1, dtype=torch.long))
        torch.manual_seed(74)
        sample = pdf.sample()
        torch.manual_seed(75)
        guided_sample = guided.sample()
        scalar = density.sum() + sample.sum() + guided_sample.sum() + scale.sum() + location.sum()
        gradient = torch.autograd.grad(scalar, logs)[0]
        calls.append((density, sample, guided_sample, location, scale, gradient))
    for first, second in zip(*calls, strict=True):
        assert torch.equal(first, second)
        assert torch.equal(first.signbit(), second.signbit())
