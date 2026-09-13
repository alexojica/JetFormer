import math

import pytest
import torch

from jetformer.model.gmm import standard_normal_nll
from jetformer.model.jetformer import JetFormer
from jetformer.model.patches import patchify
from jetformer.training.objective import JetFormerObjective, rgb_noise_sigma
from tests.conftest import tiny_config


def test_rgb_noise_schedule_is_cosine_between_scale_and_minimum():
    sigma = rgb_noise_sigma(torch.tensor([0.0, 50.0, 100.0, 150.0]), 100, noise_scale=32.0, noise_min=2.0)
    torch.testing.assert_close(sigma, torch.tensor([32.0, 17.0, 2.0, 2.0]))


@pytest.mark.parametrize("ar_dim", [6, 192])
def test_bpd_composition_with_identity_flow(ar_dim, batch):
    """At initialisation the flow is the identity, so the objective reduces to closed-form terms."""
    images, labels = batch
    torch.manual_seed(0)
    model = JetFormer.from_config(tiny_config(image={"ar_dim": ar_dim}), "cpu")
    objective = JetFormerObjective(model, tiny_config().training, dequant_noise=False).eval()
    with torch.no_grad():
        output = objective(images, labels, torch.tensor(0.0), 10, rgb_noise=False, diagnostics=True)
        tokens = patchify(images.float() / 127.5 - 1.0, model.patch_size)
        residual_nll = standard_normal_nll(tokens[..., ar_dim:])
        ar_nll = -model.pdf_from_logits(model(labels, tokens[..., :ar_dim])).log_prob(tokens[..., :ar_dim]).sum(1)
        num_subpixels = 3 * 32 * 32
        expected = (ar_nll + residual_nll) / num_subpixels / math.log(2) + math.log(127.5) / math.log(2)
    torch.testing.assert_close(output["loss"], expected.mean())
    torch.testing.assert_close(output["flow_bpd"], torch.tensor(-math.log(127.5) / math.log(2)))
    torch.testing.assert_close(output["loss"], output["ar_bpd"] + output["residual_bpd"] - output["flow_bpd"])
    assert output["sigma_rgb"] == 0.0
    assert set(output) >= {"gmm_entropy_nats", "flow_logdet_per_patch", "residual_tokens_rms", "ar_tokens_rms"}
    if ar_dim == 192:
        assert output["residual_bpd"] == 0.0 and output["residual_tokens_rms"] == 0.0


def test_noise_terms_apply_only_in_training_mode(batch):
    images, labels = batch
    torch.manual_seed(0)
    model = JetFormer.from_config(tiny_config(model={"dropout": 0.0}), "cpu")
    noisy = JetFormerObjective(
        model, tiny_config(training={"input_noise_std": 0.5}).training, dequant_noise=False, drop_labels_probability=1.0
    )
    clean = JetFormerObjective(model, tiny_config(training={"input_noise_std": 0.0}).training, dequant_noise=False)
    noisy.eval(), clean.eval()
    torch.manual_seed(0)
    before = torch.rand(1)
    torch.manual_seed(0)
    loss_noisy = noisy(images, labels, torch.tensor(0.0), 10, rgb_noise=False)["loss"]
    after = torch.rand(1)  # eval mode draws no noise, so the stream is untouched
    torch.testing.assert_close(after, before)
    torch.testing.assert_close(loss_noisy, clean(images, labels, torch.tensor(0.0), 10, rgb_noise=False)["loss"])

    seen = {}
    original_forward = model.forward

    def spy(labels_, tokens, drop=None):
        seen["tokens"], seen["drop"] = tokens, drop
        return original_forward(labels_, tokens, drop)

    model.forward = spy
    noisy.train()
    noisy(images, labels, torch.tensor(0.0), 10, rgb_noise=False)
    clean_tokens = patchify(images.float() / 127.5 - 1.0, model.patch_size)[..., : model.image_ar_dim]
    assert seen["drop"].all() and not torch.allclose(seen["tokens"], clean_tokens)


def test_teacher_forcing_noise_is_shared_by_input_and_target(batch, monkeypatch):
    images, labels = batch
    torch.manual_seed(0)
    model = JetFormer.from_config(tiny_config(model={"dropout": 0.0}), "cpu")
    objective = JetFormerObjective(model, tiny_config(training={"input_noise_std": 0.3}).training, dequant_noise=False)
    objective.train()
    seen = {}
    original_forward = model.forward

    def spy(labels_, tokens, drop=None):
        seen["input"] = tokens
        return original_forward(labels_, tokens, drop)

    monkeypatch.setattr(model, "forward", spy)
    output = objective(images, labels, torch.tensor(0.0), 10, rgb_noise=False, diagnostics=True)
    with torch.no_grad():
        logits = original_forward(labels, seen["input"])
        ar_nll = -model.pdf_from_logits(logits).log_prob(seen["input"]).sum(1)
        latents, _ = model.encode_images(images.float() / 127.5 - 1.0)
        residual = standard_normal_nll(latents[..., model.image_ar_dim :])
    expected = ((ar_nll + residual) / (3 * 32 * 32) / math.log(2)).mean()
    torch.testing.assert_close(output["ar_bpd"] + output["residual_bpd"], expected, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(output["ar_tokens_rms"], seen["input"].square().mean().sqrt())


def test_loss_backpropagates_to_flow_and_transformer(model, batch):
    images, labels = batch
    objective = JetFormerObjective(model, tiny_config().training).train()
    objective(images, labels, torch.tensor(0.0), 10)["loss"].backward()
    # The zero-initialised head receives gradient first; the decoder only once the head is non-zero.
    assert model.image_head.weight.grad.abs().sum() > 0
    assert model.flow.couplings[0].net.final_proj.weight.grad.abs().sum() > 0
    assert model.blocks[0].attention.qkv.weight.grad.abs().sum() == 0
    model.zero_grad()
    torch.nn.init.normal_(model.image_head.weight, std=0.01)
    objective(images, labels, torch.tensor(0.0), 10)["loss"].backward()
    assert model.blocks[0].attention.qkv.weight.grad.abs().sum() > 0


def test_objective_rejects_non_uint8_images(model, batch):
    images, labels = batch
    objective = JetFormerObjective(model, tiny_config().training)
    with pytest.raises(ValueError, match="uint8"):
        objective(images.float(), labels, torch.tensor(0.0), 10)
