import pytest
import torch

from jetformer.config import SamplingConfig
from jetformer.model.gmm import DiagonalGMM
from jetformer.sampling import (
    GENERATION_CHUNK,
    _draw,
    balanced_class_ids,
    generate_in_chunks,
    parse_class_ids,
    sample_batch,
    sample_images,
    save_samples,
)


def test_class_id_helpers():
    assert balanced_class_ids(5, 3) == [0, 1, 2, 0, 1] and balanced_class_ids(0, 3) == []
    assert parse_class_ids(None, 4, 10) == [0, 1, 2, 3] and parse_class_ids(" ", 2, 10) == [0, 1]
    assert parse_class_ids("3,5", 5, 10) == [3, 5, 3, 5, 3]
    for raw, message in (("a,b", "Invalid class id list"), (",", "at least one"), ("3,12", "outside")):
        with pytest.raises(ValueError, match=message):
            parse_class_ids(raw, 2, 10)
    with pytest.raises(ValueError):
        balanced_class_ids(-1, 3)
    with pytest.raises(ValueError):
        parse_class_ids(None, 0, 10)
    pdf = DiagonalGMM(torch.zeros(1, 1, 2), torch.zeros(1, 1, 2, 1), torch.zeros(1, 1, 2, 1))
    with pytest.raises(ValueError, match="Unknown sample method"):
        _draw(pdf, "median")


@pytest.fixture
def sampler(model):
    torch.manual_seed(0)
    torch.nn.init.normal_(model.image_head.weight, std=0.05)
    for coupling in model.flow.couplings:
        torch.nn.init.normal_(coupling.net.final_proj.weight, std=0.1)
    return model.eval()


@pytest.mark.parametrize("cfg_mode", ["density", "interp", "none"])
@pytest.mark.parametrize("method", ["sample", "mean", "mode"])
def test_sample_batch_is_deterministic_per_seed(sampler, cfg_mode, method):
    sampling = SamplingConfig(cfg_weight=1.5, cfg_mode=cfg_mode, sample_method=method, temperature=0.8)
    labels = torch.tensor([0, 7, 3])
    torch.manual_seed(1)
    first = sample_batch(sampler, labels, sampling)
    torch.manual_seed(1)
    second = sample_batch(sampler, labels, sampling)
    assert first.shape == (3, 3, 32, 32) and first.dtype == torch.uint8 and torch.equal(first, second)
    torch.manual_seed(2)
    assert not torch.equal(first, sample_batch(sampler, labels, sampling))  # residual channels are always sampled


def test_guided_rows_are_independent_of_batch_composition(sampler, monkeypatch):
    with torch.no_grad():  # decisive mixture weights, so the guided component never flips on rounding noise
        sampler.image_head.bias[: sampler.num_mixtures] = torch.arange(sampler.num_mixtures) * 5.0
    seen = []
    original = sampler.decode_tokens_to_images
    monkeypatch.setattr(
        sampler, "decode_tokens_to_images", lambda latents: (seen.append(latents.clone()), original(latents))[1]
    )
    sampling = SamplingConfig(cfg_weight=2.0, cfg_mode="density", sample_method="mean")
    sample_batch(sampler, torch.tensor([2, 5]), sampling)
    sample_batch(sampler, torch.tensor([5]), sampling)
    ar_dim = sampler.image_ar_dim
    assert seen[0].dtype == torch.float32
    torch.testing.assert_close(seen[0][1, :, :ar_dim], seen[1][0, :, :ar_dim], atol=1e-4, rtol=1e-4)
    assert not torch.allclose(seen[0][0, :, :ar_dim], seen[0][1, :, :ar_dim])  # different classes differ


def test_sample_batch_validates_inputs(sampler):
    sampling = SamplingConfig()
    sampler.train()
    with pytest.raises(ValueError, match="eval"):
        sample_batch(sampler, torch.tensor([1]), sampling)
    sampler.eval()
    with pytest.raises(ValueError, match=r"\[B\]"):
        sample_batch(sampler, torch.tensor([[1]]), sampling)
    with pytest.raises(ValueError, match="Class ids"):
        sample_batch(sampler, torch.tensor([10]), sampling)
    assert sample_batch(sampler, torch.tensor([], dtype=torch.long), sampling).shape == (0, 3, 32, 32)


def test_sample_images_batches_and_restores_the_mode(sampler):
    sampler.train()
    torch.manual_seed(0)
    images = sample_images(sampler, [0, 1, 2, 3, 4], SamplingConfig(cfg_weight=0.0), batch_size=2)
    assert images.shape == (5, 3, 32, 32) and images.device.type == "cpu" and sampler.training
    assert sample_images(sampler, [], SamplingConfig(), batch_size=2).shape == (0, 3, 32, 32)
    with pytest.raises(ValueError, match="batch_size"):
        sample_images(sampler, [0], SamplingConfig(), batch_size=0)


def test_generate_in_chunks_rounds_chunks_to_the_batch_size(sampler):
    sampling = SamplingConfig(cfg_weight=0.0, sample_method="mean")
    chunks = list(generate_in_chunks(sampler, list(range(10)), sampling, batch_size=4, chunk=5))
    assert [(start, images.shape[0]) for start, images in chunks] == [(0, 8), (8, 2)]
    assert GENERATION_CHUNK % 64 == 0


def test_save_samples_names_files_by_class(tmp_path):
    images = torch.zeros(3, 3, 4, 4, dtype=torch.uint8)
    directory = save_samples(images, [0, 1, 0], ["cat", "a b/c"], tmp_path / "out", start_index=5)
    assert sorted(p.name for p in directory.glob("*.png")) == [
        "00005_cat.png",
        "00006_a_b_c.png",
        "00007_cat.png",
        "_grid.png",
    ]
    assert not (save_samples(images[:0], [], ["cat"], tmp_path / "empty") / "_grid.png").exists()
    assert not (save_samples(images, [0, 0, 0], ["cat"], tmp_path / "nogrid", grid=False) / "_grid.png").exists()
    with pytest.raises(ValueError):
        save_samples(images, [0], ["cat"], tmp_path / "short")


def test_sampling_loop_avoids_host_synchronisation(sampler, monkeypatch):
    syncs = []
    for name in ("item", "tolist", "numpy", "cpu", "__bool__", "__int__", "__float__", "__index__"):
        original = getattr(torch.Tensor, name)

        def spy(self, *args, _name=name, _original=original, **kwargs):
            syncs.append(_name)
            return _original(self, *args, **kwargs)

        monkeypatch.setattr(torch.Tensor, name, spy)
    sample_batch(sampler, torch.tensor([1, 2]), SamplingConfig(cfg_weight=2.0, cfg_mode="density"))
    assert set(syncs) <= {"__int__"} and len(syncs) <= 2  # only the label range check touches the host


def test_autocast_sampling_keeps_the_flow_inverse_in_fp32(sampler, monkeypatch):
    seen = {}
    original = sampler.flow.inverse

    def spy(latents):
        seen.update(dtype=latents.dtype, autocast=torch.is_autocast_enabled("cpu"))
        return original(latents)

    monkeypatch.setattr(sampler.flow, "inverse", spy)
    torch.manual_seed(0)
    images = sample_batch(sampler, torch.tensor([1]), SamplingConfig(), autocast_dtype=torch.bfloat16)
    assert images.dtype == torch.uint8 and images.shape == (1, 3, 32, 32)
    assert seen == {"dtype": torch.float32, "autocast": False}
