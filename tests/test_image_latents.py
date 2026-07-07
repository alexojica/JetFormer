import pytest
import torch

from jetformer.latents import PatchPCA
from jetformer.utils.image import patchify, unpatchify


def test_patchify_unpatchify_roundtrip():
    image = torch.arange(2 * 4 * 4 * 3, dtype=torch.float32).view(2, 4, 4, 3)

    tokens = patchify(image, patch_size=2)
    restored = unpatchify(tokens, H=4, W=4, patch_size=2)

    assert tokens.shape == (2, 4, 12)
    torch.testing.assert_close(restored, image)


def test_patchify_rejects_non_divisible_sizes():
    with pytest.raises(ValueError, match="divisible"):
        patchify(torch.zeros(1, 5, 4, 3), patch_size=2)


def test_patch_pca_depth_to_seq_roundtrip_without_pca():
    patch_pca = PatchPCA(
        input_size=(4, 4),
        patch_size=2,
        depth_to_seq=3,
        skip_pca=True,
    )
    image = torch.linspace(-1.0, 1.0, steps=3 * 4 * 4).view(1, 3, 4, 4)

    mu, logvar = patch_pca.encode(image, train=False)
    restored = patch_pca.decode(mu, train=False)

    assert mu.shape == (1, 12, 4)
    assert logvar.shape == mu.shape
    torch.testing.assert_close(restored, image)


def test_patch_pca_missing_init_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        PatchPCA(pca_init_file=str(tmp_path / "missing.npz"), skip_pca=False)
