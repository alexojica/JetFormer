"""Native serializer contracts and actual queued MPS checkpoint transfers."""

import io
import pickle
import zipfile
from collections import OrderedDict
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from jetformer.training import _checkpoint_staging as staging
from jetformer.training import checkpoint
from tests.conftest import tiny_config


def _save(payload):
    stream = io.BytesIO()
    torch.save(payload, stream, pickle_module=staging._mps_pickle_module())
    return stream.getvalue()


def _native(payload):
    stream = io.BytesIO()
    torch.save(payload, stream)
    return stream.getvalue()


def test_cpu_checkpoint_uses_native_pickle_with_non_tensor_extra_state(tmp_path, monkeypatch):
    def forbidden():
        raise AssertionError("CPU checkpoint must use native pickle")

    monkeypatch.setattr(checkpoint, "_mps_pickle_module", forbidden)
    model = torch.nn.Linear(2, 2)
    state = model.state_dict()
    state["_extra_state"] = {"metadata": "retained"}
    monkeypatch.setattr(model, "state_dict", lambda: state)
    path = checkpoint.save_checkpoint(
        tmp_path / "checkpoint.pt",
        model=model,
        optimizer=None,
        scheduler=None,
        config=tiny_config(),
        progress={},
        rng_state_by_rank=[],
        class_names=[],
    )
    assert checkpoint.load_checkpoint(path)["model_state_dict"]["_extra_state"] == {"metadata": "retained"}


def test_staging_executes_one_copy_per_storage_and_preserves_native_archive(monkeypatch):
    base = torch.arange(24, dtype=torch.float32).reshape(4, 6)
    complex_base = base.to(torch.complex64)
    payload = OrderedDict(
        full=base,
        view=base[1:, 1::2].T,
        same=base,
        conjugate=complex_base.conj(),
        complex_base=complex_base,
        negative=torch._neg_view(base),
    )
    payload._metadata = {"alias": base, "cycle": payload}
    expected = _native(payload)
    copied, original_copy = [], torch.UntypedStorage.copy_

    def copy(target, source, non_blocking=False):
        copied.append((source._cdata, source.nbytes(), non_blocking))
        return original_copy(target, source, non_blocking=non_blocking)

    # Exercise the identical serializer path on CPU without exposing a test
    # switch in the production helper or claiming MPS transfer coverage here.
    monkeypatch.setattr(staging, "_is_mps_storage", lambda storage: storage.device.type == "cpu")
    monkeypatch.setattr(torch.UntypedStorage, "copy_", copy)
    actual = _save(payload)
    assert actual == expected
    assert copied == [(base.untyped_storage()._cdata, 96, False), (complex_base.untyped_storage()._cdata, 192, False)]
    loaded = torch.load(io.BytesIO(actual), weights_only=True)
    assert loaded["same"] is loaded["full"] is loaded._metadata["alias"]
    assert loaded._metadata["cycle"] is loaded
    assert loaded["view"].stride() == payload["view"].stride()
    assert loaded["view"].storage_offset() == payload["view"].storage_offset()
    assert loaded["negative"].is_neg() and loaded["conjugate"].is_conj()


@pytest.mark.parametrize("location", [None, "", False, 0, "mps"])
def test_native_location_tags_include_falsey_explicit_values(monkeypatch, location):
    canonical = torch.serialization.location_tag(SimpleNamespace(device=torch.device("mps:0")))
    assert canonical == "mps"
    storage = torch.arange(8).untyped_storage()
    storage._fake_device = location
    monkeypatch.setattr(torch.serialization, "location_tag", lambda source: canonical)
    expected = _native({"storage": storage})
    monkeypatch.setattr(staging, "_is_mps_storage", lambda source: source.device.type == "cpu")
    assert _save({"storage": storage}) == expected


class _MutateEarlierTensor:
    def __init__(self, tensor):
        self.tensor = tensor

    def __reduce__(self):
        self.tensor.add_(10)
        return torch.Size, ((),)


class _ResizeEarlierStorage:
    def __init__(self, tensor, elements):
        self.tensor, self.elements = tensor, elements

    def __reduce__(self):
        storage = self.tensor.untyped_storage()
        storage.resize_(self.elements * self.tensor.element_size())
        view = self.tensor.new_empty(0).set_(storage, 0, (self.elements,), (1,))
        view.copy_(torch.arange(self.elements, dtype=self.tensor.dtype))
        return torch.Size, ((),)


def test_staging_observes_mutation_by_later_reducer(monkeypatch):
    monkeypatch.setattr(staging, "_is_mps_storage", lambda source: source.device.type == "cpu")
    archives = []
    for save in (_native, _save):
        value = torch.tensor([1.0, 2.0])
        archives.append(save([value, _MutateEarlierTensor(value)]))
    assert archives[0] == archives[1]
    loaded = torch.load(io.BytesIO(archives[1]), weights_only=True)
    assert torch.equal(loaded[0], torch.tensor([11.0, 12.0]))


def test_native_dtype_conflict_with_distinct_storage_wrappers(monkeypatch):
    array = np.arange(4, dtype=np.float32)
    first, second = torch.from_numpy(array), torch.from_numpy(array.view(np.int32))
    assert first.data_ptr() == second.data_ptr()
    assert first.untyped_storage()._cdata != second.untyped_storage()._cdata
    monkeypatch.setattr(staging, "_is_mps_storage", lambda source: source.device.type == "cpu")
    for save in (_native, _save):
        with pytest.raises(RuntimeError, match="view the same data as different types"):
            save([first, second])


@pytest.mark.parametrize("elements", [4, 12], ids=["shrink", "grow"])
def test_later_storage_resize_retains_native_archive(monkeypatch, elements):
    monkeypatch.setattr(staging, "_is_mps_storage", lambda source: source.device.type == "cpu")
    archives = []
    for save in (_native, _save):
        value = torch.arange(8, dtype=torch.float32)[:2]
        archives.append(save([value, _ResizeEarlierStorage(value, elements)]))
    assert archives[0] == archives[1]
    # Native records resized bytes with earlier pickle metadata. Older readers
    # do not validate this mismatch, so inspect the deliberately invalid archive
    # directly instead of loading it through a version-dependent reader.
    with zipfile.ZipFile(io.BytesIO(archives[1])) as archive:
        assert archive.read("archive/data/0") == np.arange(elements, dtype=np.float32).tobytes()


def test_pickle_failure_never_starts_transfers(monkeypatch):
    monkeypatch.setattr(staging, "_is_mps_storage", lambda source: source.device.type == "cpu")

    def forbidden(*args, **kwargs):
        raise AssertionError("Copies must follow successful pickling")

    monkeypatch.setattr(torch.UntypedStorage, "copy_", forbidden)
    monkeypatch.setattr(torch.mps, "synchronize", forbidden)
    with pytest.raises((pickle.PicklingError, AttributeError)):
        _save([torch.arange(4), lambda: None])


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS required")
def test_mps_checkpoint_selects_staging_and_loads_with_native_reader(tmp_path, monkeypatch):
    model = torch.nn.Linear(2, 2).to("mps")
    copied, original_copy = [], torch.UntypedStorage.copy_

    def copy(target, source, non_blocking=False):
        if source.device.type == "mps" and target.device.type == "cpu":
            copied.append(non_blocking)
        return original_copy(target, source, non_blocking=non_blocking)

    monkeypatch.setattr(torch.UntypedStorage, "copy_", copy)
    path = checkpoint.save_checkpoint(
        tmp_path / "checkpoint.pt",
        model=model,
        optimizer=None,
        scheduler=None,
        config=tiny_config(),
        progress={"global_step": 2},
        rng_state_by_rank=[],
        class_names=[],
    )
    assert copied == [True, True]
    loaded = checkpoint.load_checkpoint(path)
    assert loaded["format_version"] == 6 and loaded["global_step"] == 2
    for name, value in model.state_dict().items():
        assert torch.equal(loaded["model_state_dict"][name], value.cpu())
    assert not list(tmp_path.glob("*.tmp-*"))


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS required")
def test_mps_queued_first_and_later_saves_keep_values_aliases_tags_and_cpu_metadata(monkeypatch):
    expected = torch.arange(24, dtype=torch.float32).reshape(4, 6)
    base = expected.to("mps", non_blocking=False)
    payload = OrderedDict(full=base, view=base[1:, 1::2].T, same=base, cpu_metadata=torch.arange(5))
    payload._metadata = {"alias": base}
    original_copy, original_wait = torch.UntypedStorage.copy_, torch.mps.synchronize
    copies, waits, archives, expected_values = [], [], [], []

    def copy(target, source, non_blocking=False):
        if source.device.type == "mps" and target.device.type == "cpu":
            copies.append(non_blocking)
        return original_copy(target, source, non_blocking=non_blocking)

    def wait():
        waits.append(True)
        original_wait()

    monkeypatch.setattr(torch.UntypedStorage, "copy_", copy)
    monkeypatch.setattr(torch.mps, "synchronize", wait)
    # No device reads or added synchronization between saves: the helper must
    # finish its own transfers before file writing and the next in-place update.
    for _ in range(3):
        base.add_(1)
        expected.add_(1)
        pressure = torch.empty_like(base).fill_(91)
        del pressure
        archives.append(_save(payload))
        expected_values.append(expected.clone())
    assert copies == [True] * 3 and waits == [True] * 3
    for archive, values in zip(archives, expected_values, strict=True):
        tags = []

        def cpu(storage, location, tags=tags):
            tags.append(location)
            return storage

        loaded = torch.load(io.BytesIO(archive), map_location=cpu, weights_only=True)
        assert tags == ["mps", "cpu"]
        assert torch.equal(loaded["full"], values)
        assert torch.equal(loaded["view"], values[1:, 1::2].T)
        assert torch.equal(loaded["cpu_metadata"], torch.arange(5))
        assert loaded["full"] is loaded["same"] is loaded._metadata["alias"]
    assert archives[-1] == _native(payload)


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS required")
def test_mps_partial_copy_failure_waits_for_the_first_transfer(monkeypatch):
    first = torch.arange(16, dtype=torch.float32).to("mps", non_blocking=False)
    second = torch.arange(5).to("mps", non_blocking=False)
    original_copy, original_wait = torch.UntypedStorage.copy_, torch.mps.synchronize
    events = []

    def copy(target, source, non_blocking=False):
        events.append(("copy", non_blocking))
        if len(events) == 2:
            raise RuntimeError("Intentional second-copy failure")
        return original_copy(target, source, non_blocking=non_blocking)

    def wait():
        events.append(("wait", None))
        original_wait()

    monkeypatch.setattr(torch.UntypedStorage, "copy_", copy)
    monkeypatch.setattr(torch.mps, "synchronize", wait)
    with pytest.raises(RuntimeError, match="Intentional second-copy failure"):
        _save([first, second])
    assert events == [("copy", True), ("copy", True), ("wait", None)]
    assert torch.equal(first.cpu(), torch.arange(16, dtype=torch.float32))
