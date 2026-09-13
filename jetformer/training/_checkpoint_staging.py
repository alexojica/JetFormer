"""Batch MPS checkpoint transfers while retaining native PyTorch serialization.

PyTorch's ZIP writer derives a Pickler and calls dump once before writing storage
bytes. Its native reducers, TypedStorage and _fake_device protocol remain in use;
serializer-contract tests must accompany PyTorch upgrades. This is scoped to the
filename-based ZIP saves in checkpoint.py, not a general pickle replacement.
"""

from __future__ import annotations

import pickle
from types import SimpleNamespace

import torch


def _ignore_persistent_id(_value):
    return None


def _is_mps_storage(storage):
    return storage.device.type == "mps"


def _mps_pickle_module():
    """Queue D2H copies, then wait once before the synchronous checkpoint write.

    A complete optimizer checkpoint measured 427 -> 121 ms on MPS. CPU staging
    equals all MPS storage bytes and is held through the native archive write.
    """

    class StagingPickler:
        def __init__(self, *args, **kwargs):
            self._pickler = pickle.Pickler(*args, **kwargs)
            # torch.save supplies a derived Pickler with its native persistent_id.
            # Retain that bound method for the plain Pickler's callback below.
            native_persistent_id = self.persistent_id
            self._staged = {}
            self._source_dtypes = {}
            self._pending_mps = False

            def persistent_id(value):
                typed = isinstance(value, torch.storage.TypedStorage)
                if typed or torch.is_storage(value):
                    source = value._untyped_storage if typed else value
                    # Native validation concerns original aliases, including
                    # distinct storage identities that share one data pointer.
                    if source.device.type != "meta" and (pointer := source.data_ptr()):
                        dtype = value.dtype if typed else torch.uint8
                        previous = self._source_dtypes.setdefault(pointer, dtype)
                        if previous != dtype:
                            raise RuntimeError(
                                "Cannot save multiple tensors or storages that view the same data as different types"
                            )
                    if _is_mps_storage(source):
                        identity = source._cdata
                        if identity not in self._staged:
                            target = torch.UntypedStorage(source.nbytes(), device="cpu")
                            # Keep targets in native storage records, but defer
                            # their copy until every reducer has finished. A
                            # later reducer may mutate an earlier source tensor.
                            self._staged[identity] = (source, target)
                        target = self._staged[identity][1]
                        replacement = (
                            torch.storage.TypedStorage(wrap_storage=target, dtype=value.dtype, _internal=True)
                            if typed
                            else target
                        )
                        location = getattr(value, "_fake_device", None)
                        replacement._fake_device = (
                            location if location is not None else torch.serialization.location_tag(source)
                        )
                        return native_persistent_id(replacement)
                return native_persistent_id(value)

            self._pickler.persistent_id = persistent_id

        def dump(self, obj):
            try:
                result = self._pickler.dump(obj)
                for source, target in self._staged.values():
                    # Reducers may also resize a previously recorded storage.
                    # Native writing observes its final byte size after pickling.
                    if target.nbytes() != source.nbytes():
                        target.resize_(source.nbytes())
                    # Hold both endpoints before enqueueing; finally waits even
                    # if a later copy fails after earlier transfers were queued.
                    self._pending_mps |= source.device.type == "mps"
                    target.copy_(source, non_blocking=source.device.type == "mps")
                return result
            finally:
                try:
                    if self._pending_mps:
                        torch.mps.synchronize()
                finally:
                    # Native serialization owns the CPU storages after dump.
                    # Release endpoints and the delegate callback's closure
                    # cycle after this one-save Pickler is done.
                    self._staged.clear()
                    self._source_dtypes.clear()
                    self._pickler.persistent_id = _ignore_persistent_id

    return SimpleNamespace(__name__="pickle", Pickler=StagingPickler)
