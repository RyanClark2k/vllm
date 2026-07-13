# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Capture-time storage manifest for weight-reload identity checking.

CUDA graphs bake device addresses at capture time. Weight reload preserves
storage for registered parameters and buffers; any other tensor a captured
graph references (plain attributes, tensors on kernel objects, tensors held
inside callables) can be silently rebound, leaving live graphs pointing at
freed memory (#48312, #48438).

``ReloadStorageManifest`` records, at the moment graphs are (or would be)
captured, a weak reference to every storage that could be baked into a graph,
then verifies after a reload that every recorded storage is still alive and
every named slot still resolves to the storage it had at capture. Storage
lifetime semantics are device-independent, so the mechanism is testable on
CPU even though the consumer (graph replay) is CUDA-only.

Two recording modes, covering each other's blind spots:

- ``record_from_walk``: traverse holders (modules, kernel objects) through
  parameters, buffers, plain attributes, containers, ``functools.partial``
  bindings, and closure cells.
- ``recording()``: a ``TorchDispatchMode`` that records every tensor argument
  flowing through every op while active (e.g. during graph-capture forward
  passes). Sees tensors reachable through no attribute path; cannot see inside
  regions that bypass dispatch (e.g. fully compiled artifacts).
"""

import functools
from dataclasses import dataclass, field

import torch
from torch.multiprocessing.reductions import StorageWeakRef
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_flatten


@dataclass
class ManifestReport:
    """Result of checking a manifest after a reload."""

    expired: list[str] = field(default_factory=list)
    moved: list[str] = field(default_factory=list)
    checked: int = 0

    @property
    def ok(self) -> bool:
        return not self.expired and not self.moved


class ReloadStorageManifest:
    """Records storages at capture time; verifies them after reload."""

    def __init__(self) -> None:
        # slot path -> (weak storage ref, data_ptr at record time)
        self._slots: dict[str, tuple[StorageWeakRef, int]] = {}

    def __len__(self) -> int:
        return len(self._slots)

    def _add(self, path: str, tensor: torch.Tensor) -> None:
        storage = tensor.untyped_storage()
        self._slots.setdefault(path, (StorageWeakRef(storage), tensor.data_ptr()))

    def _collect(self, value, path: str, depth: int) -> None:
        if depth < 0:
            return
        if isinstance(value, torch.Tensor):
            self._add(path, value)
        elif isinstance(value, functools.partial):
            for i, arg in enumerate(value.args):
                self._collect(arg, f"{path}.args[{i}]", depth - 1)
            for key, arg in (value.keywords or {}).items():
                self._collect(arg, f"{path}.kw[{key}]", depth - 1)
            self._collect(value.func, f"{path}.func", depth - 1)
        elif callable(value) and getattr(value, "__closure__", None):
            for i, cell in enumerate(value.__closure__):
                self._collect(cell.cell_contents, f"{path}.closure[{i}]", depth - 1)
        elif isinstance(value, (list, tuple)):
            for i, item in enumerate(value):
                self._collect(item, f"{path}[{i}]", depth - 1)
        elif isinstance(value, dict):
            for key, item in value.items():
                self._collect(item, f"{path}[{key!r}]", depth - 1)

    def record_from_walk(self, *holders, depth: int = 3) -> None:
        """Record every tensor reachable from each holder."""
        for holder in holders:
            tag = type(holder).__name__
            sources: dict[str, object] = {}
            if isinstance(holder, torch.nn.Module):
                for name, module in holder.named_modules():
                    prefix = f"{tag}.{name}" if name else tag
                    sources.update(
                        {f"{prefix}.{k}": v for k, v in module._parameters.items()}
                    )
                    sources.update(
                        {f"{prefix}.{k}": v for k, v in module._buffers.items()}
                    )
                    sources.update(
                        {f"{prefix}.{k}": v for k, v in vars(module).items()}
                    )
            else:
                sources.update({f"{tag}.{k}": v for k, v in vars(holder).items()})
            for name, value in sources.items():
                self._collect(value, name, depth)

    def recording(self) -> "_DispatchRecorder":
        """Context manager recording every tensor argument passed to any op
        while active. Use around the forward passes that graphs capture."""
        return _DispatchRecorder(self)

    def check(self, *holders, depth: int = 3) -> ManifestReport:
        """Verify the manifest after a reload.

        A recorded storage that has been freed means captured graphs hold a
        dangling address: always a violation. If holders are provided, slots
        are re-resolved and a slot whose tensor no longer starts at its
        recorded address is reported as moved (rebinding where the old
        storage is kept alive by another reference).
        """
        report = ManifestReport(checked=len(self._slots))
        for path, (ref, _) in self._slots.items():
            if ref.expired():
                report.expired.append(path)

        if holders:
            current = ReloadStorageManifest()
            current.record_from_walk(*holders, depth=depth)
            for path, (_, ptr) in self._slots.items():
                if path in current._slots and current._slots[path][1] != ptr:
                    report.moved.append(path)
        return report


class _DispatchRecorder(TorchDispatchMode):
    def __init__(self, manifest: ReloadStorageManifest) -> None:
        super().__init__()
        self._manifest = manifest
        self._seen = 0

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        flat, _ = tree_flatten((args, kwargs or {}))
        for value in flat:
            if isinstance(value, torch.Tensor):
                self._manifest._add(f"op:{func}[arg{self._seen}]", value)
                self._seen += 1
        return func(*args, **(kwargs or {}))
