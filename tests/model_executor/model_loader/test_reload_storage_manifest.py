# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU red/green tests for ReloadStorageManifest.

Storage lifetime (weakref expiry, data_ptr stability) is device-independent,
so the manifest mechanism is provable on CPU; only its consumer (CUDA graph
replay) needs a GPU. The red tests restore the historical rebinding behavior
that #48438 fixed and assert the manifest reports it at the reload boundary.
"""

import torch

from tests.model_executor.model_loader.test_post_load_storage_stability import (
    _machete_stubs,
    _marlin_stubs,
    load_checkpoint_format_weights,
    make_config,
)
from vllm.model_executor.kernels.linear.mixed_precision.machete import (
    MacheteLinearKernel,
)
from vllm.model_executor.kernels.linear.mixed_precision.marlin import (
    MarlinLinearKernel,
)
from vllm.model_executor.model_loader.reload.storage_manifest import (
    ReloadStorageManifest,
)
from vllm.model_executor.model_loader.reload.utils import get_layer_params_buffers


def _copy_back(layer, params0, buffers0):
    """Production copy-back semantics (_copy_and_restore_kernel_tensors)."""
    for name, param in params0.items():
        new = getattr(layer, name)
        if new is not param:
            param.data.copy_(new.data)
            delattr(layer, name)
            layer.register_parameter(name, param)
    for name, buffer in buffers0.items():
        new = getattr(layer, name)
        if new is not buffer:
            buffer.data.copy_(new.data)
            delattr(layer, name)
            layer.register_buffer(name, buffer)


def _make_marlin(monkeypatch):
    for module, attr, replacement in _marlin_stubs():
        monkeypatch.setattr(module, attr, replacement)
    kernel = object.__new__(MarlinLinearKernel)
    kernel.config = make_config(has_g_idx=True)
    kernel.w_q_name = "qweight"
    kernel.w_s_name = "scales"
    kernel.w_zp_name = None
    kernel.w_gidx_name = "g_idx"
    layer = torch.nn.Module()
    return kernel, layer


def _reload_cycle(kernel, layer, manifest=None):
    """load -> process (capture point on first call) -> reload -> copy-back."""
    load_checkpoint_format_weights(layer, has_g_idx=True, seed=0)
    kernel.process_weights_after_loading(layer)
    if manifest is not None:
        manifest.record_from_walk(layer, kernel)
    params0, buffers0 = get_layer_params_buffers(layer)
    load_checkpoint_format_weights(layer, has_g_idx=True, seed=1)
    kernel.process_weights_after_loading(layer)
    _copy_back(layer, params0, buffers0)


def test_manifest_green_on_fixed_marlin(monkeypatch, dist_init):
    kernel, layer = _make_marlin(monkeypatch)
    manifest = ReloadStorageManifest()
    _reload_cycle(kernel, layer, manifest)
    assert len(manifest) > 0
    report = manifest.check(layer, kernel)
    assert report.ok, f"expired={report.expired} moved={report.moved}"


def test_manifest_red_on_historical_marlin_behavior(monkeypatch, dist_init):
    """Restore the pre-#48438 behavior on the real kernel: fresh workspace
    every call, sort indices assigned as a plain attribute. The manifest must
    report both at the reload boundary."""
    from vllm.model_executor.kernels.linear.mixed_precision import marlin as marlin_mod

    orig_make_workspace = marlin_mod.marlin_make_workspace_new
    monkeypatch.setattr(
        marlin_mod,
        "marlin_make_workspace_new",
        lambda device, max_blocks_per_sm=1, existing=None: orig_make_workspace(
            device, max_blocks_per_sm
        ),
    )
    monkeypatch.setattr(
        marlin_mod,
        "replace_parameter",
        lambda layer, name, tensor, prefer_copy=False: object.__setattr__(
            layer, name, tensor
        ),
    )

    kernel, layer = _make_marlin(monkeypatch)
    manifest = ReloadStorageManifest()
    _reload_cycle(kernel, layer, manifest)
    report = manifest.check(layer, kernel)

    assert not report.ok
    assert any("workspace" in path for path in report.expired + report.moved)
    assert any("g_idx_sort_indices" in path for path in report.expired + report.moved)


def test_manifest_reports_unfixed_machete_act_perm(monkeypatch, dist_init):
    """Machete's act-order path captures a fresh argsort permutation in
    ``self.act_perm`` (a ``functools.partial``) on every post-load call;
    unfixed as of this prototype, reported on RFC #48312. The manifest must
    report it. When Machete is fixed, this test starts failing and should be
    inverted to assert ``report.ok``."""
    for module, attr, replacement in _machete_stubs():
        monkeypatch.setattr(module, attr, replacement)
    kernel = object.__new__(MacheteLinearKernel)
    kernel.config = make_config(has_g_idx=True)
    kernel.w_q_name = "qweight"
    kernel.w_s_name = "scales"
    kernel.w_zp_name = None
    kernel.w_gidx_name = "g_idx"
    layer = torch.nn.Module()

    manifest = ReloadStorageManifest()
    _reload_cycle(kernel, layer, manifest)
    report = manifest.check(layer, kernel)

    assert not report.ok
    assert any("act_perm" in path for path in report.expired + report.moved)


def test_manifest_red_on_toy_rebinder(dist_init):
    """Checker sanity: a holder that rebinds a plain tensor attribute on every
    processing call is reported as expired once the old storage is dropped."""

    class ToyKernel:
        def process(self, layer):
            self.workspace = torch.zeros(64)
            layer.scratch = torch.arange(16)

    kernel = ToyKernel()
    layer = torch.nn.Module()
    kernel.process(layer)

    manifest = ReloadStorageManifest()
    manifest.record_from_walk(layer, kernel)
    kernel.process(layer)

    report = manifest.check(layer, kernel)
    assert set(report.expired) == {"ToyKernel.workspace", "Module.scratch"}
    assert set(report.moved) == {"ToyKernel.workspace", "Module.scratch"}


def test_dispatch_recorder_sees_tensors_no_walk_can_find(dist_init):
    """A tensor referenced only from a module-level registry is invisible to
    any attribute walk of the model, but its address is baked into captured
    graphs all the same. The dispatch recorder catches it."""
    registry = {"scale": torch.randn(4)}

    class Toy(torch.nn.Module):
        def forward(self, x):
            return x * registry["scale"]

    model = Toy()
    x = torch.ones(4)

    walk_only = ReloadStorageManifest()
    walk_only.record_from_walk(model)
    assert not any("scale" in p or "op:" in p for p in walk_only._slots)

    manifest = ReloadStorageManifest()
    with manifest.recording():
        model(x)
    assert len(manifest) > 0

    registry["scale"] = torch.randn(4)  # "reload" rebinds it; old storage dies
    report = manifest.check()
    assert report.expired, "recorder failed to catch the rebound registry tensor"
