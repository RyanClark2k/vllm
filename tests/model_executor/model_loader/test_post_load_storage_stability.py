# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Registry-parametrized storage-stability check for post-load runtime tensors.

Weight reload (RL weight sync) reruns ``process_weights_after_loading`` and
copy-backs registered parameters/buffers into their original storage so that
addresses captured by CUDA graphs stay valid. Any OTHER tensor a kernel
creates during post-load — a plain attribute, a tensor on the kernel object,
or a tensor captured inside a callable — escapes that protection: reload
rebinds it and captured graphs keep the freed pointer (see #48312, #48438).

This test enumerates the mixed-precision kernel registry, runs post-load
twice on a minimal checkpoint-format layer with the production copy-back
semantics in between, and asserts that every tensor reachable from the layer
and the kernel object keeps its storage. New kernels are covered by default:
on GPU they run against real ops; on CPU they run if their post-load path is
pure torch (e.g. CPUWNA16), or via a small stub adapter for device-only ops
(Marlin and Machete below), and skip otherwise.

The pointer walk unwraps ``functools.partial`` and closure cells: tensors
smuggled into callables (e.g. Machete's ``act_perm``) are baked into CUDA
graphs just like attribute tensors, but an attribute-only walk cannot see
them.
"""

import functools

import pytest
import torch

from vllm.model_executor.kernels.linear import _POSSIBLE_KERNELS
from vllm.model_executor.kernels.linear.mixed_precision.MPLinearKernel import (
    MPLinearKernel,
    MPLinearLayerConfig,
)
from vllm.model_executor.model_loader.reload.utils import get_layer_params_buffers
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.parameter import (
    GroupQuantScaleParameter,
    PackedvLLMParameter,
    RowvLLMParameter,
)
from vllm.scalar_type import scalar_types

SIZE_K, SIZE_N, GROUP_SIZE = 128, 64, 64


def _collect_tensors(value, path: str, out: dict[str, int], depth: int) -> None:
    if depth < 0:
        return
    if isinstance(value, torch.Tensor):
        out[path] = value.data_ptr()
    elif isinstance(value, functools.partial):
        for i, arg in enumerate(value.args):
            _collect_tensors(arg, f"{path}.args[{i}]", out, depth - 1)
        for key, arg in (value.keywords or {}).items():
            _collect_tensors(arg, f"{path}.kw[{key}]", out, depth - 1)
        _collect_tensors(value.func, f"{path}.func", out, depth - 1)
    elif callable(value) and getattr(value, "__closure__", None):
        for i, cell in enumerate(value.__closure__):
            _collect_tensors(cell.cell_contents, f"{path}.closure[{i}]", out, depth - 1)
    elif isinstance(value, (list, tuple)):
        for i, item in enumerate(value):
            _collect_tensors(item, f"{path}[{i}]", out, depth - 1)
    elif isinstance(value, dict):
        for key, item in value.items():
            _collect_tensors(item, f"{path}[{key!r}]", out, depth - 1)


def tensor_slots(*holders) -> dict[str, int]:
    """Map "holder.attr[...]" -> data_ptr for every tensor reachable from each
    holder: params, buffers, plain attributes, and tensors inside callables."""
    slots: dict[str, int] = {}
    for holder in holders:
        tag = type(holder).__name__
        sources: dict[str, object] = {}
        if isinstance(holder, torch.nn.Module):
            sources.update(holder._parameters)
            sources.update(holder._buffers)
        sources.update(vars(holder))
        for name, value in sources.items():
            _collect_tensors(value, f"{tag}.{name}", slots, depth=3)
    return slots


def make_config(has_g_idx: bool) -> MPLinearLayerConfig:
    return MPLinearLayerConfig(
        full_weight_shape=(SIZE_K, SIZE_N),
        partition_weight_shape=(SIZE_K, SIZE_N),
        weight_type=scalar_types.uint4b8,
        act_type=torch.float16,
        group_size=GROUP_SIZE,
        zero_points=False,
        has_g_idx=has_g_idx,
    )


def load_checkpoint_format_weights(layer: torch.nn.Module, has_g_idx: bool, seed: int):
    gen = torch.Generator().manual_seed(seed)
    layer.qweight = PackedvLLMParameter(
        data=torch.randint(
            -(2**31), 2**31 - 1, (SIZE_K // 8, SIZE_N), dtype=torch.int32, generator=gen
        ),
        input_dim=0,
        output_dim=1,
        packed_dim=0,
        packed_factor=8,
        weight_loader=default_weight_loader,
    )
    layer.scales = GroupQuantScaleParameter(
        data=torch.ones(SIZE_K // GROUP_SIZE, SIZE_N, dtype=torch.float16),
        input_dim=0,
        output_dim=1,
        weight_loader=default_weight_loader,
    )
    if has_g_idx:
        layer.g_idx = RowvLLMParameter(
            data=torch.randint(
                0, SIZE_K // GROUP_SIZE, (SIZE_K,), dtype=torch.int32, generator=gen
            ),
            input_dim=0,
            weight_loader=default_weight_loader,
        )


def _marlin_stubs():
    from vllm import _custom_ops as ops
    from vllm.model_executor.layers.quantization.utils import marlin_utils

    return [
        (marlin_utils, "num_compute_units", lambda _: 4),
        (
            ops,
            "gptq_marlin_repack",
            lambda b_q_weight, perm, size_k, size_n, num_bits, is_a_8bit=False: (
                torch.zeros(size_k // 16, size_n * 2, dtype=torch.int32)
            ),
        ),
    ]


def _machete_stubs():
    from vllm import _custom_ops as ops

    return [
        (
            ops,
            "machete_prepack_B",
            lambda x, a_type, b_type, group_scales_type: torch.zeros_like(x),
        ),
    ]


# CPU replacements for device-only post-load ops; kernels without an adapter
# run as-is (and skip on CPU if they hit a device-only op).
STUB_ADAPTERS = {
    "MarlinLinearKernel": _marlin_stubs,
    "MacheteLinearKernel": _machete_stubs,
}


def all_registry_kernels() -> list[type[MPLinearKernel]]:
    seen: dict[str, type[MPLinearKernel]] = {}
    for kernels in _POSSIBLE_KERNELS.values():
        for kernel_cls in kernels:
            seen.setdefault(kernel_cls.__name__, kernel_cls)
    return sorted(seen.values(), key=lambda k: k.__name__)


@pytest.mark.parametrize("has_g_idx", [False, True], ids=["plain", "act_order"])
@pytest.mark.parametrize("kernel_cls", all_registry_kernels(), ids=lambda k: k.__name__)
def test_post_load_runtime_tensors_stable(
    kernel_cls, has_g_idx, monkeypatch, dist_init
):
    config = make_config(has_g_idx)
    adapter = STUB_ADAPTERS.get(kernel_cls.__name__)

    if adapter is not None:
        for module, attr, replacement in adapter():
            monkeypatch.setattr(module, attr, replacement)
        kernel = object.__new__(kernel_cls)  # bypass device-capability gate
        kernel.config = config
        kernel.w_q_name = "qweight"
        kernel.w_s_name = "scales"
        kernel.w_zp_name = None
        kernel.w_gidx_name = "g_idx" if has_g_idx else None
    else:
        ok, reason = kernel_cls.can_implement(config)
        if not ok:
            pytest.skip(f"can_implement: {reason}")
        kernel = kernel_cls(
            config, "qweight", "scales", None, "g_idx" if has_g_idx else None
        )

    layer = torch.nn.Module()

    def process(seed: int):
        load_checkpoint_format_weights(layer, has_g_idx, seed)
        try:
            kernel.process_weights_after_loading(layer)
        except (RuntimeError, NotImplementedError, AssertionError, OSError) as e:
            if adapter is not None:
                raise
            pytest.skip(f"post-load needs device ops on this platform: {e}")

    process(seed=0)
    params0, buffers0 = get_layer_params_buffers(layer)
    before = tensor_slots(layer, kernel)

    process(seed=1)  # reload: fresh checkpoint-format weights, reprocess

    # Mirror the production copy-back (_copy_and_restore_kernel_tensors,
    # reload/layerwise.py): registered params/buffers get their values copied
    # into the ORIGINAL storage and the original objects re-registered. The
    # bug class under test is every tensor this protection cannot see.
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

    after = tensor_slots(layer, kernel)
    moved = {
        name: (hex(ptr), hex(after[name]))
        for name, ptr in before.items()
        if name in after and after[name] != ptr
    }
    vanished = sorted(set(before) - set(after))
    assert not moved and not vanished, (
        f"runtime tensors escape reload copy-back protection: "
        f"moved={moved} vanished={vanished}"
    )
