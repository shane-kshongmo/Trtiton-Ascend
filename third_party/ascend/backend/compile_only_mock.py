"""Compile-only mock layer for Triton-Ascend.

When TRITON_COMPILE_ONLY=1 is set, this module patches torch and torch_npu so
that Triton kernel scripts can run unmodified without NPU hardware.  Device
operations (tensor allocation on 'npu', stream queries, etc.) are transparently
redirected to CPU, while the Triton MLIR compilation pipeline runs normally and
dumps IR artifacts.

Activation
----------
This module is auto-imported by the Ascend backend driver when
``TRITON_COMPILE_ONLY`` is set.  No user code changes are required.
"""

import functools
import importlib
import os
import sys

_installed = False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _is_compile_only():
    return os.environ.get("TRITON_COMPILE_ONLY", "").lower() in ("1", "true")


def _npu_to_cpu(device):
    """Convert an 'npu'-flavoured device spec to 'cpu'."""
    if device is None:
        return device
    s = str(device)
    if "npu" in s:
        return "cpu"
    return device


# ---------------------------------------------------------------------------
# Layer 1 — Torch factory-function patching
# ---------------------------------------------------------------------------

_FACTORY_NAMES = [
    "randn", "rand", "empty", "zeros", "ones", "full",
    "arange", "linspace", "logspace", "tensor", "as_tensor",
]

_LIKE_FACTORY_NAMES = [
    "empty_like", "zeros_like", "ones_like",
    "rand_like", "randn_like", "full_like",
]

_orig_factories = {}
_orig_like_factories = {}
_orig_tensor_to = None
_orig_tensor_npu = None


def _make_factory_wrapper(orig):
    @functools.wraps(orig)
    def wrapper(*args, **kwargs):
        if "device" in kwargs:
            kwargs["device"] = _npu_to_cpu(kwargs["device"])
        return orig(*args, **kwargs)
    return wrapper


def _make_like_factory_wrapper(orig):
    @functools.wraps(orig)
    def wrapper(*args, **kwargs):
        if "device" in kwargs:
            kwargs["device"] = _npu_to_cpu(kwargs["device"])
        return orig(*args, **kwargs)
    return wrapper


def _patch_torch_factories():
    import torch

    for name in _FACTORY_NAMES:
        orig = getattr(torch, name, None)
        if orig is not None and name not in _orig_factories:
            _orig_factories[name] = orig
            setattr(torch, name, _make_factory_wrapper(orig))

    for name in _LIKE_FACTORY_NAMES:
        orig = getattr(torch, name, None)
        if orig is not None and name not in _orig_like_factories:
            _orig_like_factories[name] = orig
            setattr(torch, name, _make_like_factory_wrapper(orig))


# ---------------------------------------------------------------------------
# Layer 1 — Tensor.to() / Tensor.npu() patching
# ---------------------------------------------------------------------------

def _patched_tensor_to(self, *args, **kwargs):
    # Positional: .to(device), .to(dtype), .to(device, dtype), .to(tensor)
    args = list(args)
    for i, a in enumerate(args):
        if isinstance(a, str) and "npu" in a:
            args[i] = "cpu"
        elif hasattr(a, "type") and "npu" in str(a):
            # torch.device object
            import torch
            args[i] = torch.device("cpu")
    if "device" in kwargs:
        kwargs["device"] = _npu_to_cpu(kwargs["device"])
    return _orig_tensor_to(self, *args, **kwargs)


def _patched_tensor_npu(self, *args, **kwargs):
    # .npu() should be a no-op in compile-only mode (tensor stays on CPU)
    return self


def _patch_tensor_methods():
    import torch
    global _orig_tensor_to, _orig_tensor_npu

    if _orig_tensor_to is None:
        _orig_tensor_to = torch.Tensor.to
        torch.Tensor.to = _patched_tensor_to

    if _orig_tensor_npu is None and hasattr(torch.Tensor, "npu"):
        _orig_tensor_npu = torch.Tensor.npu
        torch.Tensor.npu = _patched_tensor_npu


# ---------------------------------------------------------------------------
# Layer 1 — torch.npu namespace patching
# ---------------------------------------------------------------------------

def _patch_torch_npu_namespace():
    """Patch ``torch.npu.*`` methods that query hardware."""
    import torch

    if not hasattr(torch, "npu"):
        return

    npu = torch.npu

    # Only patch if not already patched
    if getattr(npu, "_compile_only_patched", False):
        return
    npu._compile_only_patched = True

    _orig_current_device = getattr(npu, "current_device", None)
    _orig_device_count = getattr(npu, "device_count", None)

    if _orig_current_device is not None:
        npu.current_device = lambda: 0
    if _orig_device_count is not None:
        npu.device_count = lambda: 1

    if hasattr(npu, "set_device"):
        npu.set_device = lambda *a, **kw: None
    if hasattr(npu, "synchronize"):
        npu.synchronize = lambda *a, **kw: None
    if hasattr(npu, "is_available"):
        npu.is_available = lambda: True
    if hasattr(npu, "mem_get_info"):
        # Return 8GB free / 16GB total as dummy
        npu.mem_get_info = lambda *a, **kw: (8 * 1024**3, 16 * 1024**3)
    if hasattr(npu, "empty_cache"):
        npu.empty_cache = lambda: None


# ---------------------------------------------------------------------------
# Import hook — patch torch.npu after torch_npu is loaded
# ---------------------------------------------------------------------------

class _TorchNpuPostImportHook:
    """``sys.meta_path`` finder that patches ``torch.npu`` after ``torch_npu``
    is fully imported."""

    _active = True

    def find_module(self, fullname, path=None):
        if fullname == "torch_npu" and self._active:
            return self
        return None

    def load_module(self, fullname):
        # Let the real importer handle torch_npu
        self._active = False  # prevent recursion
        try:
            if fullname in sys.modules:
                mod = sys.modules[fullname]
            else:
                mod = importlib.import_module(fullname)
        finally:
            self._active = True

        # Now torch.npu should exist — patch it
        _patch_torch_npu_namespace()
        _patch_tensor_methods()  # Tensor.npu() may be added by torch_npu
        return mod


def _install_torch_npu_import_hook():
    """Register the post-import hook for torch_npu."""
    # If torch_npu is already imported, patch immediately
    if "torch_npu" in sys.modules:
        _patch_torch_npu_namespace()
        return

    # Otherwise, install the hook
    for hook in sys.meta_path:
        if isinstance(hook, _TorchNpuPostImportHook):
            return  # already installed
    sys.meta_path.insert(0, _TorchNpuPostImportHook())


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def install():
    """Apply all compile-only patches.  Idempotent."""
    global _installed
    if _installed:
        return
    if not _is_compile_only():
        return
    _installed = True

    _patch_torch_factories()
    _patch_tensor_methods()
    _install_torch_npu_import_hook()

    # If torch_npu already loaded, patch namespace now
    if "torch_npu" in sys.modules:
        _patch_torch_npu_namespace()

    print("[compile-only] NPU device mocking activated — tensors redirected to CPU")
