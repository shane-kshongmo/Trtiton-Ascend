# Build and Dump IR for Ascend Compiler Research

This guide shows how to build Triton-Ascend in WSL and dump IR produced by
each Ascend compiler pass -- no NPU hardware required.

## 1. Build In WSL

Use WSL on Windows and open the repo from the Linux side. In our setup the repo is available at:

```bash
/mnt/d/work/git/triton-ascend
```

Install the build dependencies first:

```bash
sudo apt-get update
sudo apt-get install -y \
  build-essential \
  cmake \
  ninja-build \
  clang \
  lld \
  git \
  pkg-config \
  zlib1g-dev \
  python3-pip \
  python3-venv \
  python3.11 \
  python3.11-dev \
  python3.11-venv \
  libxml2-dev \
  libzstd-dev
```

Before creating the virtual environment, confirm that you are using a
supported interpreter. The repository installation guide currently targets
Python 3.9 to 3.11. In WSL, use Python 3.11 unless you have a project-specific
reason to pin a different supported version.

Run this quick preflight first:

```bash
python3.11 --version
which python3.11
which cmake
which ninja
which clang
which lld
```

Create and prepare a Python virtual environment:

```bash
python3.11 -m venv ~/venvs/triton-ascend
source ~/venvs/triton-ascend/bin/activate
python --version
pip install -U pip setuptools wheel pybind11 lit ninja
```

Build and install Triton-Ascend:

```bash
cd /mnt/d/work/git/triton-ascend/python

export TRITON_BUILD_WITH_CCACHE=true
export TRITON_BUILD_WITH_CLANG_LLD=true
export TRITON_BUILD_PROTON=OFF
export TRITON_WHEEL_NAME=triton-ascend
export TRITON_APPEND_CMAKE_ARGS="-DTRITON_BUILD_UT=OFF"

python3.11 setup.py install
```

If the workspace cannot write to the configured `ccache` location, disable it:

```bash
export TRITON_BUILD_WITH_CCACHE=false
export CCACHE_DISABLE=1
python3.11 setup.py install
```

If the import works, the compiler build is ready:

```bash
python3.11 - <<'PY'
import triton
print(triton.__file__)
PY
```

The `python setup.py install` step is important. A raw checkout is not enough
for a fresh environment because Triton's backend discovery walks the installed
`python/triton/backends/*` directories at import time.

If you need to rebuild the local `bishengir-compile` helper, apply the root-repo patch in
[`third_party/ascend_patches/0001-Fix-LLVM-20-compatibility-for-bishengir-compile.patch`](../third_party/ascend_patches/0001-Fix-LLVM-20-compatibility-for-bishengir-compile.patch)
to the `third_party/ascend/AscendNPU-IR` submodule before running its standalone build.

## 1.1 Full-Pipeline Prerequisites Checklist

Before dumping IR, make sure the active environment satisfies all of the
following:

- `import triton` works
- `import torch` and `import torch_npu` work
- Triton-Ascend was installed with `python setup.py install` or from a built wheel
- `bishengir-compile` is available on `PATH`, or `TRITON_NPU_COMPILER_PATH` is set
- `ASCEND_HOME_PATH` is set to a valid Ascend toolkit installation

Quick check:

```bash
python --version
python - <<'PY'
import triton
print("triton:", triton.__file__)
for mod in ("torch", "torch_npu"):
    try:
        m = __import__(mod)
        print(f"{mod}:", m.__file__)
    except Exception as exc:
        print(f"{mod}: missing ({exc})")
PY

which bishengir-compile
echo "$ASCEND_HOME_PATH"
```

If `ASCEND_HOME_PATH` is empty, load the Ascend toolkit first:

```bash
source /path/to/Ascend/cann-*/set_env.sh
echo "$ASCEND_HOME_PATH"
```

## 2. Dump IR with Compile-Only Mode

When `TRITON_COMPILE_ONLY=1` is set, the Ascend backend automatically mocks
all NPU device operations (tensor allocation, streams, device queries) so that
**unmodified** Triton kernel scripts run the full MLIR compilation pipeline
without hardware. Kernel launch is skipped; IR dumps are produced normally.

### 2.1 Set Environment Variables

```bash
export TRITON_ASCEND_ARCH=Ascend910_9599
export TRITON_ALWAYS_COMPILE=1
export TRITON_COMPILE_ONLY=1
export TRITON_KERNEL_DUMP=1
export TRITON_DEBUG=1
export TRITON_DUMP_DIR=/path/to/dump/dir
```

### 2.2 Run Any Tutorial

No code changes are needed. Just run an existing tutorial script:

```bash
cd /tmp   # avoid source-tree shadowing
python /path/to/triton-ascend/third_party/ascend/tutorials/01-vector-add.py
python /path/to/triton-ascend/third_party/ascend/tutorials/03-layer-norm.py
```

The mock layer prints `[compile-only] NPU device mocking activated` to confirm
it is active. The `torch.allclose` validation at the end of tutorials will fail
because the kernel didn't execute -- this is expected and harmless.

### 2.3 Inspect the Output

The dump directory contains subdirectories (named by content hash):

```
$TRITON_DUMP_DIR/
  <hash>/
    kernel.ttir.mlir         # Triton dialect MLIR
    kernel.ttadapter.mlir    # Linalg/memref dialect MLIR
    kernel.npuir.mlir        # BiShengIR (NPU binary IR)
    npuir_passes/            # Per-pass AscendNPU-IR dumps
```

Quick inspection:

```bash
python research_artifacts/inspect_triton_dump.py "$TRITON_DUMP_DIR" --latest
```

### 2.4 How the Mock Works

The mock layer (`third_party/ascend/backend/compile_only_mock.py`) is
auto-activated at `import triton` time when `TRITON_COMPILE_ONLY=1` is set.
It:

- Patches `torch.randn`, `torch.zeros`, etc. to redirect `device='npu'` to CPU
- Patches `Tensor.to()` and `Tensor.npu()` to keep tensors on CPU
- Installs an import hook that patches `torch.npu.*` namespace after torch_npu loads
- Guards driver methods (`get_current_device`, `get_current_stream`, `load_binary`, etc.)
  to return stubs instead of calling into NPU hardware

No changes to kernel source code or calling conventions are needed.

### 2.5 Success Checks

After the run, check the dump directory:

- `kernel.ttir.mlir` + `kernel.ttadapter.mlir` present = Triton-Ascend pipeline ran
- `kernel.npuir.mlir` present = compile reached `bishengir-compile`
- `npuir_passes/` present = full downstream per-pass dumping enabled

The script may raise a Python exception after compilation (e.g., `torch.allclose`
fails). This does not invalidate the IR dumps.

### 2.6 Understanding the IR Stages

| File | Dialect | What It Shows |
|---|---|---|
| `kernel.ttir.mlir` | Triton (`tt.func`, `tt.load`, `tt.store`) | High-level tensor ops, program structure |
| `kernel.ttadapter.mlir` | Linalg/memref (`func.func`, `linalg.fill`) | Buffer semantics, Ascend-specific attributes |
| `kernel.npuir.mlir` | BiShengIR | Further lowered for NPU execution |

## 3. Common Failure Patterns

When a fresh environment cannot dump the full pipeline:

- `import triton` fails: Triton-Ascend was not installed
- `import torch_npu` fails: the Ascend PyTorch environment is incomplete
- `which bishengir-compile` fails: the downstream compiler is not on `PATH`
- `ASCEND_HOME_PATH` is empty: the Ascend toolkit was not configured
- `0 active drivers`: the compiler entrypoint is not visible to Triton in WSL

Debugging order:

1. Verify Python imports
2. Verify `bishengir-compile`
3. Verify `ASCEND_HOME_PATH`
4. Rerun with `TRITON_DEBUG=1`
5. Inspect the newest dump directory

## 4. Suggested Reading Order

If you are new to the compiler, read these pages in order:

1. [Compiler Structure and Ascend Hardware Model](./compiler_structure_and_hardware.md)
2. [Compiler Pass Guide with Ascend Hardware Rationale](./compiler_passes_and_hardware.md)
3. [Build and Dump IR for Ascend Compiler Research](./build_and_dump_ir.md) (this page)
4. [Fused Attention Example](./examples/04_fused_attention_example.md)
