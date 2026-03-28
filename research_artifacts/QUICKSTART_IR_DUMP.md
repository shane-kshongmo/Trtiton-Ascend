# Quickstart: Triton-Ascend Compile-Only IR Dump

Dump Triton compiler IR at each pipeline stage without NPU hardware.

## Prerequisites

| Component | Version | Location |
|---|---|---|
| Python venv | 3.12+ | Any virtualenv with triton-ascend wheel installed |
| CANN toolkit | 8.5.0 | `~/Ascend/ascend-toolkit/` (provides `bishengir-compile`) |
| Ninja | >= 1.11.1 | `~/.local/bin/ninja` |
| PyTorch + torch_npu | 2.7.x | Installed in the venv |
| triton-ascend wheel | 3.2.0 | Installed in the venv (see "Building from Source" below) |

No Ascend NPU hardware or `npu-smi` is required. The `ttir` and `ttadapter` stages
run entirely on the host CPU. The `npuir` stage (BiShengIR binary generation) will
fail without hardware -- this is expected and harmless.

## Step 1: Set Up the Environment

```bash
# Add Ninja to PATH
export PATH="$HOME/.local/bin:$PATH"

# Source CANN environment (provides bishengir-compile and CANN libraries)
source ~/Ascend/ascend-toolkit/set_env.sh

# Activate your Python virtualenv
source /path/to/your/venv/bin/activate
```

## Step 2: Verify Imports

Run these from `/tmp` (not from the repo root) to avoid importing from the
source tree instead of the installed wheel:

```bash
cd /tmp
python -c "import triton; print(triton.__version__, triton.__file__)"
python -c "import torch; import torch_npu; print('OK')"
python -c "from triton.backends.ascend.driver import NPUDriver; print('active:', NPUDriver.is_active())"
which bishengir-compile
```

Expected: triton 3.2.0 from site-packages, NPUDriver active = True,
bishengir-compile found in the CANN bin directory. The `npu-smi` warnings
are harmless.

## Step 3: Run the IR Dump

```bash
# Configure compile-only mode
export TRITON_ASCEND_ARCH=Ascend910_9599
export TRITON_ALWAYS_COMPILE=1
export TRITON_COMPILE_ONLY=1
export TRITON_KERNEL_DUMP=1
export TRITON_DEBUG=1
export TRITON_DUMP_DIR=/path/to/your/dump/dir

# Run from /tmp to avoid source-tree shadowing
cd /tmp
python /path/to/triton-ascend/research_artifacts/compile_only_layer_norm.py
```

The compiler will run the full pipeline. The `npuir` stage will fail with a
`get_arch` error (no NPU hardware) -- this is expected. The earlier stages
still produce their output.

## Step 4: Inspect the Output

The dump directory will contain two subdirectories (named by content hash):

```
$TRITON_DUMP_DIR/
  <hash1>/                          # From TRITON_DEBUG=1
    kernel.ttir.mlir                # Triton dialect MLIR (158 lines for layer-norm)
    kernel.ttadapter.mlir           # Linalg/memref dialect MLIR (197 lines)
  <hash2>/                          # From TRITON_KERNEL_DUMP=1
    _layer_norm_fwd_fused.ttir      # Same ttir content, named by kernel
    _layer_norm_fwd_fused.ttadapter # Same ttadapter content, named by kernel
```

Use the inspection tool for a quick summary:

```bash
cd /tmp
python /path/to/triton-ascend/research_artifacts/inspect_triton_dump.py \
    "$TRITON_DUMP_DIR" --latest
```

### Understanding the IR Stages

| File | Dialect | What It Shows |
|---|---|---|
| `kernel.ttir.mlir` | Triton (`tt.func`, `tt.load`, `tt.store`) | High-level tensor ops, program structure, pointer arithmetic |
| `kernel.ttadapter.mlir` | Linalg/memref (`func.func`, `linalg.fill`, `memref`) | Lowered to buffer semantics, Ascend-specific attributes (`mix_mode`, `parallel_mode`) |
| `kernel.npuir.mlir` | BiShengIR (not produced without HW) | Further lowered for NPU execution |

## Known Limitations

- **No `npuir` without hardware**: The `linalg_to_bin` stage calls `NPUUtils().get_arch()`
  which requires `npu-smi`. The `ttir` and `ttadapter` stages are the useful research
  artifacts.
- **`npu-smi` warnings**: Printed to stderr on every run. Harmless -- ignore them.
- **Run from `/tmp`**: Running from the repo root causes Python to import `triton` from
  the source tree at `python/triton/`, which lacks the compiled `_C` extension and will
  fail with import errors.
- **Cache invalidation**: Set `TRITON_ALWAYS_COMPILE=1` to force recompilation. Otherwise
  Triton uses `~/.triton/cache/` and may skip compilation entirely.

## Building the Wheel from Source

If you need to rebuild after modifying compiler passes:

```bash
cd /path/to/triton-ascend/python

TRITON_BUILD_WITH_CLANG_LLD=true \
TRITON_BUILD_PROTON=OFF \
TRITON_WHEEL_NAME="triton-ascend" \
MAX_JOBS=$(nproc) \
  python3 setup.py bdist_wheel

# Install the new wheel
pip install dist/triton_ascend-*.whl --force-reinstall --no-deps
```

Build requires: CMake >= 3.18, Ninja >= 1.11.1, pybind11 >= 2.13.1, Clang/LLD.

For incremental rebuilds after the initial build:

```bash
cd /path/to/triton-ascend
make all
```

## Environment Variables Reference

| Variable | Purpose |
|---|---|
| `TRITON_ASCEND_ARCH` | Target architecture (e.g., `Ascend910_9599`) |
| `TRITON_ALWAYS_COMPILE` | Force recompilation, bypass cache |
| `TRITON_COMPILE_ONLY` | Skip kernel launch (no hardware needed) |
| `TRITON_KERNEL_DUMP` | Dump kernel IR with kernel-name filenames |
| `TRITON_DEBUG` | Dump IR with `kernel.{stage}.mlir` filenames |
| `TRITON_DUMP_DIR` | Directory for all IR dump output |
