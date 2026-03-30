# Quickstart: Triton-Ascend Compile-Only IR Dump

Dump Triton compiler IR at each pipeline stage without NPU hardware.

## Prerequisites

| Component | Version | Location |
|---|---|---|
| Python venv | 3.10+ | Any virtualenv with triton-ascend wheel installed |
| CANN toolkit | 8.5.0 | `~/Ascend/ascend-toolkit/` (provides `bishengir-compile`) |
| Ninja | >= 1.12.0 | `~/.local/bin/ninja` |
| PyTorch + torch_npu | 2.7.x | Installed in the venv |
| triton-ascend wheel | 3.2.0 | Installed in the venv |

No Ascend NPU hardware or `npu-smi` is required. When `TRITON_COMPILE_ONLY=1`
is set, the Ascend backend automatically mocks NPU device operations so that
unmodified Triton kernel scripts run the full MLIR compilation pipeline.

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

Run from `/tmp` (not from the repo root) to avoid importing from the
source tree instead of the installed wheel:

```bash
cd /tmp
python -c "import triton; print(triton.__version__, triton.__file__)"
python -c "import torch; import torch_npu; print('OK')"
python -c "from triton.backends.ascend.driver import NPUDriver; print('active:', NPUDriver.is_active())"
which bishengir-compile
```

Expected: triton 3.2.0 from site-packages, NPUDriver active = True,
bishengir-compile found in the CANN bin directory.

## Step 3: Run the IR Dump

Set the environment variables and run any tutorial directly -- no code
changes needed:

```bash
export TRITON_ASCEND_ARCH=Ascend910_9599
export TRITON_ALWAYS_COMPILE=1
export TRITON_COMPILE_ONLY=1
export TRITON_KERNEL_DUMP=1
export TRITON_DEBUG=1
export TRITON_DUMP_DIR=/path/to/your/dump/dir

# Run from /tmp to avoid source-tree shadowing
cd /tmp
python /path/to/triton-ascend/third_party/ascend/tutorials/03-layer-norm.py
```

The mock layer automatically:
- Redirects `device='npu'` tensor operations to CPU
- Stubs out NPU device queries (device count, streams, etc.)
- Skips kernel launch while still running the full compilation pipeline

The script will print `[compile-only] NPU device mocking activated` to
confirm the mock is active. The `torch.allclose` validation at the end
will fail (expected -- the kernel didn't execute, output tensors are
uninitialized).

## Step 4: Inspect the Output

The dump directory will contain subdirectories (named by content hash):

```
$TRITON_DUMP_DIR/
  <hash>/
    kernel.ttir.mlir         # Triton dialect MLIR
    kernel.ttadapter.mlir    # Linalg/memref dialect MLIR
    kernel.npuir.mlir        # BiShengIR (NPU binary IR)
    npuir_passes/            # Per-pass AscendNPU-IR dumps
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
| `kernel.ttadapter.mlir` | Linalg/memref (`func.func`, `linalg.fill`, `memref`) | Lowered to buffer semantics, Ascend-specific attributes |
| `kernel.npuir.mlir` | BiShengIR | Further lowered for NPU execution |

## Other Tutorials

The same approach works for any tutorial or custom kernel script. Just set
the environment variables and run:

```bash
python /path/to/triton-ascend/third_party/ascend/tutorials/01-vector-add.py
python /path/to/triton-ascend/third_party/ascend/tutorials/03-layer-norm.py
```

## Known Limitations

- **Run from `/tmp`**: Running from the repo root causes Python to import `triton` from
  the source tree at `python/triton/`, which lacks the compiled `_C` extension.
- **Validation failures**: `torch.allclose(...)` checks at the end of tutorials will
  fail because the kernel didn't actually execute. This is expected.
- **Cache invalidation**: Set `TRITON_ALWAYS_COMPILE=1` to force recompilation.

## Environment Variables Reference

| Variable | Purpose |
|---|---|
| `TRITON_ASCEND_ARCH` | Target architecture (e.g., `Ascend910_9599`) |
| `TRITON_ALWAYS_COMPILE` | Force recompilation, bypass cache |
| `TRITON_COMPILE_ONLY` | Enable compile-only mode with NPU mocking |
| `TRITON_KERNEL_DUMP` | Dump kernel IR with kernel-name filenames |
| `TRITON_DEBUG` | Dump IR with `kernel.{stage}.mlir` filenames + enable npuir_passes |
| `TRITON_DUMP_DIR` | Directory for all IR dump output |
