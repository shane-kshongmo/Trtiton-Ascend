# Build and Dump IR for Ascend Compiler Research

This guide shows how to build Triton-Ascend in WSL and how to dump the IR produced by each Ascend compiler pass.

The workflow is designed for compiler research:

- you can stop after `ttadapter`
- you do not need NPU hardware
- you do not need the CANN toolkit for the pass-dump workflow shown here

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
  python3.12-dev \
  libxml2-dev \
  libzstd-dev
```

Create and prepare a Python virtual environment:

```bash
python3 -m venv ~/venvs/triton-ascend
source ~/venvs/triton-ascend/bin/activate
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

python setup.py install
```

If the import works, the compiler build is ready:

```bash
python - <<'PY'
import triton
print(triton.__file__)
PY
```

## 2. Dump IR After Each Pass

For research, use the helper script:

```bash
/mnt/d/work/git/vTriton/ascend_pipeline_research_dump.py
```

It runs the Ascend pipeline pass by pass, writes one MLIR file after each pass, and stops before the binary stage.

Example output layout:

```text
ascend_pass_dumps/
  to_buffer/
    00_initial_ast.ttir.mlir
    ttir_01_inliner.mlir
    ttir_02_combine.mlir
    ...
    ascend_18_triton_to_linalg.mlir
    final_ttadapter.mlir
```

The same structure is generated for the fused-attention example:

```text
ascend_pass_dumps/
  fused_attention/
    00_initial_ast.ttir.mlir
    ttir_01_inliner.mlir
    ...
    final_ttadapter.mlir
```

## 3. Run The Dumper

Use the default offline arch `Ascend910_9599`:

```bash
PATH=/home/shane/venvs/triton-ascend/bin:$PATH \
TRITON_ASCEND_ARCH=Ascend910_9599 \
/home/shane/venvs/triton-ascend/bin/python \
/mnt/d/work/git/vTriton/ascend_pipeline_research_dump.py \
  --kernel to_buffer \
  --output /mnt/d/work/git/vTriton/ascend_pass_dumps
```

Run the fused-attention case the same way:

```bash
PATH=/home/shane/venvs/triton-ascend/bin:$PATH \
TRITON_ASCEND_ARCH=Ascend910_9599 \
/home/shane/venvs/triton-ascend/bin/python \
/mnt/d/work/git/vTriton/ascend_pipeline_research_dump.py \
  --kernel fused_attention \
  --output /mnt/d/work/git/vTriton/ascend_pass_dumps
```

## 4. What The Dumps Mean

The files are ordered in the same direction as the compiler pipeline:

- `00_initial_ast.ttir.mlir` is the raw Triton IR produced from the Python kernel
- `ttir_01_*` to `ttir_08_*` are the standard TTIR cleanup passes
- `ascend_01_*` to `ascend_18_*` are Ascend-specific lowering passes
- `final_ttadapter.mlir` is the last useful IR stage for research and performance modeling

For fused attention, the most interesting files are usually:

- `ttir_01_inliner.mlir`
- `ttir_04_reorder_broadcast.mlir`
- `ascend_01_auto_blockify.mlir`
- `ascend_09_triton_to_structure.mlir`
- `ascend_10_discrete_mask_access_conversion.mlir`
- `ascend_13_triton_to_hivm.mlir`
- `ascend_14_triton_to_hfusion.mlir`
- `ascend_18_triton_to_linalg.mlir`
- `final_ttadapter.mlir`

## 5. Suggested Reading Order

If you are new to the compiler, read these pages in order:

1. [Compiler Structure and Ascend Hardware Model](./compiler_structure_and_hardware.md)
2. [Compiler Pass Guide with Ascend Hardware Rationale](./compiler_passes_and_hardware.md)
3. [Build and Dump IR for Ascend Compiler Research](./build_and_dump_ir.md)
4. [Fused Attention Example](./examples/04_fused_attention_example.md)

## 6. Notes

- This guide is for compile-time research only.
- The pipeline is intentionally stopped before `npubin`.
- If you want final kernel generation later, you will need the Ascend toolkit and a valid `ASCEND_HOME_PATH`.
