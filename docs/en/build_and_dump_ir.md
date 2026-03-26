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

The `python setup.py install` step is important. A raw checkout is not enough
for a fresh environment because Triton's backend discovery walks the installed
`python/triton/backends/*` directories at import time. If Triton-Ascend has not
been installed into the active Python environment, a new user or agent may see
missing-backend errors even though the repo source tree exists.

If you need to rebuild the local `bishengir-compile` helper, apply the root-repo patch in
[`third_party/ascend_patches/0001-Fix-LLVM-20-compatibility-for-bishengir-compile.patch`](../third_party/ascend_patches/0001-Fix-LLVM-20-compatibility-for-bishengir-compile.patch)
to the `third_party/ascend/AscendNPU-IR` submodule before running its standalone build.

## 1.1 Full-Pipeline Prerequisites Checklist

Before trying to dump the full pipeline from any unchanged tutorial or custom
kernel harness, make sure the active environment satisfies all of the
following:

- `import triton` works in the Python interpreter you are about to use
- `import torch` and `import torch_npu` both work
- Triton-Ascend was installed with `python setup.py install` or from a built wheel
- `bishengir-compile` is available on `PATH`, or `TRITON_NPU_COMPILER_PATH` is set
- `ASCEND_HOME_PATH` is set to a valid Ascend toolkit installation

You can check the environment quickly with:

```bash
python - <<'PY'
import triton
import torch
import torch_npu
print("triton:", triton.__file__)
print("torch:", torch.__file__)
print("torch_npu:", torch_npu.__file__)
PY

which bishengir-compile
echo "$ASCEND_HOME_PATH"
```

If one of these checks fails, the full pipeline will not run end to end.

In WSL, compiler visibility matters just as much as installation. Even if the
Ascend toolkit exists on the Windows side, Triton's NPU backend inside WSL will
still report `0 active drivers` unless the compiler entrypoint is reachable from
the Linux environment. In practice, at least one of the following must be true:

- `bishengir-compile` is on the WSL `PATH`
- `bisheng` is on the WSL `PATH`
- `TRITON_NPU_COMPILER_PATH` points to a working compiler binary visible from WSL

If these are missing, a fresh agent will fail before any IR is dumped.

## 1.2 Quick Start For Humans

If you already have a working Triton-Ascend, PyTorch NPU, and Ascend compiler
environment, the shortest path is:

1. Activate the Python environment where Triton-Ascend was installed.
2. Export the dump variables from section 3.1.
3. Run:

```bash
python third_party/ascend/tutorials/03-layer-norm.py
```

4. Inspect the newest kernel dump directory:

```bash
python research_artifacts/inspect_triton_dump.py /path/to/kernel_dump_hash_dir
```

For a successful full compile-only dump, expect:

- `kernel.ttir.mlir`
- `kernel.ttadapter.mlir`
- `kernel.npuir.mlir`
- `npuir_passes/`

This quick path applies to more than `03-layer-norm.py`. It also works for any
kernel that already has a runnable Python entry point in the repo, as long as
that script eventually invokes a `@triton.jit` kernel.

## 1.3 Deterministic Runbook For A Fresh Agent

If a new agent or teammate needs to dump the full IR pipeline from scratch, use
this exact sequence and do not skip steps:

1. Activate the Python environment where Triton-Ascend was installed.
2. Verify `import triton`, `import torch`, and `import torch_npu`.
3. Verify `bishengir-compile` is visible on `PATH`.
4. Verify `ASCEND_HOME_PATH` is set.
5. Enable the dump environment variables shown in section 3.1.
6. Run the unchanged tutorial script:

```bash
python third_party/ascend/tutorials/03-layer-norm.py
```

7. Inspect the newest kernel dump directory:

```bash
python research_artifacts/inspect_triton_dump.py /path/to/kernel_dump_hash_dir
```

For a successful full compile-only dump, expect all of the following:

- `kernel.ttir.mlir`
- `kernel.ttadapter.mlir`
- `kernel.npuir.mlir`
- `npuir_passes/`

If `kernel.ttir.mlir` exists but `kernel.npuir.mlir` does not, the pipeline
started in Triton but did not reach the downstream AscendNPU-IR toolchain.
That usually means the environment is still missing a valid Ascend compiler
setup.

If the run fails with `0 active drivers`, stop and fix compiler visibility
before debugging the Python script. That error means Triton could not activate
the NPU backend at all.

## 2. Trigger The Pipeline Through Triton Compile

The generic way to dump IR is to trigger the normal Triton compile pipeline for a
`@triton.jit` kernel.

You do not need a special external dumper to make the compiler run. The usual
Triton compile conventions already do that:

- call the kernel through Triton JIT, for example `kernel.warmup(...)`
- or call `triton.compile(...)` on an `ASTSource`

For compile-time research, the important part is that you provide enough
compile-time type and constant information for the kernel to specialize. Real
execution is not required.

For example, the layer-norm tutorial kernel in
[`third_party/ascend/tutorials/03-layer-norm.py`](../third_party/ascend/tutorials/03-layer-norm.py)
can be compiled without launching on hardware by passing compile-time-only
arguments:

```python
import triton
from triton.compiler import ASTSource
from triton.backends.compiler import GPUTarget

src = ASTSource(
    fn=_layer_norm_fwd_fused,
    signature=signature,
    constants=constants,
    attrs=attrs,
)

kernel = triton.compile(
    src,
    target=GPUTarget("npu", "Ascend910_9599", 0),
    options={"debug": True, "num_warps": 1, "num_ctas": 1},
)
```

If you want to stay closer to the normal JIT entry point, the same compile
pipeline can also be triggered through a warmup-style call:

```python
_layer_norm_fwd_fused.warmup(
    mock_x, mock_y, mock_w, mock_b, mock_mean, mock_rstd,
    stride, N, eps,
    BLOCK_SIZE=128,
    grid=(M,),
    num_warps=1,
    num_ctas=1,
)
```

In both cases, Triton lowers the kernel through the usual stages:

- Python AST to TTIR
- TTIR cleanup
- Ascend TTIR-to-TTAdapter lowering
- optional later binary stages

For IR research, you can stop after `ttadapter` and skip final binary
generation.

### 2.1 Choose The Right Compile Entry

For a new kernel, choose the compile trigger that matches how much integration
already exists:

- use the unchanged script entry point when the repo already has a runnable example
- use `kernel.warmup(...)` when you can import the JIT kernel and provide mock arguments
- use `triton.compile(...)` when you want the most explicit and controlled compile harness

As a rule of thumb:

- start with the unchanged script for existing tutorials and examples
- switch to `warmup(...)` when the script does too much unrelated work after compilation
- switch to `triton.compile(...)` when you are building a one-off research harness for a new kernel

### 2.2 Generic Workflow For More Kernels

To dump a kernel other than layer norm, walk through this checklist:

1. find the `@triton.jit` function
2. identify its runtime tensor arguments
3. identify its scalar arguments
4. identify its `tl.constexpr` meta-parameters such as `BLOCK_SIZE`
5. identify the launch grid
6. choose one compile trigger from section 2.1
7. run with the dump variables from section 3.1

For existing tutorials, step 6 is often just:

```bash
python path/to/tutorial.py
```

For imported kernels, step 6 is often:

```python
kernel.warmup(
    *mock_args,
    grid=grid,
    **meta,
)
```

The generic compile-and-dump story is the same across kernels. What changes is
only the argument preparation and the launch metadata needed for
specialization.

### 2.3 Preferred Adaptation Strategy For Arbitrary Kernels

For a fresh agent, `kernel.warmup(...)` is usually the easiest generic path for
an arbitrary kernel because Triton can infer most compile-time details from the
JIT function itself.

Prefer the methods in this order:

1. unchanged script entry point
2. `kernel.warmup(...)`
3. `triton.compile(...)`

Use `triton.compile(...)` only when you specifically need an explicit
`ASTSource` harness. It is more flexible, but a new agent must then derive
items such as `signature`, `constants`, and `attrs` correctly for that kernel.

For most kernels, the practical adaptation recipe is:

1. import the module that defines the `@triton.jit` kernel
2. identify the JIT function object
3. create mock tensors with the right dtypes and shapes
4. pass realistic scalar arguments
5. pass all `tl.constexpr` meta-parameters explicitly
6. call `kernel.warmup(...)` with the correct `grid`
7. inspect the newest dump directory

A minimal skeleton looks like this:

```python
kernel.warmup(
    *mock_tensor_args,
    *scalar_args,
    grid=grid,
    num_warps=num_warps,
    num_ctas=num_ctas,
    **meta,
)
```

Where:

- `mock_tensor_args` match the runtime tensor parameters in order
- `scalar_args` match the runtime scalar parameters in order
- `meta` contains the `tl.constexpr` parameters such as `BLOCK_SIZE`
- `grid` matches the original launch configuration as closely as possible

If the kernel already has a Python wrapper function in the repo, read that
wrapper first. It is usually the fastest way to recover the correct argument
order, `grid`, and meta-parameters.

## 3. Dump The IR Files

There are two useful dump modes, depending on what you want to inspect.

### 3.1 Standard Triton Dump During Normal Compilation

If you mainly want the normal stage artifacts produced by the backend, enable
Triton's dump switches and compile the kernel normally.

Use an offline arch such as `Ascend910_9599`:

```bash
export TRITON_ASCEND_ARCH=Ascend910_9599
export TRITON_ALWAYS_COMPILE=1
export TRITON_COMPILE_ONLY=1
export TRITON_KERNEL_DUMP=1
export TRITON_DEBUG=1
export TRITON_DUMP_DIR=/mnt/d/work/git/triton-ascend/kernel_dumps
```

Then run the Python code that compiles the target kernel. The Ascend backend
writes stage outputs such as:

- `kernel.ttir.mlir`
- `kernel.ttadapter.mlir`
- `kernel.npuir.mlir` after the compile reaches the AscendNPU-IR toolchain
- `npuir_passes/` containing per-pass AscendNPU-IR dumps when debug dumping is enabled
- possibly later-stage files such as `kernel.llir.mlir` or `kernel.ll`, if the
  pipeline continues far enough

This is the default and most generic workflow because it relies only on Triton's
own compile entry points.

For the layer-norm tutorial, you can keep the source file unchanged and run it
through its normal entry point:

```bash
python third_party/ascend/tutorials/03-layer-norm.py
```

With `TRITON_COMPILE_ONLY=1`, the Triton JIT compile pipeline still runs, but
the Ascend launcher skips the actual kernel launch. This is often the simplest
way to trigger compilation and dump `kernel.ttir.mlir` and `kernel.ttadapter.mlir`
from an existing tutorial script.

One practical caveat is that the rest of the Python script still continues after
compilation. In `03-layer-norm.py`, tensor creation and later validation code
such as `torch.allclose(...)` still execute. That means:

- compile-only mode does not require rewriting the Triton kernel call
- the script may still fail later in Python because no real NPU kernel result was produced
- for pure IR dumping, this later failure is usually acceptable as long as the
  dump files were already written

If the compile reaches the `npubin` stage successfully, the backend now also
asks `bishengir-compile` to dump its own pass pipeline. Those files are copied
into the Triton dump directory under `npuir_passes/`.

`TRITON_DEBUG=1` is required for the downstream AscendNPU-IR pass tree. Using
`TRITON_KERNEL_DUMP=1` without `TRITON_DEBUG=1` is enough for Triton-stage dump
files, but not enough for `npuir_passes/`.

To inspect one dumped kernel directory quickly, you can use:

```bash
python research_artifacts/inspect_triton_dump.py /path/to/kernel_dump_hash_dir
```

Or point the helper at the root dump directory and let it select the newest
kernel automatically:

```bash
python research_artifacts/inspect_triton_dump.py "$TRITON_DUMP_DIR" --latest
```

If you want a cleaner compile-only flow with no follow-up validation failure,
use a dedicated warmup or `triton.compile(...)` driver instead.

### 3.1.1 Quick Success Checks

After the run, inspect the dump directory before worrying about later
Python-side validation failures. In compile-only research mode, the real
question is whether the compiler already produced the files you need.

To list the newest dump directories:

```bash
ls -dt "$TRITON_DUMP_DIR"/* | head
```

To inspect the newest one:

```bash
python research_artifacts/inspect_triton_dump.py "$TRITON_DUMP_DIR" --latest
```

Treat these results as the practical meaning of success:

- if `kernel.ttir.mlir` and `kernel.ttadapter.mlir` are present, the Triton-Ascend side ran
- if `kernel.npuir.mlir` is present, the compile reached `bishengir-compile`
- if `npuir_passes/` is present, full downstream per-pass dumping is enabled

The script may still raise an exception afterward because
`TRITON_COMPILE_ONLY=1` skips the actual NPU launch. That later exception does
not invalidate the IR dumps if the files were already written.

### 3.2 Per-Pass Research Dumps

If you want one file after each individual pass, you need a research helper that
replays the same Triton and Ascend passes one by one and writes the module after
each pass.

An example implementation is available in
[`research_artifacts/ascend_pipeline_research_dump.py`](../research_artifacts/ascend_pipeline_research_dump.py).
That script is not the generic compile API. It is only a convenience layer for
research when you want files such as:

```text
ascend_pass_dumps/
  layer_norm/
    00_initial_ast.ttir.mlir
    ttir_01_inliner.mlir
    ttir_02_combine.mlir
    ...
    ascend_18_triton_to_linalg.mlir
    final_ttadapter.mlir
```

For example, the layer-norm kernel was dumped into:

[`research_artifacts/layer_norm_ir_dumps/03-layer-norm`](../research_artifacts/layer_norm_ir_dumps/03-layer-norm)

That directory contains the TTIR cleanup sequence and the Ascend lowering
sequence up to `final_ttadapter.mlir`.

## 4. Common Failure Patterns

When a fresh environment cannot dump the full layer-norm pipeline, the problem
is usually one of these:

- `import triton` fails: Triton-Ascend was not installed into the active Python environment
- `import torch_npu` fails: the Ascend PyTorch environment is incomplete
- `which bishengir-compile` fails: the downstream compiler is not available on `PATH`
- `ASCEND_HOME_PATH` is empty: the Ascend toolkit was not configured
- only Triton-stage files are dumped: `TRITON_DEBUG=1` was missing, or the full backend toolchain was not usable
- `0 active drivers ([]). There should only be one.`: the WSL-side compiler entrypoint is still not visible to Triton

For a new agent, the fastest debugging order is:

1. verify Python imports
2. verify `bishengir-compile`
3. verify `ASCEND_HOME_PATH`
4. rerun with `TRITON_DEBUG=1`
5. inspect the newest dump directory with `inspect_triton_dump.py`

## 5. What The Dumps Mean

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

## 6. Suggested Reading Order

If you are new to the compiler, read these pages in order:

1. [Compiler Structure and Ascend Hardware Model](./compiler_structure_and_hardware.md)
2. [Compiler Pass Guide with Ascend Hardware Rationale](./compiler_passes_and_hardware.md)
3. [Build and Dump IR for Ascend Compiler Research](./build_and_dump_ir.md)
4. [Fused Attention Example](./examples/04_fused_attention_example.md)

## 7. Notes

- This guide is for compile-time research only.
- The generic workflow is: compile the Triton kernel normally and enable dump output.
- The same workflow applies to layer norm and other kernels. The only kernel-specific work is argument preparation and launch metadata.
- The per-pass workflow is a research helper layered on top of the same Triton compile pipeline.
- When you stop at `ttadapter`, you do not need final `npubin` generation.
- When you want AscendNPU-IR pass dumps too, you must let the compile reach `npubin`.
- A fresh agent or human should validate the Python packages and Ascend toolchain first, otherwise the unchanged tutorial entry point will fail before or during backend compilation.
- If you want final kernel generation later, you will need the Ascend toolkit and a valid `ASCEND_HOME_PATH`.
