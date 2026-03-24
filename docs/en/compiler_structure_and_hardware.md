# Triton-Ascend Compiler Structure and Ascend Hardware Model

This document explains how the Triton-Ascend compiler is organized and why its structure is shaped by Ascend NPU hardware constraints. It is written for beginner AI compiler developers who want to connect source-level Triton code, MLIR passes, and final Ascend kernel generation.

## 1. Why the Ascend hardware model matters

On Triton-Ascend, compiler design is not just about converting one IR into another. The compiler must also produce code that matches how Ascend devices actually execute work.

The most important hardware facts exposed by this repository are:

- **Task scheduling is tied to physical core counts.**
  On Ascend, the recommended programming model is to bind work to the number of physical cores and iterate over remaining tasks inside the kernel, instead of relying on large GPU-style oversubscription. The programming guide explicitly recommends fixing the number of launched cores to hardware core counts and looping over task blocks inside the kernel.
- **Ascend distinguishes vector-core-heavy and cube-core-heavy execution.**
  Pure vector kernels mainly target vector cores. CV-style kernels involve both vector and cube resources, and the compiler needs to preserve enough structure for downstream tools to choose good tiling and scheduling strategies.
- **On-chip memory pressure dominates many compiler decisions.**
  Global memory (GM) is large but slow. The unified buffer (UB) and related on-chip storage are much faster but much smaller. For A2-series devices, the programming guide calls out a UB size of 192 KB. Many lowering and optimization choices exist to keep data movement local, reduce temporary buffers, and avoid UB overflow.
- **Multi-buffering improves throughput but increases memory demand.**
  Ascend favors pipelining data-in, compute, and data-out. That is good for throughput, but it also means the compiler must reason about extra buffering space and synchronization.
- **Cross-core synchronization is part of the execution model.**
  Ascend-specific synchronization passes and backend options exist because some kernels need explicit coordination across blocks or pipeline stages.
- **Alignment and layout rules affect performance.**
  The programming guide documents different tail-axis alignment expectations for vector-only and CV kernels. This is why layout-cleanup, masking, and restructuring passes matter even when the original Triton kernel looks simple.

In short, the Ascend compiler pipeline is trying to preserve just enough high-level tensor structure for good hardware mapping while also rewriting awkward Triton patterns into forms that BiSheng and the Ascend backend can schedule, bufferize, and synchronize correctly.

## 2. End-to-end compiler structure

The main path for user kernels starts in Python and is then handed to the Ascend backend:

```text
Triton Python kernel
  -> AST to TTIR
  -> Ascend TTIR cleanup and restructuring
  -> TTAdapter / Linalg-oriented MLIR
  -> BiSheng / AscendNPU IR compilation
  -> kernel object (.o / reloc object)
```

The main code locations are:

- `python/triton/compiler/compiler.py`
  - Owns the generic Triton compilation entrypoint.
  - Builds the stage pipeline by asking the active backend to register stages.
  - Handles caching, dump files, and stage-by-stage artifact storage.
- `third_party/ascend/backend/compiler.py`
  - Owns the Ascend-specific compilation stages.
  - Registers the `ttir`, `ttadapter`, and `npubin` stages for the NPU path.
  - Selects different BiSheng compile flows for `compile_on_910_95` versus A2/A3-style targets.
- `third_party/ascend/lib/...`
  - Contains the Ascend-specific MLIR passes that rewrite Triton IR into hardware-aware forms.
- `third_party/ascend/backend/utils.py`
  - Resolves toolchain dependencies such as `LLVM_ROOT`, `MLIR_ROOT`, and `bishengir-compile`.
- `third_party/ascend/backend/driver.py`
  - Handles runtime-side loading of compiled kernels. This is not required for compile-only research, but it is useful for understanding the final runtime contract.

## 3. Real pipeline stages used by the Ascend backend

The active NPU backend is implemented in `third_party/ascend/backend/compiler.py` inside `AscendBackend.add_stages`.

### 3.1 Stage 1: `ttir`

The `ttir` stage is created by `make_ttir`.

Its job is to clean and simplify frontend-generated Triton IR before hardware-specific lowering starts. The pass manager includes:

- `inliner`
- `combine`
- `canonicalizer`
- `reorder_broadcast`
- `cse`
- `licm`
- `symbol_dce`
- `loop_unroll`

Why this matters on Ascend:

- Cleaner TTIR gives later passes more regular pointer and mask expressions.
- Hoisted and deduplicated expressions reduce downstream temporary values and memory pressure.
- Loop unrolling can expose tighter block-local compute patterns that are easier to map to vector/cube execution.

### 3.2 Stage 2: `ttadapter`

The `ttadapter` stage is created by `ttir_to_linalg`.

This is the most important stage for compiler research because it is where Triton IR is progressively reshaped into forms that the Ascend backend and BiSheng can understand.

This stage runs:

- `auto_blockify`
- optionally `dag_sync`, `dag_scope`, `dag_ssbuffer`
- `triton_to_structure`
- `discrete_mask_access_conversion`
- `triton_to_annotation`
- `triton_to_unstructure`
- `triton_to_hivm`
- `triton_to_hfusion`
- `triton_to_llvm`
- `bubble_up_operation`
- `triton_to_structure` again
- `triton_to_linalg`

The result is stored as `kernel.ttadapter.mlir` when dumping is enabled.

Why this matters on Ascend:

- This stage decides whether accesses stay structured enough for efficient memory movement or fall back to more scalar, loop-oriented lowering.
- It introduces hardware-visible concepts needed for synchronization, memory planning, and backend code generation.
- It is the stage where many UB-sensitive and pipeline-sensitive transformations become visible.

### 3.3 Stage 3: `npubin`

The final NPU stage is either:

- `linalg_to_bin_enable_npu_compile_910_95`, or
- `linalg_to_bin_enable_npu_compile_A2_A3`

depending on target settings.

Both routes:

- parse metadata from the adapter IR,
- assemble BiSheng compile options,
- invoke `bishengir-compile`,
- save debug output such as `kernel.npuir.mlir` when debugging is enabled,
- emit the final kernel object.

Why this matters on Ascend:

- This is where backend policy knobs become real hardware compilation choices.
- Options such as `multibuffer`, `sync_solver`, `unit_flag`, `tile_mix_vector_loop`, and `tile_mix_cube_loop` directly express hardware tradeoffs rather than generic compiler tuning.

## 4. Hardware-aware interpretation of backend options

`NPUOptions` in `third_party/ascend/backend/compiler.py` is worth reading as a hardware policy surface, not just a list of flags.

The most important option groups are:

- **Memory and pipeline pressure**
  - `multibuffer`
  - `enable_ubuf_saving`
  - `limit_auto_multi_buffer_only_for_local_buffer`
  - `limit_auto_multi_buffer_of_local_buffer`
  - `set_workspace_multibuffer`
- **Task/block mapping**
  - `auto_blockify_size`
  - `enable_auto_bind_sub_block`
  - `TRITON_ALL_BLOCKS_PARALLEL`
- **Synchronization**
  - `sync_solver`
  - `unit_flag`
  - `inject_barrier_all`
  - `inject_block_all`
  - `disable_auto_inject_block_sync`
- **Vector/cube policy**
  - `tile_mix_vector_loop`
  - `tile_mix_cube_loop`
  - `enable_hivm_auto_cv_balance`
  - `enable_mixed_cv`
- **Lowering shape and codegen behavior**
  - `enable_nd2nz_on_vector`
  - `enable_drop_unit_dims`
  - `enable_flatten`
  - `enable_auto_vectorize_v2`

These options exist because the compiler is trying to balance:

- UB usage versus throughput,
- regular vectorizable loops versus irregular fallback loops,
- data transfer overlap versus synchronization cost,
- vector-core utilization versus cube-core utilization.

## 5. A2/A3 and 910_95 backend differences

The repository does not treat all Ascend targets as identical.

The two important backend branches are:

- **A2/A3 path**
  - uses `linalg_to_bin_enable_npu_compile_A2_A3`,
  - enables a broader set of memory display and hardware policy options,
  - includes A2/A3-oriented handling for sync solver, vector/cube tiling, and related tuning switches.
- **910_95 path**
  - uses `linalg_to_bin_enable_npu_compile_910_95`,
  - carries a smaller but similar set of hardware control knobs,
  - still exposes the same core compiler themes: memory pressure, synchronization, and hardware-shaped scheduling.

For research, this means you should not assume a single universal "Ascend lowering story." The frontend stages are shared, but final compilation policy can diverge depending on hardware generation and backend mode.

## 6. Compile-only research workflow

For offline compiler research, you usually do not need to launch kernels on NPU hardware. The important goal is to make the compiler dump all intermediate artifacts.

Recommended environment variables:

```bash
export TRITON_ALWAYS_COMPILE=1
export TRITON_KERNEL_DUMP=1
export TRITON_DEBUG=1
export TRITON_DUMP_DIR=$PWD/triton_dumps
```

Recommended toolchain variables:

```bash
export LLVM_ROOT=/path/to/llvm-install
export MLIR_ROOT=/path/to/llvm-install
export ASCEND_HOME_PATH=/path/to/ascend-toolkit
export PATH=/path/to/bishengir/bin:$PATH
```

Research artifacts to preserve:

- `kernel.ttir.mlir`
- `kernel.ttadapter.mlir`
- `kernel.npuir.mlir`
- metadata JSON
- final kernel object
- any BiSheng output that reports **UB size** or memory-allocation diagnostics

Those artifacts are enough to study:

- which passes triggered,
- how structured tensor code became hardware-aware IR,
- where memory pressure appears,
- which backend policy knobs likely shaped final code generation.

## 7. What to read next

- [compiler_passes_and_hardware](./compiler_passes_and_hardware.md)
- [Architecture Design and Core Features](./architecture_design_and_core_features.md)
- [Triton Operator Development Guide](./programming_guide.md)
- [Fused Attention](./examples/04_fused_attention_example.md)
- [Triton-Ascend Debugging Guide](./debug_guide/debugging.md)
