# Triton-Ascend Pass Guide with Ascend Hardware Rationale

This guide documents the main Ascend-specific compilation passes wired into `third_party/ascend/backend/compiler.py`. Each pass is described in four dimensions:

- **IR effect**: what it rewrites
- **Hardware motivation**: which Ascend hardware concern pushed the pass into existence
- **Benefit**: what it improves for correctness or performance
- **Typical trigger patterns**: source-level Triton constructs that tend to require the pass

## 1. TTIR cleanup before Ascend lowering

These passes run in `make_ttir` before the Ascend-specific lowering chain.

| Pass | IR effect | Hardware motivation | Benefit | Typical trigger patterns |
|---|---|---|---|---|
| `inliner` | Inlines called Triton functions into the current module. | Backend analyses work better when producer-consumer structure is visible. | Exposes larger optimization regions for later memory and scheduling passes. | Helper functions inside Triton kernels. |
| `combine` | Folds simple Triton IR patterns into more compact forms. | Reduces noisy IR before structure-sensitive Ascend passes run. | Smaller IR, fewer redundant intermediates, easier pattern matching. | Repeated elementwise arithmetic, redundant reshapes or broadcasts. |
| `canonicalizer` | Normalizes equivalent IR patterns into standard form. | Ascend passes often expect regular shapes, pointer forms, and masks. | Improves pass robustness and match rate. | Any kernel with mixed arithmetic, shape transforms, or masks. |
| `reorder_broadcast` | Reorders broadcast-style operations. | Broadcast order affects later address and layout reasoning. | Helps preserve regular dataflow before memory lowering. | Tensor broadcasts around load, store, and pointwise math. |
| `cse` | Removes duplicate expressions. | Duplicate address and mask computations waste scalar work and may inflate temporary state. | Reduces IR size and temporary-value pressure. | Reused offsets, masks, shape values. |
| `licm` | Hoists loop-invariant computation. | Ascend kernels often loop over tiles because UB is limited. Invariant computations should not stay in the inner loop. | Cuts repeated scalar work and helps expose cleaner tiled loops. | Long-sequence kernels, reduction loops, tiled attention kernels. |
| `symbol_dce` | Removes dead symbols and unused functions. | Keeps the backend focused on the actual hot path. | Smaller compile surface and less debug noise. | Any module with unused helper code. |
| `loop_unroll` | Unrolls selected loops. | Unrolling can expose better vector/cube-friendly compute structure and reduce loop overhead. | Better pattern visibility for downstream lowering. | Small fixed-trip loops, block-local reductions, staged kernels. |

## 2. Ascend-specific TTIR to TTAdapter lowering

These are the important hardware-aware passes in `ttir_to_linalg`.

### 2.1 `auto_blockify`

- **IR effect**
  - Rewrites work partitioning to make block structure more explicit.
- **Hardware motivation**
  - Ascend kernels usually should not be launched with a GPU-style "huge grid and let the hardware sort it out" mentality.
  - The programming guide recommends matching work to physical core counts and iterating over remaining tasks inside the kernel.
- **Benefit**
  - Makes the kernel easier to schedule against real vector/core availability.
  - Helps reduce launch oversubscription overhead.
- **Typical trigger patterns**
  - Kernels with large flattened grids, especially long-sequence workloads like attention.

### 2.2 `dag_sync`

- **IR effect**
  - Adds synchronization-oriented scheduling information to the IR.
- **Hardware motivation**
  - Ascend performance depends heavily on overlapping data movement and computation.
  - Dependencies between data-in, compute, and data-out need to be made explicit enough for later passes and BiSheng to pipeline them.
- **Benefit**
  - Can shorten critical paths and prepare the IR for better pipeline formation.
- **Typical trigger patterns**
  - Kernels with staged loading and compute loops, especially when multi-buffering is enabled.

### 2.3 `dag_scope`

- **IR effect**
  - Refines scheduling scopes around DAG nodes and dependency regions.
- **Hardware motivation**
  - Ascend needs precise reasoning about which operations can overlap and which ones must stay ordered.
- **Benefit**
  - Improves scheduling precision and prevents overly conservative barriers.
- **Typical trigger patterns**
  - Kernels with multiple load-compute-store phases and mixed vector/cube behavior.

### 2.4 `dag_ssbuffer`

- **IR effect**
  - Adds or prepares buffering information for scheduled DAG regions.
- **Hardware motivation**
  - Multi-buffering is a first-class hardware-performance technique on Ascend, but it consumes precious UB space.
- **Benefit**
  - Helps expose or organize buffering opportunities for pipeline overlap.
- **Typical trigger patterns**
  - Tiled kernels where the next tile should be loaded while the current tile is being computed.

### 2.5 `triton_to_structure`

- **IR effect**
  - Rewrites Triton IR into a more structured form, especially around pointer and mask expressions.
- **Hardware motivation**
  - Structured accesses are much easier to map to Ascend data movement engines and on-chip buffers than arbitrary pointer arithmetic.
- **Benefit**
  - Preserves memory regularity and gives later passes a better chance to keep operations efficient.
- **Typical trigger patterns**
  - `tl.load`, `tl.store`, pointer arithmetic, range-generated offsets, structured tensor access.

### 2.6 `discrete_mask_access_conversion`

- **IR effect**
  - Converts irregular masked accesses into more explicit sequences such as load/select/store.
- **Hardware motivation**
  - Sparse or discontinuous masked traffic is a poor fit for hardware that wants regular transfers into UB.
  - The programming guide explicitly recommends moving data to UB first and then selecting values there when accesses are irregular.
- **Benefit**
  - Turns hard-to-lower irregular memory patterns into forms that are easier to buffer, select, and optimize.
- **Typical trigger patterns**
  - `tl.load` and `tl.store` with non-contiguous masks, masked atomics, causal masking with irregular access shapes.

### 2.7 `triton_to_annotation`

- **IR effect**
  - Converts Triton-side compile hints into backend-visible annotation operations.
- **Hardware motivation**
  - Some hardware guidance should survive frontend lowering so later passes and BiSheng can still act on it.
- **Benefit**
  - Preserves intent across stage boundaries.
- **Typical trigger patterns**
  - `tl.compile_hint` and similar backend-guidance constructs.

### 2.8 `triton_to_unstructure`

- **IR effect**
  - Lowers structured tensor accesses into more explicit loop-based or scalarized forms where required.
- **Hardware motivation**
  - Not all Triton tensor access patterns can be expressed as regular Ascend-friendly memory movement.
  - When access patterns are too irregular, the compiler needs a controlled fallback.
- **Benefit**
  - Guarantees correctness for hard cases while keeping the fallback localized.
- **Typical trigger patterns**
  - Indirect loads/stores, irregular gather/scatter behavior, discrete axes that cannot stay structured.

### 2.9 `triton_to_hivm`

- **IR effect**
  - Converts Triton operations into the Ascend HIVM dialect.
- **Hardware motivation**
  - HIVM is the key hardware-oriented IR used to represent Ascend execution behavior, especially synchronization and lower-level kernel structure.
- **Benefit**
  - Introduces an IR that can express hardware-facing synchronization and execution semantics more directly than TTIR.
- **Typical trigger patterns**
  - Block synchronization ops, hardware-aware memory/control patterns, CV-style kernels.

### 2.10 `triton_to_hfusion`

- **IR effect**
  - Converts eligible operations into the HFusion dialect.
- **Hardware motivation**
  - Elementwise fusion reduces intermediate memory traffic, which is especially valuable when UB capacity is limited and GM traffic is expensive.
- **Benefit**
  - Reduces temporary storage and can improve vector-core utilization.
- **Typical trigger patterns**
  - Chains of elementwise ops, histogram-style or fusion-friendly math.

### 2.11 `triton_to_llvm`

- **IR effect**
  - Lowers certain Triton constructs, especially inline assembly related ones, toward LLVM dialect operations.
- **Hardware motivation**
  - Some low-level backend hooks and intrinsics are only naturally expressed at the LLVM layer.
- **Benefit**
  - Preserves access to hardware-specific low-level functionality.
- **Typical trigger patterns**
  - Inline assembly or backend-specific intrinsic-style operations.

### 2.12 `bubble_up_operation`

- **IR effect**
  - Moves `extract` and `extract_slice`-like operations upward when profitable.
- **Hardware motivation**
  - Poorly placed slice/extract operations can create unnecessary loops and extra local buffering, both of which are costly under tight UB budgets.
- **Benefit**
  - Improves locality and removes avoidable loop structure.
- **Typical trigger patterns**
  - Kernels using `tl.extract_slice`, `tl.insert_slice`, or slice-heavy reshaping patterns.

### 2.13 second `triton_to_structure`

- **IR effect**
  - Re-regularizes the IR after several lowering steps.
- **Hardware motivation**
  - Intermediate lowering can temporarily make the IR less structured. Reintroducing structure before Linalg lowering helps final analysis and memory planning.
- **Benefit**
  - Gives the final lowering stage a cleaner, more analyzable input.
- **Typical trigger patterns**
  - Any kernel that mixes restructuring, fallback lowering, and backend-specific conversions.

### 2.14 `triton_to_linalg`

- **IR effect**
  - Converts remaining Triton operations into Linalg, SCF, MemRef, Arith, and related MLIR forms.
- **Hardware motivation**
  - Linalg-style IR is a much better substrate for downstream bufferization, loop lowering, and backend compilation than high-level TTIR.
- **Benefit**
  - Produces the `ttadapter` form that BiSheng can consume more effectively.
- **Typical trigger patterns**
  - Almost every kernel that reaches the normal SIMD-style Ascend compilation path.

## 3. Why these passes matter for Ascend memory hierarchy

The pass chain above is easiest to understand as a memory-regularization pipeline:

- keep accesses structured when possible,
- materialize irregular masks explicitly when necessary,
- move data into representations where UB/L1 planning is possible,
- reduce temporary tensors and extra slices,
- expose synchronization where overlap is possible,
- only fall back to explicit scalar loops when structure cannot be preserved.

This is why so many passes exist around masks, pointer expressions, synchronization, and restructuring. Ascend performance is often limited by memory movement shape and UB pressure before raw arithmetic throughput becomes the main problem.

## 4. Why these passes matter for vector and cube execution

The repository also exposes options such as:

- `tile_mix_vector_loop`
- `tile_mix_cube_loop`
- `enable_hivm_auto_cv_balance`
- `enable_mixed_cv`
- `enable_auto_vectorize_v2`

These options only make sense if the compiler preserves enough operation structure to decide:

- which loops are vector-like,
- which loops should map to cube-style matmul or CV execution,
- how much fusion is still safe before UB usage grows too much,
- where synchronization is required once mixed execution resources are involved.

For that reason, passes such as `triton_to_structure`, `triton_to_hivm`, `triton_to_hfusion`, and `triton_to_linalg` are not isolated lowering steps. Together they prepare the kernel for a hardware policy decision: how much vector work, how much cube work, and how much buffering/synchronization the final kernel should contain.

## 5. Why these passes matter for `04_fused_attention`

Fused attention is a particularly good case study because it combines nearly all difficult backend concerns:

- large sequence dimensions require loop tiling because UB cannot hold the full problem,
- the kernel mixes block pointers, reductions, and online softmax state,
- causal masking creates hardware-sensitive masked access and `where` behavior,
- `HEAD_DIM >= 256` forces chunked accumulator updates to control UB usage,
- the fixed-core outer scheduling style matches Ascend physical core recommendations,
- matmul-like `tl.dot` operations interact with vector/cube lowering choices,
- multi-stage processing naturally interacts with pipeline overlap and synchronization.

When you inspect the generated `ttir`, `ttadapter`, and `kernel.npuir.mlir` for fused attention, you are not just watching IR syntax change. You are watching the compiler progressively make the kernel fit:

- Ascend's physical core scheduling model,
- Ascend's on-chip memory budget,
- Ascend's preference for regular buffered transfers,
- Ascend's need for explicit synchronization in more complex pipelines.

## 6. Recommended research method

When studying a pass, use this order:

1. Find the pass registration in `third_party/ascend/backend/compiler.py`.
2. Find the pass implementation in `third_party/ascend/lib/...`.
3. Compare `kernel.ttir.mlir` and `kernel.ttadapter.mlir`.
4. Ask which Ascend hardware concern became easier after the rewrite:
   - fewer irregular transfers?
   - less UB pressure?
   - more explicit synchronization?
   - better vector/cube partitioning?
   - smaller temporary tensors?
5. Confirm whether BiSheng output reports UB size or related memory diagnostics.

That workflow keeps the analysis grounded in code, artifacts, and documented hardware behavior rather than in guesswork.
