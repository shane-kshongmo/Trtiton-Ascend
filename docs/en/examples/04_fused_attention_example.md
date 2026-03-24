# Fused Attention

This section implements a **fused attention forward pass kernel of the Flash Attention v2 style** based on **Triton**, which is applicable to the Ascend NPU platform. The implementation supports:
- **Causal and non-causal attention**
- **Tiling for processing long sequences**
- **Max-shifted softmax for numerical stability optimization**

The overall structure contains two core Triton kernels:
1. `_attn_fwd_inner`: performs attention computation between a single query block and key/value blocks (causal masks are processed in phases).
2. `_attn_fwd`: schedules all query blocks and manages the block pointer, accumulator, and normalization.

The `attention` function is encapsulated as a callable function using PyTorch `autograd.Function` and is verified for precision alignment with `torch_npu.npu_fusion_attention`.

```Python
import pytest
import torch
import torch_npu
import triton
import triton.language as tl


DEVICE = "npu"


@triton.jit
def _attn_fwd_inner(acc_ptr, l_i, m_i, q,  # Accumulator, local l, local m, query vector
                    K_block_ptr, V_block_ptr,  # Key and value block pointers for current stage
                    start_m, qk_scale,  # Starting position of current query block, qk scale factor
                    BLOCK_M: tl.constexpr, HEAD_DIM: tl.constexpr, BLOCK_N: tl.constexpr,  # Block size constants
                    STAGE: tl.constexpr, offs_m: tl.constexpr, offs_n: tl.constexpr,  # Current stage flag, m and n offset indices
                    N_CTX: tl.constexpr, fp8_v: tl.constexpr):  # Total context length, whether to enable FP8 for value precision
    # Set the processing range [lo, hi) for the current stage (in column block units)
    # causal = true
    # stage = 1
    # Causal attention, as the name implies, restricts the flow of information during computation,
    # only allowing the model to see the current and previous positions.
    # In other words, the output at the current position can only depend on the input at or before this position,
    # and cannot access information from future positions.
    # Causal attention ensures sequential order and prevents "leakage of future information."
    # But the following logic will also be triggered
    if STAGE == 1:
        # Stage 1: process all tokens before the query block
        tl.static_assert(BLOCK_M >= BLOCK_N)
        lo, hi = 0, start_m * BLOCK_M
    elif STAGE == 2:
        # Stage 2: process the current query block
        tl.static_assert(BLOCK_M >= BLOCK_N)
        lo, hi = start_m * BLOCK_M, (start_m + 1) * BLOCK_M
        lo = tl.multiple_of(lo, BLOCK_M)  # Align starting position
    # causal = False (no need for masking)
    else:
        lo, hi = 0, N_CTX  # Process the entire context

    # Adjust K and V block pointers to the starting position `lo`
    K_block_ptr = tl.advance(K_block_ptr, (lo, 0))  # K is [HEAD_DIM, N_CTX], shift along the second dim by lo
    V_block_ptr = tl.advance(V_block_ptr, (lo, 0))  # V is [N_CTX, HEAD_DIM], shift along the first dim by lo

    # Index mapping for the accumulator , used for slicing when HEAD_DIM >= 256
    row = tl.arange(0, BLOCK_M)[:, None]
    col_head_dim = tl.arange(0, HEAD_DIM)[None, :]
    block2d_acc = row * HEAD_DIM + col_head_dim

    # Iterate over all k, v blocks in the current stage and accumulate the output
    for start_n in range(lo, hi, BLOCK_N):  # Process BLOCK_N columns at a time
        start_n = tl.multiple_of(start_n, BLOCK_N)  # Align column start position
        # -- Compute qk ----
        k = tl.load(K_block_ptr)
        # Modify K
        trans_k = tl.trans(k)
        qk = tl.dot(q, trans_k)
        # Apply causal mask for STAGE 2
        if STAGE == 2:
            mask = offs_m[:, None] >= (start_n + offs_n[None, :])  # Construct upper triangular mask
            qk = qk * qk_scale + tl.where(mask, 0, -1.0e6)  # Set invalid positions to -∞
            m_ij = tl.maximum(m_i, tl.max(qk, 1))  # Update m_ij = max(m_i, max(qk))
            qk -= m_ij[:, None]  # Subtract max for softmax stability
        else:
            qk = qk * qk_scale
            m_ij = tl.maximum(m_i, tl.max(qk, 1))  # Scaled max
            qk = qk - m_ij[:, None]  # Stabilize

        # Softmax weights p = exp(qk)
        p = tl.math.exp(qk)

        # Convert softmax weight type depending on FP8 usage
        if fp8_v:
            p_cast = p.to(tl.float8e5)  # Convert to FP8 format (save memory)
        else:
            p_cast = p.to(k.dtype)

        v = tl.load(V_block_ptr)  # Load corresponding V block
        pv = tl.dot(p_cast, v)
        l_ij = tl.sum(p, 1)  # Softmax denominator (sum of each row)
        # -- Update m_i and l_i
        alpha = tl.math.exp(m_i - m_ij)  # Update factor: exp difference between old and new max
        l_i = l_i * alpha + l_ij  # Update softmax denominator
        # -- Update output accumulator --
        if HEAD_DIM < 256:
            acc_ptr = acc_ptr * alpha[:, None]
            acc_ptr = tl.dot(p_cast, v, acc_ptr)
        else:
            # 1. Load current slice of accumulator
            acc = tl.load(acc_ptr + block2d_acc)
            # 2. Update in slices (split by 1/4 of BLOCK_M to avoid ub overflow)
            for i in range(4):
                # Calculate start/end rows for current slice
                offset = i * (BLOCK_M // 4)
                # Extract slice data
                acc_i = tl.extract_slice(acc, (offset, 0), (BLOCK_M // 4, HEAD_DIM), (1, 1))
                alpha_i = tl.extract_slice(alpha, [offset], [BLOCK_M // 4], [1])
                pv_i = tl.extract_slice(pv, (offset, 0), (BLOCK_M // 4, HEAD_DIM), (1, 1))
                # Incrementally update slice: acc = acc * alpha + pv
                acc_i = acc_i * alpha_i[:, None] + pv_i
                # Write updated slice back to accumulator
                acc = tl.insert_slice(acc, acc_i, (offset, 0), (BLOCK_M // 4, HEAD_DIM), (1, 1))
            # 3. updated accumulator
            tl.store(acc_ptr + block2d_acc, acc)

        m_i = m_ij  # Update current block max
        # Advance V and K block pointers to next BLOCK_N range
        V_block_ptr = tl.advance(V_block_ptr, (BLOCK_N, 0))
        K_block_ptr = tl.advance(K_block_ptr, (BLOCK_N, 0))
    # Return accumulated output acc_ptr, softmax denominator l_i, and max value m_i
    return acc_ptr, l_i, m_i


@triton.jit
def _attn_fwd(Q, K, V, M, Out, acc, sm_scale,
              stride_qz: tl.constexpr, stride_qh: tl.constexpr, stride_qm: tl.constexpr, stride_qk: tl.constexpr,
              stride_kz: tl.constexpr, stride_kh: tl.constexpr, stride_kn: tl.constexpr, stride_kk: tl.constexpr,
              stride_vz: tl.constexpr, stride_vh: tl.constexpr, stride_vn: tl.constexpr, stride_vk: tl.constexpr,
              stride_oz: tl.constexpr, stride_oh: tl.constexpr, stride_om: tl.constexpr, stride_on: tl.constexpr,
              Z: tl.constexpr, H: tl.constexpr,
              N_CTX: tl.constexpr,
              HEAD_DIM: tl.constexpr,
              BLOCK_M: tl.constexpr,
              BLOCK_N: tl.constexpr,
              STAGE: tl.constexpr
              ):
    # Total number of blocks in sequence dimension (M)
    NUM_BLOCKS_M = N_CTX // BLOCK_M
    # Total tasks = number of sequence blocks × batch size (Z) × number of attention heads (H)
    NUM_BLOCKS = NUM_BLOCKS_M * Z * H

    # Current M-dimension block index
    pid = tl.program_id(0)

    for block_idx in range(pid, NUM_BLOCKS, 20):
        task_hz_idx = block_idx // NUM_BLOCKS_M
        task_m_idx = block_idx % NUM_BLOCKS_M
        off_z = task_hz_idx // H
        off_h = task_hz_idx % H
        qvk_offset = off_z.to(tl.int64) * stride_qz + off_h.to(tl.int64) * stride_qh
        # Create block pointers for Q, K, V, Output
        Q_block_ptr = tl.make_block_ptr(
            base=Q + qvk_offset,
            shape=(N_CTX, HEAD_DIM),
            strides=(stride_qm, stride_qk),
            offsets=(task_m_idx * BLOCK_M, 0),
            block_shape=(BLOCK_M, HEAD_DIM),
            order=(1, 0),
        )
        V_block_ptr = tl.make_block_ptr(
            base=V + qvk_offset,
            shape=(N_CTX, HEAD_DIM),
            strides=(stride_vn, stride_vk),
            offsets=(0, 0),
            block_shape=(BLOCK_N, HEAD_DIM),
            order=(1, 0),
        )
        K_block_ptr = tl.make_block_ptr(
            base=K + qvk_offset,
            shape=(N_CTX, HEAD_DIM),
            strides=(stride_kn, stride_kk),
            offsets=(0, 0),
            block_shape=(BLOCK_N, HEAD_DIM),
            order=(1, 0),
        )
        O_block_ptr = tl.make_block_ptr(
            base=Out + qvk_offset,
            shape=(N_CTX, HEAD_DIM),
            strides=(stride_om, stride_on),
            offsets=(task_m_idx * BLOCK_M, 0),
            block_shape=(BLOCK_M, HEAD_DIM),
            order=(1, 0),
        )
        # Initialize offsets
        offs_m = task_m_idx * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = tl.arange(0, BLOCK_N)

        m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
        l_i = tl.zeros([BLOCK_M], dtype=tl.float32) + 1.0

        # Initialize accumulator
        if HEAD_DIM < 256:
            acc_ptr = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
        else:
            acc_offset = (
                off_z.to(tl.int64) * stride_qz // stride_qm * HEAD_DIM +
                off_h.to(tl.int64) * stride_qh // stride_qm * HEAD_DIM +
                task_m_idx * BLOCK_M * HEAD_DIM
            )
            acc_ptr = acc + acc_offset

        q = tl.load(Q_block_ptr)

        # stage 1: off-band
        # For causal = True, STAGE = 3 and _attn_fwd_inner gets 1 as its STAGE
        # For causal = False, STAGE = 1, and _attn_fwd_inner gets 3 as its STAGE
        if STAGE & 1:
            acc_ptr, l_i, m_i = _attn_fwd_inner(acc_ptr, l_i, m_i, q, K_block_ptr, V_block_ptr,  #
                                                task_m_idx, sm_scale,  #
                                                BLOCK_M, HEAD_DIM, BLOCK_N,  #
                                                4 - STAGE, offs_m, offs_n, N_CTX, V.dtype.element_ty == tl.float8e5  #
                                                )
        # stage 2: on-band
        if STAGE & 2:
            # barrier makes it easier for compiler to schedule the
            # two loops independently
            acc_ptr, l_i, m_i = _attn_fwd_inner(acc_ptr, l_i, m_i, q, K_block_ptr, V_block_ptr,  #
                                                task_m_idx, sm_scale,  #
                                                BLOCK_M, HEAD_DIM, BLOCK_N,  #
                                                2, offs_m, offs_n, N_CTX, V.dtype.element_ty == tl.float8e5  #
                                                )

        m_i += tl.math.log(l_i)
        if HEAD_DIM < 256:
            accumulator = acc_ptr / l_i[:, None]
        else:
            row = tl.arange(0, BLOCK_M)[:, None]
            col_head_dim = tl.arange(0, HEAD_DIM)[None, :]
            block2d_acc = row * HEAD_DIM + col_head_dim
            accumulator = tl.load(acc_ptr + block2d_acc)
            accumulator = accumulator / l_i[:, None]

        m_ptrs = M + task_hz_idx * N_CTX + offs_m

        tl.store(m_ptrs, m_i)
        tl.store(O_block_ptr, accumulator.to(Out.type.element_ty))


class _attention(torch.autograd.Function):

    @staticmethod
    def forward(ctx, q, k, v, causal, sm_scale, BM, BN):
        """
        Forward computation interface:
        Args:
            ctx: Context object
            q: Query tensor (Q), shape [Z, H, N_CTX, HEAD_DIM]
            k: Key tensor (K), shape [Z, H, N_CTX, HEAD_DIM]
            v: Value tensor (V), shape [Z, H, N_CTX, HEAD_DIM]
            causal: Whether to enable causal attention
            sm_scale: Scaling factor for QK product
            BM: Q block size (BLOCK_M)
            BN: K/V block size (BLOCK_N)
        Returns:
            o: Attention output tensor, shape [Z, H, N_CTX, HEAD_DIM]
        """
        # shape constraints
        HEAD_DIM_Q, HEAD_DIM_K = q.shape[-1], k.shape[-1]
        # when v is in float8_e5m2 it is transposed.
        HEAD_DIM_V = v.shape[-1]
        assert HEAD_DIM_Q == HEAD_DIM_K and HEAD_DIM_K == HEAD_DIM_V
        assert HEAD_DIM_K in {16, 32, 64, 128, 256}

        o = torch.empty_like(q)
        stage = 3 if causal else 1
        extra_kern_args = {}
        

        # Number of NPU cores (adjust based on hardware)
        num_cores = 20
        acc = torch.zeros((q.shape[0], q.shape[1], q.shape[2], HEAD_DIM_K), dtype=torch.float32, device=q.device)
        M = torch.empty((q.shape[0], q.shape[1], q.shape[2]), device=q.device, dtype=torch.float32)

        _attn_fwd[(num_cores,)](
            q, k, v, M, o, acc, sm_scale,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            o.stride(0), o.stride(1), o.stride(2), o.stride(3),
            q.shape[0], q.shape[1], N_CTX=q.shape[2],
            HEAD_DIM=HEAD_DIM_K,
            BLOCK_M=BM,
            BLOCK_N=BN,
            STAGE=stage,
            **extra_kern_args)

        ctx.save_for_backward(q, k, v, o, M)
        ctx.sm_scale = sm_scale
        ctx.HEAD_DIM = HEAD_DIM_K
        ctx.causal = causal
        return o

attention = _attention.apply


@pytest.mark.parametrize("Z, H, N_CTX, HEAD_DIM, causal, dtype, BM, BN", [
    (1, 1, 128, 128, False, torch.float16, 32, 128),
    (1, 1, 128, 128, False, torch.bfloat16, 64, 128),
    (1, 2, 256, 256, False, torch.bfloat16, 32, 256),
    (2, 2, 128, 256, False, torch.float16, 64, 128),
    (4, 32, 64, 64, False, torch.float16, 32, 64),
    (4, 32, 1024, 64, False, torch.bfloat16, 64, 128),
    (4, 32, 4096, 64, False, torch.float16, 128, 128),
])
def test_op(Z, H, N_CTX, HEAD_DIM, causal, dtype, BM, BN):
    # Filter out non-integer cases; N_CTX must be divisible by BM and BN, and HEAD_DIM must be divisible by 16.
    if N_CTX % BM != 0 or N_CTX % BN != 0 or HEAD_DIM % 16 != 0:
        pytest.skip("Skipping non-divisible case")

    torch.manual_seed(20)
    q = (torch.empty((Z, H, N_CTX, HEAD_DIM), dtype=dtype, device=DEVICE).normal_(mean=0.0, std=0.5).requires_grad_())
    k = (torch.empty((Z, H, N_CTX, HEAD_DIM), dtype=dtype, device=DEVICE).normal_(mean=0.0, std=0.5).requires_grad_())
    v = (torch.empty((Z, H, N_CTX, HEAD_DIM), dtype=dtype, device=DEVICE).normal_(mean=0.0, std=0.5).requires_grad_())

    sm_scale = 0.5

    tri_out = attention(q, k, v, causal, sm_scale, BM, BN)
    ref_out = torch_npu.npu_fusion_attention(
            q, k, v, H,
            padding_mask=None,
            atten_mask=None,
            scale=sm_scale,
            keep_prob=1.0,
            input_layout="BNSD",
            pre_tockens=65535,
            next_tockens=65535,
            sparse_mode=0,
            )[0]

    torch.testing.assert_close(ref_out, tri_out, atol=1e-2, rtol=1e-2, equal_nan=True)
    print(f"[PASSED] Attention shape:({Z}, {H}, {N_CTX}, {HEAD_DIM}), BM: {BM}, BN: {BN}, dtype: {dtype}")


if __name__ == "__main__":
    test_op(1, 1, 128, 128, causal=False, dtype=torch.float16, BM=32, BN=128)
    test_op(1, 1, 128, 128, causal=False, dtype=torch.bfloat16, BM=64, BN=128)
    test_op(1, 2, 256, 256, causal=False, dtype=torch.bfloat16, BM=32, BN=256)
    test_op(2, 2, 128, 256, causal=False, dtype=torch.float16, BM=64, BN=128)
    test_op(4, 32, 64, 64, causal=False, dtype=torch.float16, BM=32, BN=64)
    test_op(4, 32, 1024, 64, causal=False, dtype=torch.bfloat16, BM=64, BN=128)
    test_op(4, 32, 4096, 64, causal=False, dtype=torch.float16, BM=128, BN=128)
```

Output:

```bash
[PASSED] Attention shape:(1, 1, 128, 128), BM: 32, BN: 128, dtype: torch.float16
[PASSED] Attention shape:(1, 1, 128, 128), BM: 64, BN: 128, dtype: torch.bfloat16
[PASSED] Attention shape:(1, 2, 256, 256), BM: 32, BN: 256, dtype: torch.bfloat16
[PASSED] Attention shape:(2, 2, 128, 256), BM: 64, BN: 128, dtype: torch.float16
[PASSED] Attention shape:(4, 32, 64, 64), BM: 32, BN: 64, dtype: torch.float16
[PASSED] Attention shape:(4, 32, 1024, 64), BM: 64, BN: 128, dtype: torch.bfloat16
[PASSED] Attention shape:(4, 32, 4096, 64), BM: 128, BN: 128, dtype: torch.float16
```

The preceding logs indicate that the output on Triton is the same as that on PyTorch.

## Hardware-Aware Compilation Walkthrough

This section explains how the fused-attention example is compiled for Ascend NPUs and why each compiler stage exists from a hardware point of view.

### 1. Why this kernel is a good Ascend case study

This kernel stresses the exact compiler concerns that matter on Ascend:

- it processes a long sequence dimension `N_CTX`, so the kernel must tile work instead of materializing the whole attention matrix at once,
- it uses `tl.make_block_ptr`, `tl.advance`, `tl.load`, and `tl.store`, so address generation and memory regularity matter,
- it uses `tl.where` for causal masking, which can trigger hardware-sensitive masked-access lowering,
- it uses `tl.dot`, so the compiler must preserve enough structure for vector/cube-friendly lowering,
- it carries online softmax state `m_i` and `l_i`, which avoids writing the full score matrix to global memory,
- it contains a special `HEAD_DIM >= 256` path that uses `tl.extract_slice` and `tl.insert_slice` to avoid UB overflow by chunking accumulator updates.

For Ascend hardware, these are direct responses to:

- limited on-chip memory such as UB,
- the need to keep data movement regular,
- the need to bind work to physical cores efficiently,
- the need to overlap transfer and compute when possible.

### 2. Why the kernel uses fixed-core scheduling

The kernel launches:

```python
_attn_fwd[(num_cores,)](...)
```

and then iterates:

```python
for block_idx in range(pid, NUM_BLOCKS, 20):
```

This is an Ascend-shaped scheduling style. The programming guide recommends binding the launch grid to the number of physical cores and then distributing extra logical work inside the kernel. That is different from the more common GPU mental model where programmers often over-subscribe a very large grid and rely on the GPU scheduler to manage it.

Why this is useful on Ascend:

- it reduces launch-side overhead when the logical task count is much larger than the physical core count,
- it matches the documented hardware model where vector and cube resources are limited and should be used deliberately,
- it gives the compiler and backend a more stable task shape to optimize.

### 3. Why `BLOCK_M`, `BLOCK_N`, and `HEAD_DIM` are hardware-sensitive

The block parameters are not just algorithmic tuning knobs.

- `BLOCK_M` determines how many query rows are processed per tile.
- `BLOCK_N` determines how many key/value rows are processed per tile.
- `HEAD_DIM` controls the width of each block-local vector or matrix fragment.

On Ascend, these parameters affect:

- how much GM data must be transferred into UB,
- whether multi-buffering still fits inside UB,
- whether vector or cube resources can be used efficiently,
- whether temporaries such as `q`, `k`, `v`, `p`, `pv`, and the accumulator still fit on chip.

That is why the code introduces a separate path when `HEAD_DIM >= 256`. In that case the accumulator is updated in four slices:

- this reduces peak UB demand,
- it avoids carrying an oversized live temporary through the whole block update,
- it gives later passes a chunked pattern that is easier to lower than one huge update.

### 4. Stage-by-stage lowering

The real stage pipeline is assembled in `third_party/ascend/backend/compiler.py`.

#### 4.1 Python AST to TTIR

The Triton frontend lowers the Python kernel to TTIR.

At this stage, the compiler still sees:

- Triton block pointers,
- tensor-style loads and stores,
- `tl.dot`,
- `tl.where`,
- loops over `BLOCK_N`,
- explicit softmax bookkeeping with `m_i` and `l_i`.

For fused attention, this is the last stage where the kernel still looks close to the algorithm.

#### 4.2 TTIR cleanup

The `ttir` stage runs:

- `inliner`
- `combine`
- `canonicalizer`
- `reorder_broadcast`
- `cse`
- `licm`
- `symbol_dce`
- `loop_unroll`

Why this matters here:

- repeated offset and mask expressions inside the attention loop can be deduplicated,
- loop-invariant values can move out of inner loops,
- the later hardware-aware passes receive cleaner pointer and mask structure,
- a cleaner IR reduces the chance of unnecessary temporaries increasing UB pressure.

With dump files enabled, this stage is captured as `kernel.ttir.mlir`.

#### 4.3 Ascend restructuring and adapter lowering

The `ttadapter` stage is the most important one for research. It runs:

- `auto_blockify`
- optional `dag_sync`, `dag_scope`, `dag_ssbuffer`
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

This sequence is where the kernel is turned from "Triton algorithm IR" into "Ascend-compilable IR."

How these passes matter specifically for fused attention:

- `auto_blockify`
  - helps align logical attention blocks with a more hardware-feasible block schedule.
- `dag_sync`, `dag_scope`, `dag_ssbuffer`
  - become relevant because attention naturally alternates data movement and compute across tiles; these passes prepare pipeline-friendly scheduling and buffering.
- `triton_to_structure`
  - makes block-pointer and mask patterns more regular so later lowering can reason about GM-to-UB movement.
- `discrete_mask_access_conversion`
  - matters because causal masking and `tl.where` create cases where implicit irregular masking is not a good hardware fit.
- `triton_to_unstructure`
  - acts as a fallback if some memory or index behavior cannot remain in a clean structured form.
- `triton_to_hivm`
  - moves the kernel toward an execution model that can express Ascend synchronization and lower-level execution details.
- `triton_to_hfusion`
  - helps keep elementwise chains from creating unnecessary intermediates, which is important when UB is tight.
- `bubble_up_operation`
  - is useful here because `extract_slice` and `insert_slice` can otherwise leave expensive slice traffic or avoidable loops in the wrong places.
- `triton_to_linalg`
  - converts the remaining structure into a lower-level IR that BiSheng can compile and analyze for memory planning.

With dump files enabled, this stage is captured as `kernel.ttadapter.mlir`.

#### 4.4 TTAdapter to BiSheng / NPU IR

The final stage is `npubin`.

Depending on backend mode, the compiler uses either:

- `linalg_to_bin_enable_npu_compile_A2_A3`, or
- `linalg_to_bin_enable_npu_compile_910_95`

This stage parses metadata such as:

- `mix_mode`
- `parallel_mode`
- `kernel_name`
- tensor kinds

and then passes hardware policy options into `bishengir-compile`.

For fused attention, this is where the backend makes hardware-facing choices around:

- multi-buffering,
- synchronization solver usage,
- vector-loop and cube-loop tiling policy,
- UB reporting and memory planning,
- final kernel packaging.

When `TRITON_DEBUG=1` is enabled, the backend also saves BiSheng output as `kernel.npuir.mlir`. This is the best place to inspect:

- whether synchronization-related transformations fired,
- whether memory-planning diagnostics were emitted,
- whether UB size information was reported.

### 5. Why causal masking is compiler-sensitive on Ascend

In `_attn_fwd_inner`, causal mode builds:

```python
mask = offs_m[:, None] >= (start_n + offs_n[None, :])
qk = qk * qk_scale + tl.where(mask, 0, -1.0e6)
```

From an algorithm point of view, this is masked softmax preparation.

From a compiler point of view, it is important because:

- mask shape and access regularity determine whether loads/stores remain structured,
- irregular masking is often easier to implement by moving more data into UB and selecting values there,
- explicit mask lowering can introduce additional temporaries, which increases UB pressure,
- poor handling here can reduce overlap between memory movement and compute.

This is why `discrete_mask_access_conversion` and the structured/unstructured lowering chain are especially relevant to attention kernels.

### 6. Why online softmax is a hardware win

The kernel keeps only:

- the running max `m_i`,
- the running normalization term `l_i`,
- the running output accumulator.

This avoids storing the full attention score matrix.

On Ascend hardware, that is valuable because:

- it avoids large GM traffic for temporary attention matrices,
- it keeps the working set closer to what can fit into UB-backed tiled execution,
- it makes long-sequence attention feasible without exploding on-chip memory use.

The compiler benefits too:

- fewer global intermediates means fewer bufferization and memory-planning burdens,
- later passes only need to manage tile-local state and reductions, not a full materialized score tensor.

### 7. How to study the compilation offline

For compiler research, you do not need to execute the kernel on real NPU hardware. The key is to force recompilation and save all stage artifacts.

Recommended environment variables:

```bash
export TRITON_ALWAYS_COMPILE=1
export TRITON_KERNEL_DUMP=1
export TRITON_DEBUG=1
export TRITON_DUMP_DIR=$PWD/triton_dumps
```

Then compile the kernel in a Linux environment with the Ascend toolchain configured. Preserve:

- `kernel.ttir.mlir`
- `kernel.ttadapter.mlir`
- `kernel.npuir.mlir`
- metadata JSON
- final kernel object
- any UB-related messages printed by BiSheng

For this example, ask the following questions at each stage:

1. Did pointer and mask expressions become more regular?
2. Did the loop body become easier to pipeline?
3. Did `extract_slice` and `insert_slice` remain, or were they simplified?
4. Did any debug output report UB usage or memory planning decisions?
5. Which backend options appear to control vector/cube tiling or synchronization behavior for this kernel?

### 8. What this example teaches a beginner compiler developer

The most useful lesson is that the Ascend backend is not only translating syntax. It is negotiating between:

- the high-level structure of the Triton algorithm,
- the limited size of UB and other on-chip resources,
- the need for regular memory movement,
- the desire to overlap transfer and compute,
- the need to coordinate vector-style and cube-style execution resources.

That is why fused attention is such a good research kernel. Its source code already contains the clues:

- fixed-core scheduling,
- explicit tiling,
- online reductions,
- careful accumulator management,
- masking and block pointers.

The compiler's job is to preserve the parts of that structure that help Ascend hardware, and rewrite the parts that do not.
