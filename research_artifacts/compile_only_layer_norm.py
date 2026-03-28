"""Compile-only harness for layer-norm kernel — no NPU hardware needed.

Usage:
    export TRITON_ASCEND_ARCH=Ascend910_9599
    export TRITON_ALWAYS_COMPILE=1
    export TRITON_COMPILE_ONLY=1
    export TRITON_KERNEL_DUMP=1
    export TRITON_DEBUG=1
    export TRITON_DUMP_DIR=/path/to/dump

    python research_artifacts/compile_only_layer_norm.py
"""

import os
import sys

# Ensure dump dir exists
dump_dir = os.environ.get("TRITON_DUMP_DIR", "/tmp/triton_ir_dump")
os.makedirs(dump_dir, exist_ok=True)

import triton
import triton.language as tl
from triton.compiler import ASTSource
from triton.backends.compiler import GPUTarget


@triton.jit
def _layer_norm_fwd_fused(
    X,
    Y,
    W,
    B,
    Mean,
    Rstd,
    stride,
    N,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    Y += row * stride
    X += row * stride
    mean = 0
    _mean = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        a = tl.load(X + cols, mask=cols < N, other=0.).to(tl.float32)
        _mean += a
    mean = tl.sum(_mean, axis=0) / N
    _var = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        x = tl.load(X + cols, mask=cols < N, other=0.).to(tl.float32)
        x = tl.where(cols < N, x - mean, 0.)
        _var += x * x
    var = tl.sum(_var, axis=0) / N
    rstd = 1 / tl.sqrt(var + eps)
    tl.store(Mean + row, mean)
    tl.store(Rstd + row, rstd)
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        w = tl.load(W + cols, mask=mask)
        b = tl.load(B + cols, mask=mask)
        x = tl.load(X + cols, mask=mask, other=0.).to(tl.float32)
        x_hat = (x - mean) * rstd
        y = x_hat * w + b
        tl.store(Y + cols, y, mask=mask)


def main():
    arch = os.environ.get("TRITON_ASCEND_ARCH", "Ascend910_9599")
    target = GPUTarget("npu", arch, 0)

    signature = {
        "X": "*fp16",
        "Y": "*fp16",
        "W": "*fp16",
        "B": "*fp16",
        "Mean": "*fp32",
        "Rstd": "*fp32",
        "stride": "i32",
        "N": "i32",
        "eps": "fp32",
    }
    constants = {"BLOCK_SIZE": 128}

    src = ASTSource(
        fn=_layer_norm_fwd_fused,
        constants=constants,
        signature=signature,
    )

    print(f"Compiling layer-norm kernel for {arch}...")
    print(f"Dump dir: {dump_dir}")

    kernel = triton.compile(
        src,
        target=target,
        options={"debug": True, "num_warps": 1, "num_ctas": 1},
    )

    print("Compilation completed successfully!")
    print(f"Check IR dumps in: {dump_dir}")

    # List dump contents
    for root, dirs, files in os.walk(dump_dir):
        for f in sorted(files):
            path = os.path.join(root, f)
            size = os.path.getsize(path)
            rel = os.path.relpath(path, dump_dir)
            print(f"  {rel} ({size} bytes)")


if __name__ == "__main__":
    main()
