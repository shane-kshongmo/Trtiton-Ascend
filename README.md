# Triton-Ascend (Compile-Only Research Fork)

This is a **personal research fork** of [Triton-Ascend](https://gitee.com/ascend/triton-ascend) for studying the Ascend compiler pipeline. The primary goal is to dump IR at each compilation pass stage -- **no NPU hardware required**.

## What This Fork Adds

When `TRITON_COMPILE_ONLY=1` is set, the Ascend backend automatically mocks NPU device operations (tensor allocation, streams, device queries) so that **unmodified** Triton kernel scripts run the full MLIR compilation pipeline and dump IR, while kernel launch is skipped.

```bash
export TRITON_ASCEND_ARCH=Ascend910_9599
export TRITON_ALWAYS_COMPILE=1
export TRITON_COMPILE_ONLY=1
export TRITON_KERNEL_DUMP=1
export TRITON_DEBUG=1
export TRITON_DUMP_DIR=/path/to/dump/dir

python third_party/ascend/tutorials/01-vector-add.py
```

This produces per-stage IR files:
- `kernel.ttir.mlir` -- Triton dialect (high-level tensor ops)
- `kernel.ttadapter.mlir` -- Linalg/memref dialect (buffer semantics)
- `kernel.npuir.mlir` -- BiShengIR (NPU binary IR)
- `npuir_passes/` -- per-pass AscendNPU-IR dumps

No code changes needed in kernel scripts. The `torch.allclose` validation at the end of tutorials will fail (expected -- kernel didn't execute, output tensors are uninitialized).

## Compilation Pipeline

Triton Python code is compiled through MLIR lowering passes:

1. **Triton IR (TTIR)** -- high-level tensor operations from `@triton.jit` kernels
2. **TTAdapter** -- Ascend-specific lowering: Triton -> Linalg/HFusion/HIVM/Annotation
3. **NPUIR** -- BiShengIR binary generation via `bishengir-compile`

## Getting Started

- [Quickstart IR Dump](./research_artifacts/QUICKSTART_IR_DUMP.md) -- fastest path to dumping IR
- [Build and Dump IR Guide](./docs/en/build_and_dump_ir.md) -- full build instructions for WSL
- [Environment Variables](./docs/en/environment_variable_reference.md)

## Prerequisites

| Component | Version |
|---|---|
| Python | 3.10+ |
| CANN toolkit | 8.5.0 |
| PyTorch + torch_npu | 2.7.x |
| Ninja | >= 1.12.0 |
| CMake | >= 3.18 |
| Clang/LLD | for building from source |

## Build from Source

```bash
cd python
TRITON_BUILD_WITH_CLANG_LLD=true \
TRITON_BUILD_PROTON=OFF \
TRITON_WHEEL_NAME="triton-ascend" \
MAX_JOBS=$(nproc) \
  python3 setup.py bdist_wheel

pip install dist/triton_ascend-*.whl --force-reinstall --no-deps
```

## Upstream Documentation

For the full Triton-Ascend documentation (hardware support, performance benchmarks, operator development), see the [upstream project](https://gitee.com/ascend/triton-ascend).

- [Architecture Design and Core Features](./docs/en/architecture_design_and_core_features.md)
- [Compiler Structure and Ascend Hardware Model](./docs/en/compiler_structure_and_hardware.md)
- [Compiler Pass Guide with Ascend Hardware Rationale](./docs/en/compiler_passes_and_hardware.md)
- [Operator Development Guide](./docs/en/programming_guide.md)

## License

[MIT License](./LICENSE)
