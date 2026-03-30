# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**This is a personal research fork** of the official [Triton-Ascend](https://gitee.com/ascend/triton-ascend) project. The primary goal is to study the Ascend compiler pipeline and implement a compile-only mode for dumping IR at each pass stage — no NPU hardware required.

Triton-Ascend adapts the [Triton](https://github.com/openai/triton) compiler stack to run on Huawei Ascend NPUs. It compiles Triton Python IR through MLIR lowering passes (Triton → Linalg → HiVM → BiShengIR) into kernels executable on Atlas A2/A3 hardware via the CANN toolkit (8.5.0). The project carries the full upstream Triton tree plus an Ascend-specific backend under `third_party/ascend/`.

### Compile-Only IR Dumping

The fork's main addition is a compile-only workflow for IR research — **no NPU hardware required**. When `TRITON_COMPILE_ONLY=1` is set, the Ascend backend automatically mocks NPU device operations (tensor allocation, streams, device queries) so that unmodified Triton kernel scripts run the full MLIR compilation pipeline and dump IR, while kernel launch is skipped.

- Set environment variables and run any tutorial directly:
  ```bash
  export TRITON_ASCEND_ARCH=Ascend910_9599
  export TRITON_ALWAYS_COMPILE=1
  export TRITON_COMPILE_ONLY=1
  export TRITON_KERNEL_DUMP=1
  export TRITON_DEBUG=1
  export TRITON_DUMP_DIR=/path/to/dump/dir

  python third_party/ascend/tutorials/01-vector-add.py
  ```
- Produces per-stage IR files: `kernel.ttir.mlir`, `kernel.ttadapter.mlir`, `kernel.npuir.mlir`.
- The mock layer (`third_party/ascend/backend/compile_only_mock.py`) redirects `device='npu'` tensor ops to CPU and stubs out driver hardware queries. No code changes needed in kernel scripts.
- `torch.allclose` validation at the end of tutorials will fail (expected — kernel didn't execute, output tensors are uninitialized).

## Build Commands

All build/test commands assume a Linux environment with CANN installed.

```bash
# Full wheel build (from repo root)
cd python && \
  TRITON_BUILD_WITH_CLANG_LLD=true \
  TRITON_BUILD_PROTON=OFF \
  TRITON_WHEEL_NAME="triton-ascend" \
  MAX_JOBS=$(nproc) \
  python3 setup.py bdist_wheel

# Incremental rebuild (after initial build)
make all

# Build wheel via Makefile
make package
```

Build requires: CMake >= 3.18, Ninja >= 1.12.0, pybind11 >= 2.13.1, Clang/LLD.

## Testing

```bash
# All unit tests (parallel via pytest-xdist)
make test-unit

# Individual test suites
cd third_party/ascend/unittest/pytest_ut && python3 -m pytest -s -v -n auto --dist=loadfile
cd third_party/ascend/unittest/autotune_ut && python3 -m pytest -s -v -n auto --dist=loadfile
cd third_party/ascend/unittest/kernels && python3 -m pytest -s -v test_triton_kernel.py

# Inductor integration tests
make test-inductor

# Generalization tests
make test-gen

# Run a single test file or test
cd third_party/ascend/unittest/pytest_ut && python3 -m pytest -s -v path/to/test_file.py::test_name
```

## Linting and Formatting

Pre-commit hooks enforce formatting. Run against your changes before committing:

```bash
pre-commit run --from-ref origin/main --to-ref HEAD
```

- **Python**: ruff (line-length 120, ignores E501/E701/E731/E741) + yapf (PEP8, column limit 120)
- **C++**: clang-format v16.0.6

## Architecture

### Compilation Pipeline

Triton Python code → `triton.language` API → Triton MLIR dialects → backend-specific lowering:

1. **Frontend** (`python/triton/language/`): Triton Python DSL — `core.py` defines tensor ops, `math.py` provides math intrinsics
2. **Compiler** (`python/triton/compiler/`): Orchestrates compilation stages, MLIR pass pipeline
3. **Ascend Backend** (`third_party/ascend/backend/`):
   - `compiler.py` — Registers Ascend-specific MLIR lowering passes (TritonToLinalg, TritonToHFusion, TritonToHIVM, TritonToAnnotation)
   - `driver.py` — Ascend device management and kernel launch via CANN runtime
4. **C++ passes** (`third_party/ascend/lib/`): MLIR conversion pass implementations (Triton→Linalg, HFusion, HIVM, Annotation)
5. **AscendNPU-IR** (`third_party/ascend/AscendNPU-IR/`): Git submodule — BiShengIR compiler that produces final device binaries

### Key Directories

| Path | Purpose |
|---|---|
| `python/triton/` | Core Triton package (language, compiler, runtime) |
| `third_party/ascend/backend/` | Ascend compiler & driver (Python) |
| `third_party/ascend/lib/` | Ascend MLIR conversion passes (C++) |
| `third_party/ascend/include/` | Ascend pass headers (C++) |
| `third_party/ascend/language/` | Ascend-specific language extensions and CANN bindings |
| `third_party/ascend/unittest/` | All Ascend test suites |
| `third_party/ascend/tutorials/` | Example Triton kernels for Ascend |
| `lib/`, `include/` | Core Triton MLIR dialects and analyses (C++) |
| `test/` | MLIR lit tests (FileCheck-based) |

### Runtime Flow

`python/triton/runtime/jit.py` handles JIT compilation and caching. When a `@triton.jit` kernel is called, the runtime selects the active backend (Ascend via `third_party/ascend/backend/backend_register.py`), compiles through the MLIR pipeline, and launches via the Ascend driver.

## PR and CI Conventions

- Fork-and-pull workflow; PRs need 2+ LGTM approvals
- Comment `/compile` on a PR to trigger CI
- GitHub Actions CI: `.github/workflows/integration-tests.yml` (auto-generated from `.yml.in` — edit the `.in` file, not the `.yml`)
- PR template checklist in `.github/PULL_REQUEST_TEMPLATE.md`
