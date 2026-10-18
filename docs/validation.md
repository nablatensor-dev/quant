# GPU validation

Install JAX4J as described in [README](../README.md). Use JDK 25 and Maven 3.9+.
The test JVM enables FFM native access.

```sh
mvn clean verify        # Vulkan
mvn -Pwgpu verify       # wgpu-native
```

Only `*GpuValidationTest` classes run. Older finance tests compile but do not run
under these commands. Explicit unavailable-device requests fail; tensor tests do
not fall back to CPU.

## Prerequisites

Vulkan needs a working loader, compute driver and shaderc. wgpu needs JAX4J's
pinned wgpu-native ABI. See [upstream wgpu setup](../../jax4j/docs/wgpu.md).
Set `JAX4J_WGPU_NATIVE_PATH` to the installed `libwgpu_native.so`, or pass
`-Djax4j.wgpu.native_path=/absolute/path/libwgpu_native.so`.

## Coverage

| Check | Vulkan | wgpu |
| --- | --- | --- |
| Compiled tensor call/put prices, changed inputs and all five derivatives against independent double arithmetic and finite differences | yes | yes |
| Tensor prices and Greeks against Black-Scholes with stratified normal quantiles | yes | yes |
| Reused scalar Monte Carlo price/Greeks against Black-Scholes and common-seed spot bumps | yes | upstream unsupported; skipped |
| CVA positivity and loss-given-default scaling, explicitly pinned to Vulkan | yes | upstream scalar replay unsupported; skipped |

Tensor arithmetic is F32. Tolerances account for rounding, payoff kinks and
quadrature or sampling error. Upstream scalar Vulkan does not report sampling
standard error, so its analytic price check uses an explicit tolerance. The
independent host references do not execute a CPU compute backend.

CVA exposure profiles, scalar multi-output replay and retained FP64 legacy tests
are outside this validation. No CPU, SIMD, CUDA, ROCm or OpenCL backend tests run.

GPU release validation runs on configured hardware using the commands above.
The Central publishing process is described in [releasing](releasing.md).
Release evidence is recorded in [validation-record.json](validation-record.json).
