# NablaTensor to quant 0.3.0

The finance project is now **NablaTensor Quant**, hosted at
https://github.com/nablatensor-dev/quant. Maven groupId remains `com.nablatensor`.
The quant parent and modules use version `0.3.0`; their JAX4J dependency is pinned
to `0.1.0` independently.

## Ownership and API changes

Quant owns finance: markets, models, products, calibration, curves, scenarios,
risk and CVA. These classes use `com.nablatensor`. JAX4J owns tensor operations,
autodiff, scalar recording/replay, JIT and backends under `com.jax4j`.

The earlier local migration depended on `jax4j-neo-finance`. Current JAX4J removed
that module, so its 33 finance classes now belong in `quant-pricing`. They retain
the Apache headers from the source lineage. Tensor/runtime implementations remain
shared through JAX4J; no local backend modules are restored.

Update scalar API usages: `ADouble` becomes `SDouble`; `TypedPricer` becomes
`TypedEvaluator`; `TypedValuation` becomes `TypedEvaluation`; `priceOnly()` becomes
`valueOnly()`; `greeks()`/`greek(...)` become `gradients()`/`gradient(...)`; and
JAX4J evaluation results expose `value()`. Quant analytic price APIs retain
`price()`. Shared markets use record constructors and `OptionType` replaces
`OptionTypeEnum`. Named builders remain for quant's additional configuration types.

These changes are source and binary incompatible. Do not combine the old
NablaTensor JARs with quant/JAX4J JARs. Maven relocation identifies the destination;
it does not translate source imports or preserve old classes.

## Retired coordinates

The old published releases are immutable and remain available. Version `0.3.0`
under each retired artifact name is a POM-only migration notice. For relocations,
Maven emits the supplied migration message and resolves the replacement version.
Consumers pinned to older versions will need to update explicitly to see it.

| Old artifactId | Replacement coordinates, group `com.nablatensor` |
| --- | --- |
| `nablatensor-backend-cuda` | `jax4j-engine-cuda:0.1.0` |
| `nablatensor-backend-rocm` | `jax4j-engine-rocm:0.1.0` |
| `nablatensor-backend-vulkan` | `jax4j-engine-vulkan:0.1.0` |
| `nablatensor-core` | `jax4j-core:0.1.0` |
| `nablatensor-cpu` | `jax4j-core:0.1.0` |
| `nablatensor-credit` | `quant-pricing:0.3.0` |
| `nablatensor-cuda` | `jax4j-engine-cuda:0.1.0` |
| `nablatensor-cva` | `quant-cva:0.3.0` |
| `nablatensor-engine-cpu` | `jax4j-core:0.1.0` |
| `nablatensor-engine-cuda` | `jax4j-engine-cuda:0.1.0` |
| `nablatensor-engine-rocm` | `jax4j-engine-rocm:0.1.0` |
| `nablatensor-engine-simd` | `jax4j-simd:0.1.0` |
| `nablatensor-engine-vulkan` | `jax4j-engine-vulkan:0.1.0` |
| `nablatensor-jit-cpu` | `jax4j-core:0.1.0` |
| `nablatensor-lattice` | `quant-pricing:0.3.0` |
| `nablatensor-ops` | `jax4j-core:0.1.0` |
| `nablatensor-parent` | `quant:0.3.0` |
| `nablatensor-quant` | `quant-pricing:0.3.0` |
| `nablatensor-risk` | `quant-risk:0.3.0` |
| `nablatensor-rocm` | `jax4j-engine-rocm:0.1.0` |
| `nablatensor-scenario` | `quant-pricing:0.3.0` |
| `nablatensor-simd` | `jax4j-simd:0.1.0` |
| `nablatensor-tensor` | `jax4j-core:0.1.0` |
| `nablatensor-vulkan` | `jax4j-engine-vulkan:0.1.0` |
| `nablatensor-engine-opencl` | Retired; OpenCL replay is retired and has no supported replacement. Use quant with the JAX4J Vulkan or wgpu tensor backend; this requires an API migration. |
| `nablatensor-validate` | Retired; The old validation harness is retired. See quant/docs/validation.md for Vulkan and wgpu validation; there is no replacement library artifact. |

The OpenCL and old validation notices deliberately have no relocation target or
JAR: they explain the move without falsely redirecting callers to an unrelated
API. Upgrade those usages manually using this guide.

## Backend limits

The active validation is Vulkan and wgpu only. Scalar Vulkan replay supports F32
and one output; wgpu currently supports tensor execution and tensor autodiff,
with no scalar AAD adapter. Some retained finance paths require FP64 or
multi-output scalar execution. `ExposureSimulation.run` may use CPU replay for
exposure profiles; the active suite checks only `cvaOnly` pinned to Vulkan.

No claim of CUDA, ROCm, SIMD, OpenCL or complete legacy finance GPU coverage is
made by this release. See [validation](validation.md) for exact checks.
