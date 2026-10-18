# NablaTensor Quant

Quantitative finance, risk and CVA on Java 25, built on
[JAX4J](https://github.com/nablatensor-dev/jax4j).

The Maven group is `com.nablatensor`. Quant **0.3.0** uses JAX4J **0.1.0**.
JAX4J supplies tensors, automatic differentiation, JIT and compute backends;
quant owns the finance models, products, calibration, risk and CVA APIs.

| Artifact | Responsibility |
| --- | --- |
| `quant` | Maven parent POM |
| `quant-pricing` | Products, Monte Carlo, models, calibration, curves, analytic oracles, transforms, lattice, credit and scenarios |
| `quant-risk` | Risk factors, aggregation, VaR/ES and backtesting |
| `quant-cva` | Exposure simulation, CVA, BA-CVA and SA-CVA |
| `quant-codegen` | Named construction of finance configuration classes and records |

`quant-examples` contains worked examples and is built from source rather than
published as a library.

## Maven

```xml
<dependency>
  <groupId>com.nablatensor</groupId>
  <artifactId>quant-pricing</artifactId>
  <version>0.3.0</version>
</dependency>
<dependency>
  <groupId>com.nablatensor</groupId>
  <artifactId>jax4j-engine-vulkan</artifactId>
  <version>0.1.0</version>
  <scope>runtime</scope>
</dependency>
```

Add `quant-risk` or `quant-cva` at the same quant version as needed. Add
`jax4j-engine-wgpu:0.1.0` for tensor valuation through wgpu-native.

## Build and GPU validation

Requires JDK 25 and Maven 3.9+.

```sh
mvn -DskipTests package    # compile and package, including retained test sources
mvn clean verify          # Vulkan validation; requires native libraries and hardware
mvn -Pwgpu verify         # wgpu tensor validation
```

Only `*GpuValidationTest` classes execute in the active suite. Requested but
unavailable devices fail validation. Other retained finance tests compile but
are outside the current GPU validation. See [validation](docs/validation.md).

## European valuation

```java
import com.nablatensor.quant.*;

try (var mc = MonteCarlo.of(Products.europeanCall())
    .market(EquityMarket.atmOneYear()).steps(1)
    .fp32().gradients().on("vulkan").build()) {
  var result = mc.run(262_144, 42L);
  System.out.println(result.value());
  System.out.println(result.gradient(EquityMarket::spot));
}
```

Finance types live under `com.nablatensor`; compute APIs live under `com.jax4j`.
`TensorEuropean.price` composes JAX4J tensor operations and supports both Vulkan
and wgpu through `Jit` and `Transforms.valueAndGrad`.

## Migration from NablaTensor

The GitHub repository is now `nablatensor-dev/quant`. Existing NablaTensor
releases remain available. New `0.3.0` migration POMs under the retired artifact
names identify their replacements and link to the migration guide.

This is a breaking API migration. Some legacy artifacts move to quant; general
compute artifacts move to JAX4J. OpenCL and the old validation harness have
retirement notices and no direct replacement library. See
[the migration guide and artifact map](docs/migration.md).

Historical notebook/Python/container/demo tooling targets the old layout and is
not part of the validated release. Release maintainers: [publishing](docs/releasing.md).
