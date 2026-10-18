# Contributing to quant

Use JDK 25 and Maven 3.9+. Install JAX4J as described in [README](README.md).
Keep tensor operations, autodiff, JIT and device runtimes in JAX4J. Add finance
behavior here when JAX4J does not already provide it, and import finance types from quant and compute types from
JAX4J.

Validate with `mvn clean verify` and `mvn -Pwgpu verify` on configured hardware.
The active tests explicitly select Vulkan or wgpu and fail when the selected
device is unavailable. Use closed forms or independent host calculations as
numerical references. Keep the backend validation matrix limited to these two.

Describe API changes and numerical tolerances in the PR. Performance claims
should identify workload, device, runtime, seed and synchronization.
