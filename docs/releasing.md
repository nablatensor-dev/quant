# Publishing quant and migration notices

Quant versions and `jax4j.version` are independent. Released quant builds must
use released JAX4J dependencies. Never overwrite a Central version or move an
existing release tag.

For 0.3.0, the publication consists of:

- JAX4J 0.1.0: parent, core, Vulkan, wgpu, CUDA, ROCm and SIMD artifacts. Main JARs
  match the existing GitHub v0.1.0 assets; sources are built from that tag. The
  publishing POM adds Central's developer/SCM metadata without moving the tag.
- quant 0.3.0: parent, codegen, pricing, risk and CVA, plus sources and Javadocs.
- 24 standalone legacy relocation POMs and two retirement notices from
  `release/retired-artifacts.json`. Examples are not published as libraries.

Validate only Vulkan and wgpu. Optional JAX4J CUDA/ROCm/SIMD modules can compile
without running their hardware; compilation is not evidence of backend behavior.

Build quant artifacts with `mvn -Pcentral-artifacts verify` and validate the
second backend with `mvn -Pwgpu verify`. The `central-artifacts` profile creates
source/Javadoc JARs and GPG signatures without uploading. An unlocked local GPG
signing key is required. Keep Maven publisher credentials outside the repository
in the `central` server entry of `~/.m2/settings.xml`.

After preparing the matching signed JAX4J tag checkout, assemble a fresh bundle:

```sh
python3 scripts/prepare-central-bundle.py \
  --jax4j-tree /path/to/prepared-jax4j-tag-checkout \
  --output /tmp/quant-central-release
```

The assembler checks signatures, release versions and personal-path exposure.
It writes a manifest, Maven-layout repository and `central-bundle.zip`. Review
the manifest and artifact mapping before uploading.

```sh
python3 scripts/central-publisher.py upload \
  --bundle /tmp/quant-central-release/central-bundle.zip \
  --name 'JAX4J 0.1.0, quant 0.3.0 and NablaTensor migration' \
  --record /tmp/quant-central-release/deployment-id.txt
python3 scripts/central-publisher.py status --deployment DEPLOYMENT_UUID
python3 scripts/central-publisher.py publish --deployment DEPLOYMENT_UUID
```

Upload uses USER_MANAGED so validation completes before the explicit publish
command. Credentials are read inside the process and are never command-line
arguments. Record the deployment ID and successful state. Verify public POM/JAR
URLs and checksums after Central synchronization.

Commit the reviewed quant release source, tag it `v0.3.0`, and push to the renamed
`nablatensor-dev/quant` repository. Create a GitHub release linking the migration
and validation records. Existing `nablatensor` releases/tags stay preserved.
