package com.nablatensor.quant;

import com.nablatensor.quant.OptionType;
import com.jax4j.tensor.Shape;
import com.jax4j.tensor.Tensor;

/** European GBM valuation composed from JAX4J tensor operations on Vulkan or wgpu. */
public final class TensorEuropean {
  private TensorEuropean() {}

  /**
   * Discounted mean payoff using caller-supplied standard-normal draws.
   * The market vector is [spot, strike, vol, rate, maturity]; normals is a nonempty
   * vector on the same device. Use inside JAX4J Jit and Transforms to compile
   * valuation and obtain all five pathwise sensitivities. Inputs remain caller-owned.
   */
  public static Tensor price(Tensor market, Tensor normals, OptionType type) {
    if (!market.shape().equals(Shape.of(5))) {
      throw new IllegalArgumentException("market must have shape [5]");
    }
    if (normals.shape().rank() != 1 || normals.shape().size() == 0) {
      throw new IllegalArgumentException("normals must be a nonempty vector");
    }
    if (!market.device().equals(normals.device())) {
      throw new IllegalArgumentException("market and normals must use the same device");
    }
    Tensor spot = market.sliceAxis0(0);
    Tensor strike = market.sliceAxis0(1);
    Tensor vol = market.sliceAxis0(2);
    Tensor rate = market.sliceAxis0(3);
    Tensor maturity = market.sliceAxis0(4);
    Tensor logReturn = rate.sub(vol.mul(vol).mul(0.5)).mul(maturity)
        .add(normals.mul(vol).mul(maturity.sqrt()));
    return spot.mul(logReturn.exp()).sub(strike).mul(type.sign()).relu().mean()
        .mul(rate.mul(maturity).neg().exp());
  }
}
