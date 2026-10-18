package com.nablatensor.quant;

import static org.junit.jupiter.api.Assertions.*;
import com.nablatensor.quant.BlackScholes;
import com.nablatensor.quant.EquityMarket;
import com.nablatensor.quant.OptionType;
import com.jax4j.tensor.*;
import com.nablatensor.quant.analytic.Normal;
import org.junit.jupiter.api.Test;

/** Runs real device valuation and differentiation; missing hardware is a failure. */
class TensorGpuValidationTest {
  private static Device device() {
    return switch (System.getProperty("quant.backend", "vulkan")) {
      case "vulkan" -> Device.vulkan();
      case "wgpu" -> Device.wgpu();
      default -> throw new IllegalArgumentException("quant.backend must be vulkan or wgpu");
    };
  }

  private static Tensor array(float[] values, Device device) {
    return Tensors.arrayOn(values, Shape.of(values.length), device);
  }

  private static float[] normals(int n) {
    float[] values = new float[n];
    for (int i = 0; i < n; i++) values[i] = (float) Normal.inverseCdf((i + 0.5) / n);
    return values;
  }

  @Test
  void compiledPricesAndAllGreeksMatchIndependentPathwiseReference() {
    Device device = device();
    float[] z = normals(4097);
    try (Tensor draws = array(z, device)) {
      for (OptionType type : OptionType.values()) {
        var compiled = Jit.compileTree((Tensor market) ->
            Transforms.valueAndGrad(m -> TensorEuropean.price(m, draws, type), market));
        for (float spot : new float[] {90, 100, 112}) {
          float[] parameters = {spot, 100, 0.2f, 0.03f, 1};
          try (Tensor market = array(parameters, device)) {
            var result = compiled.apply(market);
            try (Tensor price = result.value(); Tensor greeks = result.grad()) {
              assertEquals(device, price.device());
              assertEquals(reference(parameters, z, type), price.toFloatArray()[0], 4e-4);
              float[] gradient = greeks.toFloatArray();
              for (int i = 0; i < parameters.length; i++) {
                float[] up = parameters.clone(), down = parameters.clone();
                float bump = i < 2 ? 0.01f : 0.0001f;
                up[i] += bump; down[i] -= bump;
                double expected = (reference(up, z, type) - reference(down, z, type)) / (up[i] - down[i]);
                assertEquals(expected, gradient[i], 0.02 + Math.abs(expected) * 0.002,
                    type + " spot=" + spot + " component=" + i);
              }
            }
          }
        }
        assertEquals(1, compiled.cachedPlanCount());
      }
    }
  }

  @Test
  void pricesAndGreeksAgreeWithBlackScholes() {
    Device device = device();
    float[] parameters = {100, 100, 0.2f, 0.03f, 1};
    var market = new EquityMarket(100, 100, 0.2, 0.03, 1);
    try (Tensor draws = array(normals(65_536), device);
        Tensor input = array(parameters, device)) {
      for (OptionType type : OptionType.values()) {
        var compiled = Jit.compileTree((Tensor m) ->
            Transforms.valueAndGrad(x -> TensorEuropean.price(x, draws, type), m));
        var result = compiled.apply(input);
        var expected = BlackScholes.of(type, market);
        try (Tensor price = result.value(); Tensor gradient = result.grad()) {
          assertEquals(expected.price(), price.toFloatArray()[0], 0.004);
          float[] g = gradient.toFloatArray();
          assertEquals(expected.delta(), g[0], 0.001);
          assertEquals(expected.strikeSensitivity(), g[1], 0.001);
          assertEquals(expected.vega(), g[2], 0.02);
          assertEquals(expected.rho(), g[3], 0.02);
        }
      }
    }
  }

  private static double reference(float[] p, float[] draws, OptionType type) {
    double sum = 0;
    for (float z : draws) {
      double terminal = p[0] * Math.exp((p[3] - 0.5 * p[2] * p[2]) * p[4]
          + p[2] * Math.sqrt(p[4]) * z);
      sum += Math.max(type.sign() * (terminal - p[1]), 0);
    }
    return Math.exp(-p[3] * p[4]) * sum / draws.length;
  }
}
