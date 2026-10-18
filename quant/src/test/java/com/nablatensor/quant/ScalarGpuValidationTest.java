package com.nablatensor.quant;

import static org.junit.jupiter.api.Assertions.*;
import com.nablatensor.quant.*;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.condition.EnabledIfSystemProperty;

/** Validates the reused scalar finance API on its supported GPU backend. */
@EnabledIfSystemProperty(named = "quant.backend", matches = "vulkan")
class ScalarGpuValidationTest {
  @Test
  void priceAndAdjointMatchAnalyticAndBumpedValuations() {
    EquityMarket market = EquityMarket.atmOneYear();
    try (var mc = MonteCarlo.of(Products.europeanCall()).market(market)
        .steps(1).fp32().gradients().on("vulkan").build()) {
      var value = mc.run(262_144, 42L);
      var analytic = BlackScholes.of(OptionType.CALL, market);
      assertEquals("vulkan", mc.engine());
      assertEquals(analytic.price(), value.value(), 0.2);
      assertEquals(analytic.delta(), value.gradient(EquityMarket::spot), 0.01);
      assertEquals(analytic.vega(), value.gradient(EquityMarket::vol), 1);
      double bump = (mc.run(market.withSpot(100.1), 262_144, 42L).value()
          - mc.run(market.withSpot(99.9), 262_144, 42L).value()) / 0.2;
      assertEquals(bump, value.gradient(EquityMarket::spot), 0.003);
    }
  }
}
