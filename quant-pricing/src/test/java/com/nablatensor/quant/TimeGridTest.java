/*
 * Copyright 2026 The NablaTensor Authors
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     https://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package com.nablatensor.quant;

import com.nablatensor.quant.OptionType;
import com.nablatensor.quant.Products;
import com.nablatensor.quant.BlackScholes;
import com.nablatensor.quant.MonteCarlo;
import com.nablatensor.quant.EquityMarket;
import com.nablatensor.quant.TimeGrid;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;
import com.jax4j.engine.Nabla;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

/**
 * {@link TimeGrid}: a uniform grid must reproduce the earlier {@code steps(int)}
 * arithmetic bit for bit, and an equal-gap explicit grid must price the same as
 * the uniform one. A genuinely non-uniform grid still prices a European (whose
 * value depends only on the terminal marginal) against Black-Scholes.
 */
@Tag("mc")
class TimeGridTest {

  private static final EquityMarket M = EquityMarket.atmOneYear();
  private static final long N = 400_000L;
  private static final long SEED = 77L;

  @Test
  void uniformGridFractionsSumToOne() {
    TimeGrid g = TimeGrid.uniform(12);
    double s = 0.0;
    for (int i = 0; i < g.steps(); i++) {
      s += g.fraction(i);
    }
    assertEquals(1.0, s, 1e-12);
    assertTrue(g.isUniform());
    assertEquals(1.0, g.cumulative(11), 1e-12);
  }

  @Test
  void uniformTimeGridEqualsStepsIntArithmeticBitForBit() {
    try (MonteCarlo<EquityMarket> viaSteps = MonteCarlo.of(Products.asianCall())
             .market(M).steps(64).fp64().gradients().on("cpu-jit").build();
         MonteCarlo<EquityMarket> viaGrid = MonteCarlo.of(Products.asianCall())
             .market(M).timeGrid(TimeGrid.uniform(64)).fp64().gradients().on("cpu-jit").build()) {

      Nabla.TypedEvaluation<EquityMarket> a = viaSteps.run(N, SEED);
      Nabla.TypedEvaluation<EquityMarket> b = viaGrid.run(N, SEED);
      assertEquals(a.value(), b.value(), 0.0, "uniform TimeGrid price is bit-identical to steps(int)");
      assertEquals(a.gradient(EquityMarket::spot), b.gradient(EquityMarket::spot), 0.0, "delta bit-identical");
      assertEquals(a.gradient(EquityMarket::vol), b.gradient(EquityMarket::vol), 0.0, "vega bit-identical");
    }
  }

  @Test
  void equalGapExplicitGridMatchesUniform() {
    double[] fixings = new double[8];
    for (int i = 0; i < 8; i++) {
      fixings[i] = (i + 1) * (1.0 / 8);
    }
    try (MonteCarlo<EquityMarket> uniform = MonteCarlo.of(Products.asianCall())
             .market(M).steps(8).fp64().valueOnly().on("cpu-jit").build();
         MonteCarlo<EquityMarket> explicit = MonteCarlo.of(Products.asianCall())
             .market(M).timeGrid(TimeGrid.of(fixings)).fp64().valueOnly().on("cpu-jit").build()) {
      assertEquals(uniform.run(N, SEED).value(), explicit.run(N, SEED).value(), 1e-9,
          "equal-gap explicit grid == uniform grid");
    }
  }

  @Test
  void nonUniformGridPricesAEuropeanAgainstBlackScholes() {
    // front-loaded schedule: the terminal marginal is unchanged, so a European still matches BS
    TimeGrid skewed = TimeGrid.of(0.1, 0.15, 0.25, 0.45, 0.7, 1.0);
    assertTrue(!skewed.isUniform());
    BlackScholes ref = BlackScholes.of(OptionType.CALL, M);
    try (MonteCarlo<EquityMarket> mc = MonteCarlo.of(Products.europeanCall())
        .market(M).timeGrid(skewed).fp64().gradients().on("cpu-jit").build()) {
      Nabla.TypedEvaluation<EquityMarket> p = mc.run(1_000_000L, SEED);
      assertEquals(ref.price(), p.value(), 0.15, "non-uniform-grid European price vs Black-Scholes");
      assertEquals(ref.delta(), p.gradient(EquityMarket::spot), 0.02, "non-uniform-grid European delta vs Black-Scholes");
    }
  }
}
