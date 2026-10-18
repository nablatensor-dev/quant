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
import com.nablatensor.quant.Product;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;
import com.jax4j.engine.Nabla;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

/**
 * MVP correctness: the recorded valuations price sensibly, the adjoint Greeks
 * agree with an independent reference, and the bytecode kernel reproduces the
 * scalar oracle path-for-path.
 */
@Tag("mc")
class MvpPricingTest {

  private static final long SCENARIOS = 400_000L;
  private static final long SEED = 20260830L;
  private static final int STEPS = 64;

  @Test
  void europeanCallAdjointGreeksMatchBlackScholes() {
    EquityMarket m = EquityMarket.atmOneYear();
    BlackScholes ref = BlackScholes.of(OptionType.CALL, m);

    try (MonteCarlo<EquityMarket> mc = MonteCarlo.of(Products.europeanCall())
        .market(m).steps(1).fp64().gradients().on("cpu-jit").build()) {

      Nabla.TypedEvaluation<EquityMarket> p = mc.run(SCENARIOS, SEED);

      assertEquals(ref.price(), p.value(), 0.15, "price");
      assertEquals(ref.delta(), p.gradient(EquityMarket::spot), 0.02, "delta");
      assertEquals(ref.vega(), p.gradient(EquityMarket::vol), 0.60, "vega");
      assertEquals(ref.rho(), p.gradient(EquityMarket::rate), 0.60, "rho");
      assertEquals(ref.strikeSensitivity(), p.gradient(EquityMarket::strike), 0.02, "dV/dK");
    }
  }

  @Test
  void adjointGreeksMatchCentralBumpForAsian() {
    EquityMarket m = EquityMarket.atmOneYear();

    try (MonteCarlo<EquityMarket> greeks = MonteCarlo.of(Products.asianCall())
             .market(m).steps(STEPS).fp64().gradients().on("cpu-jit").build();
         MonteCarlo<EquityMarket> price = MonteCarlo.of(Products.asianCall())
             .market(m).steps(STEPS).fp64().valueOnly().on("cpu-jit").build()) {

      Nabla.TypedEvaluation<EquityMarket> adjoint = greeks.run(SCENARIOS, SEED);

      double h = 0.01 * m.spot();
      double up = price.run(m.withSpot(m.spot() + h), SCENARIOS, SEED).value();
      double dn = price.run(m.withSpot(m.spot() - h), SCENARIOS, SEED).value();
      double bumpDelta = (up - dn) / (2 * h);

      assertEquals(bumpDelta, adjoint.gradient(EquityMarket::spot), 5e-3, "adjoint delta vs central bump");
      assertTrue(adjoint.value() > 0.0, "Asian call price positive");
    }
  }

  @Test
  void bytecodeKernelReproducesScalarOracle() {
    EquityMarket m = EquityMarket.atmOneYear();

    for (Product<EquityMarket> product : java.util.List.of(
        Products.europeanCall(), Products.asianCall(), Products.lookbackCall(), Products.lookbackPut())) {

      try (MonteCarlo<EquityMarket> oracle = MonteCarlo.of(product)
               .market(m).steps(STEPS).fp64().gradients().on("cpu").build();
           MonteCarlo<EquityMarket> jit = MonteCarlo.of(product)
               .market(m).steps(STEPS).fp64().gradients().on("cpu-jit").build()) {

        Nabla.TypedEvaluation<EquityMarket> a = oracle.run(SCENARIOS, SEED);
        Nabla.TypedEvaluation<EquityMarket> b = jit.run(SCENARIOS, SEED);

        assertEquals(a.value(), b.value(), 1e-6 * (1 + Math.abs(a.value())),
            product.label() + " price: cpu-jit vs oracle");
        assertEquals(a.gradient(EquityMarket::spot), b.gradient(EquityMarket::spot), 1e-6 * (1 + Math.abs(a.gradient(EquityMarket::spot))),
            product.label() + " delta: cpu-jit vs oracle");
        assertEquals(a.gradient(EquityMarket::vol), b.gradient(EquityMarket::vol), 1e-6 * (1 + Math.abs(a.gradient(EquityMarket::vol))),
            product.label() + " vega: cpu-jit vs oracle");
      }
    }
  }

  @Test
  void setInputLadderReusesOneKernel() {
    try (MonteCarlo<EquityMarket> mc = MonteCarlo.of(Products.europeanCall())
        .market(EquityMarket.atmOneYear()).steps(1).fp64().gradients().on("cpu-jit").build()) {

      double last = Double.NEGATIVE_INFINITY;
      for (double spot = 80.0; spot <= 120.0; spot += 10.0) {
        Nabla.TypedEvaluation<EquityMarket> p = mc.run(mc.market().withSpot(spot), SCENARIOS, SEED);
        assertTrue(p.value() >= last - 1e-6, "call price monotone in spot along the ladder");
        double d = p.gradient(EquityMarket::spot);
        assertTrue(d > 0.0 && d < 1.0 + 1e-6, "call delta in (0,1)");
        last = p.value();
      }
    }
  }
}
