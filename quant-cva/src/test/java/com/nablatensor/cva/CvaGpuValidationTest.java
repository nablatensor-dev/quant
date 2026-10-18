package com.nablatensor.cva;

import static org.junit.jupiter.api.Assertions.*;
import java.util.List;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.condition.EnabledIfSystemProperty;

/** CVA number validation calls only the explicitly selected Vulkan kernel. */
@EnabledIfSystemProperty(named = "quant.backend", matches = "vulkan")
class CvaGpuValidationTest {
  @Test
  void cvaIsPositiveAndScalesWithLossGivenDefault() {
    var counterparty = CreditName.of().id("CPTY")
        .curve(HazardCurve.fromFlatSpread(150, 0.4, 5)).recovery(0.4)
        .rating(CreditName.RatingEnum.BBB).sector(CreditName.SectorEnum.FINANCIAL).build();
    var trade = FxForward.of().id("FX").side(FxForward.SideEnum.BUY_FOREIGN)
        .foreignNotional(1_000_000).strike(1).settlementYears(2).build();
    var netting = NettingSet.of().id("NS").counterparty(counterparty)
        .trades(List.of(trade)).collateral(CollateralAgreement.uncollateralised()).build();
    var simulation = ExposureSimulation.of(netting, 4).on("vulkan");
    var market = CvaMarket.demo();
    double base = simulation.cvaOnly(market, 8192, 42);
    var higherRecovery = CvaMarket.of().from(market).recovery(0.6).build();
    double reduced = simulation.cvaOnly(higherRecovery, 8192, 42);
    assertTrue(Double.isFinite(base) && base > 0);
    assertEquals(base * (1 - higherRecovery.recovery()) / (1 - market.recovery()),
        reduced, base * 0.002);
  }
}
