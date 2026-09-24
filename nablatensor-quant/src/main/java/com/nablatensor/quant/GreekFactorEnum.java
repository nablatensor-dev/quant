package com.nablatensor.quant;

import com.nablatensor.engine.Nabla;
import java.util.function.ToDoubleFunction;

/** Typed market-factor accessors and bumps for equity Greek examples. */
public enum GreekFactorEnum {
  SPOT("spot", EquityMarket::spot) {
    @Override public EquityMarket bump(EquityMarket market, double h) {
      return market.withSpot(market.spot() + h);
    }
  },
  STRIKE("strike", EquityMarket::strike) {
    @Override public EquityMarket bump(EquityMarket market, double h) {
      return market.withStrike(market.strike() + h);
    }
  },
  VOL("vol", EquityMarket::vol) {
    @Override public EquityMarket bump(EquityMarket market, double h) {
      return market.withVol(market.vol() + h);
    }
  },
  RATE("rate", EquityMarket::rate) {
    @Override public EquityMarket bump(EquityMarket market, double h) {
      return market.withRate(market.rate() + h);
    }
  },
  MATURITY("maturity", EquityMarket::maturity) {
    @Override public EquityMarket bump(EquityMarket market, double h) {
      return market.withMaturity(market.maturity() + h);
    }
  };

  public final String label;
  private final ToDoubleFunction<EquityMarket> accessor;

  GreekFactorEnum(String label, ToDoubleFunction<EquityMarket> accessor) {
    this.label = label;
    this.accessor = accessor;
  }

  public double value(EquityMarket market) {
    return accessor.applyAsDouble(market);
  }

  public double greek(Nabla.TypedValuation<EquityMarket> valuation) {
    return valuation.greek(accessor);
  }

  public abstract EquityMarket bump(EquityMarket market, double h);
}
