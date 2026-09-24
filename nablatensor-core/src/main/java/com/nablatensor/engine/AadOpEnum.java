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
package com.nablatensor.engine;

/** Scalar operations a recorded tape can hold. */
public enum AadOpEnum {
  /** A compile-time constant baked into the tape. */
  CONST(false, false) {
    @Override public double apply(double a, double b, double external) { return external; }
    @Override public void propagate(double a, double b, double value, double gradient, GradientSink sink) {}
  },
  /** A named input the caller binds a value to on each replay. */
  INPUT(false, false) {
    @Override public double apply(double a, double b, double external) { return external; }
    @Override public void propagate(double a, double b, double value, double gradient, GradientSink sink) {}
  },
  /** A standard normal draw that varies per scenario; produced on-device by the replay kernel. */
  RANDN(false, false) {
    @Override public double apply(double a, double b, double external) { return external; }
    @Override public void propagate(double a, double b, double value, double gradient, GradientSink sink) {}
  },
  /** A uniform draw on {@code [0, 1)} that varies per scenario; produced on-device by the replay kernel. */
  RANDU(false, false) {
    @Override public double apply(double a, double b, double external) { return external; }
    @Override public void propagate(double a, double b, double value, double gradient, GradientSink sink) {}
  },
  /** Binary addition, {@code a + b}. */
  ADD(true, true) {
    @Override public double apply(double a, double b, double external) { return a + b; }
    @Override public void propagate(double a, double b, double value, double gradient, GradientSink sink) { sink.addA(gradient); sink.addB(gradient); }
  },
  /** Binary subtraction, {@code a - b}. */
  SUB(true, true) {
    @Override public double apply(double a, double b, double external) { return a - b; }
    @Override public void propagate(double a, double b, double value, double gradient, GradientSink sink) { sink.addA(gradient); sink.addB(-gradient); }
  },
  /** Binary multiplication, {@code a * b}. */
  MUL(true, true) {
    @Override public double apply(double a, double b, double external) { return a * b; }
    @Override public void propagate(double a, double b, double value, double gradient, GradientSink sink) { sink.addA(gradient * b); sink.addB(gradient * a); }
  },
  /** Binary division, {@code a / b}. */
  DIV(true, true) {
    @Override public double apply(double a, double b, double external) { return a / b; }
    @Override public void propagate(double a, double b, double value, double gradient, GradientSink sink) { sink.addA(gradient / b); sink.addB(-gradient * value / b / b); }
  },
  /** Unary negation, {@code -a}. */
  NEG(true, false) {
    @Override public double apply(double a, double b, double external) { return -a; }
    @Override public void propagate(double a, double b, double value, double gradient, GradientSink sink) { sink.addA(-gradient); }
  },
  /** Natural exponential, {@code e^a}. */
  EXP(true, false) {
    @Override public double apply(double a, double b, double external) { return Math.exp(a); }
    @Override public void propagate(double a, double b, double value, double gradient, GradientSink sink) { sink.addA(gradient * value); }
  },
  /** Natural logarithm, {@code ln a}. */
  LOG(true, false) {
    @Override public double apply(double a, double b, double external) { return Math.log(a); }
    @Override public void propagate(double a, double b, double value, double gradient, GradientSink sink) { sink.addA(gradient / a); }
  },
  /** Square root, {@code sqrt(a)}. */
  SQRT(true, false) {
    @Override public double apply(double a, double b, double external) { return Math.sqrt(a); }
    @Override public void propagate(double a, double b, double value, double gradient, GradientSink sink) { sink.addA(gradient / (2.0 * value)); }
  },
  /** Binary maximum, {@code max(a, b)}. */
  MAX(true, true) {
    @Override public double apply(double a, double b, double external) { return Math.max(a, b); }
    @Override public void propagate(double a, double b, double value, double gradient, GradientSink sink) { sink.addA(a > b ? gradient : a == b ? gradient * .5 : 0); sink.addB(b > a ? gradient : a == b ? gradient * .5 : 0); }
  },
  /** Binary minimum, {@code min(a, b)}. */
  MIN(true, true) {
    @Override public double apply(double a, double b, double external) { return Math.min(a, b); }
    @Override public void propagate(double a, double b, double value, double gradient, GradientSink sink) { sink.addA(a < b ? gradient : a == b ? gradient * .5 : 0); sink.addB(b < a ? gradient : a == b ? gradient * .5 : 0); }
  },
  /** Absolute value, {@code |a|}. */
  ABS(true, false) {
    @Override public double apply(double a, double b, double external) { return Math.abs(a); }
    @Override public void propagate(double a, double b, double value, double gradient, GradientSink sink) { sink.addA(gradient * (a >= 0 ? 1 : -1)); }
  };

  private final boolean argA;
  private final boolean argB;
  AadOpEnum(boolean argA, boolean argB) { this.argA = argA; this.argB = argB; }
  public boolean usesArgA() { return argA; }
  public boolean usesArgB() { return argB; }
  public abstract double apply(double a, double b, double external);
  public abstract void propagate(double a, double b, double value, double gradient, GradientSink sink);
  public interface GradientSink {
    void addA(double gradient);
    void addB(double gradient);
  }
}
