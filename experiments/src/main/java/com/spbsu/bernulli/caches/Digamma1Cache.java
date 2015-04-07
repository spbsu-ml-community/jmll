package com.spbsu.bernulli.caches;

import org.apache.commons.math3.special.Gamma;

import java.util.Arrays;

public class Digamma1Cache {
  int maxOffset;
  double base;
  final double[] values;
  final boolean[] cached;

  public Digamma1Cache(double base, int n) {
    values = new double[n + 2];
    cached = new boolean[n + 2];
    this.base = base;
    this.maxOffset = n;
  }

  // limits for switching algorithm in digamma
  /**
   * C limit.
   */
  private static final double C_LIMIT = 49;

  /**
   * S limit.
   */
  private static final double S_LIMIT = 1e-5;

  public final double calculate(int offset) {
    if (cached[offset]) {
      return values[offset];
    } else {
      double x = base + offset;
      if (x > 0 && x <= S_LIMIT) {
        // use method 5 from Bernardo AS103
        // accurate to O(x)
        values[offset] = 1.0 / (x * x);
        cached[offset] = true;
      } else if (x >= C_LIMIT) {
        final double inv = 1.0 / (x * x);
        //  1    1      1       1       1
        //  - + ---- + ---- - ----- + -----
        //  x      2      3       5       7
        //      2 x    6 x    30 x    42 x
        values[offset] = 1.0 / x + inv / 2 + inv / x * (1.0 / 6 - inv * (1.0 / 30 + inv / 42));
        cached[offset] = true;
      } else if (offset == (maxOffset + 1)) {
        cached[offset] = true;
        values[offset] = Gamma.trigamma(base + offset);
      } else {
        values[offset] = calculate(offset + 1) + 1 / (x * x);
        cached[offset] = true;
      }
    }
    return values[offset];
  }


  final public void update(double newBase) {
    if (this.base == newBase)
      return;
    this.base = newBase;
    Arrays.fill(cached, false);
  }
}
