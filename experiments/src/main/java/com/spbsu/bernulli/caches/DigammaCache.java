package com.spbsu.bernulli.caches;

import org.apache.commons.math3.special.Gamma;
import org.apache.commons.math3.util.FastMath;

import java.util.Arrays;

//calculate and cache Beta(alpha + m, beta + n -m) / Beta(alpha,beta)
public class DigammaCache {
  int maxOffset;
  double base;
  final double[] values;
  final boolean[] cached;

  public DigammaCache(double base, int n) {
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
        values[offset] = -Gamma.GAMMA - 1 / x;
        cached[offset] = true;
      } else if (x >= C_LIMIT) {
        // use method 4 (accurate to O(1/x^8)
        double inv = 1 / (x * x);
        //            1       1        1         1
        // log(x) -  --- - ------ + ------- - -------
        //           2 x   12 x^2   120 x^4   252 x^6
        values[offset] = FastMath.log(x) - 0.5 / x - inv * ((1.0 / 12) + inv * (1.0 / 120 - inv / 252));
        cached[offset] = true;
      } else if (offset == (maxOffset + 1)) {
        cached[offset] = true;
        values[offset] = Gamma.digamma(base + offset);
      } else {
        values[offset] = calculate(offset + 1) - 1.0 / x;
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
