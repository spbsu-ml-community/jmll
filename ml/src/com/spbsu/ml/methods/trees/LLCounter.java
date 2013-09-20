package com.spbsu.ml.methods.trees;

import com.spbsu.commons.math.MathTools;

import static java.lang.Math.*;

/** Key idea is to find \min_s \sum_i log \frac{1}{1 + e^{-(x_i + s})y_i}, where x_i -- current score, y_i \in \{-1,1\} -- category
 * for this we need to get solution for \sum_i \frac{y_i}{1 + e^{y_i(x_i + s}}. This equation is difficult to solve in closed form so
 * we use Taylor series approximation. For this we need to make substitution s = log(1-v) - log(1+v) so that Maclaurin series in terms of
 * v were limited.
 *
 * LLCounter is a class which calculates Maclaurin coefficients of original function and its first derivation.
 */
public class LLCounter {
  public volatile int good = 0;
  public volatile int bad = 0;
  public volatile double ll;
  public volatile double d1;
  public volatile double d2;
  public volatile double d3;
  public volatile double d4;

  public double alpha() {
    double y = optimal();
    return log((1. - y) / (1. + y));
  }
  public double optimal() {
    if (good == 0 || bad == 0)
      return 0;
    final double[] x = new double[3];

    int cnt = MathTools.cubic(x, d4/6, d3/2, d2, d1);
    double y = 0.;
    double bestLL = 0;
    for (int i = 0; i < cnt; i++) {
      if (abs(x[i]) >= 0.5) // skip too optimistic solutions
        continue;
      final double score = score(x[i]);
      if (score > bestLL) {
        y = x[i];
        bestLL = score;
      }
    }

    return y;
  }

  private double score(double x) {
    if (good == 0 || bad == 0)
      return 0;
    return d1 * x + d2 * x * x / 2 + d3 * x * x * x / 6 + d4 * x * x * x * x / 24;
  }

  public double fastScore() {
    return -d1 * d1 / 2 / d2;
  }

  public double score() {
    final double R = log(size() + 2.);
    return size() <= 1 ? 0 : score(optimal()) * R;
  }

  public void found(double current, double target, double weight) {
    final double b = target > 0 ? 1. : -1.;
    final double a = exp(-current*b);
    final double d1 = -2 * a * b / (1 + a);
    final double d2 = -d1 * d1 / a;
    final double d3 = b * d2 * (a * a + 3) / (1 + a);
    final double d4 = 12 * d2 * (1 + a * a * a) / (1 + a) / (1 + a);
    ll += - weight * log(1 + a);
    this.d1 += d1;
    this.d2 += d2;
    this.d3 += d3;
    this.d4 += d4;
    if (b > 0)
      good++;
    else
      bad++;
  }

  public void add(LLCounter counter) {
    ll += counter.ll;
    d1 += counter.d1;
    d2 += counter.d2;
    d3 += counter.d3;
    d4 += counter.d4;
    good += counter.good;
    bad += counter.bad;
  }

  public void sub(LLCounter counter) {
    ll -= counter.ll;
    d1 -= counter.d1;
    d2 -= counter.d2;
    d3 -= counter.d3;
    d4 -= counter.d4;
    good -= counter.good;
    bad -= counter.bad;
  }

  /**
   * Combine two parts of $LL=-\sum_{\{a_c\}, b} \sum_c log(1+e^{-1^{I(b=c)}(-a-m(c)x)})$ depending on $m: C \to \{-1,1\}$
   * actually they differs in sign of odd derivations so we need only to properly sum them :)
   */
  public void invert(LLCounter counter, double sign) {
    d1 += sign * 2 * counter.d1;
    d3 += sign * 2 * counter.d3;
  }

  public int size() {
    return good + bad;
  }
}
