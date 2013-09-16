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
  public volatile double maclaurinLL0;
  public volatile double maclaurinLL1;
  public volatile double maclaurinLL2;
  public volatile double maclaurinLL3;
  public volatile double maclaurinLL4;
  private double alpha = Double.NaN;

  public double alpha() {
    if (!Double.isNaN(alpha))
      return alpha;
    if (good == 0 || bad == 0)
      return 0;
    final double[] x = new double[3];

    int cnt = MathTools.cubic(x, maclaurinLL4, maclaurinLL3, maclaurinLL2, maclaurinLL1);
    double y = 0.;
    double bestLL = maclaurinLL0;
    for (int i = 0; i < cnt; i++) {
      if (abs(x[i]) < 1 && score(x[i]) > bestLL) {
        y = x[i];
        bestLL = score(y);
      }
    }

    return alpha = log((1. - y) / (1. + y));
  }

  private double score(double x) {
    if (good == 0 || bad == 0)
      return maclaurinLL0;
    return maclaurinLL0 - 2 * maclaurinLL1 * x - maclaurinLL2 * x * x;
  }

  public double score() {
    double alpha = alpha();
    double x = (1 - exp(alpha))/(1 + exp(alpha));
    return score(x) - maclaurinLL0;
  }

  public void found(double current, double target, double weight) {
    final double b = target > 0 ? 1. : -1.;
    final double eab = exp(current*b);
    final double eabPlusOne = eab + 1;
    final double eabMinusOne = eab - 1;
    double denominator = eabPlusOne;
    maclaurinLL0 += weight * log(eab/(1. + eab));
    maclaurinLL1 += weight * b/denominator;
    denominator *= eabPlusOne;
    maclaurinLL2 += weight * 2 * eab/denominator;
    denominator *= eabPlusOne;
    maclaurinLL3 += weight * 2 * b * eab * eabMinusOne /denominator;
    denominator *= eabPlusOne;
    maclaurinLL4 += weight * 2 * eab * eabMinusOne * eabMinusOne / denominator;
    if (b > 0)
      good++;
    else
      bad++;
    alpha = Double.NaN;
  }

  public void add(LLCounter counter) {
    maclaurinLL0 += counter.maclaurinLL0;
    maclaurinLL1 += counter.maclaurinLL1;
    maclaurinLL2 += counter.maclaurinLL2;
    maclaurinLL3 += counter.maclaurinLL3;
    maclaurinLL4 += counter.maclaurinLL4;
    good += counter.good;
    bad += counter.bad;
    alpha = Double.NaN;
  }

  public void sub(LLCounter counter) {
    maclaurinLL0 -= counter.maclaurinLL0;
    maclaurinLL1 -= counter.maclaurinLL1;
    maclaurinLL2 -= counter.maclaurinLL2;
    maclaurinLL3 -= counter.maclaurinLL3;
    maclaurinLL4 -= counter.maclaurinLL4;
    good -= counter.good;
    bad -= counter.bad;
    alpha = Double.NaN;
  }

  /**
   * Combine two parts of $LL=-\sum_{\{a_c\}, b} \sum_c log(1+e^{-1^{I(b=c)}(-a-m(c)x)})$ depending on $m: C \to \{-1,1\}$
   * actually they differs in sign of odd derivations so we need only to properly sum them :)
   */
  public void add(LLCounter counter, double xsign) {
    maclaurinLL0 += counter.maclaurinLL0;
    maclaurinLL1 += xsign * counter.maclaurinLL1;
    maclaurinLL2 += counter.maclaurinLL2;
    maclaurinLL3 += xsign * counter.maclaurinLL3;
    maclaurinLL4 += counter.maclaurinLL4;
    good += counter.good;
    bad += counter.bad;
    alpha = Double.NaN;
  }

  public void sub(LLCounter counter, double xsign) {
    maclaurinLL0 -= counter.maclaurinLL0;
    maclaurinLL1 -= xsign * counter.maclaurinLL1;
    maclaurinLL2 -= counter.maclaurinLL2;
    maclaurinLL3 -= xsign * counter.maclaurinLL3;
    maclaurinLL4 -= counter.maclaurinLL4;
    good -= counter.good;
    bad -= counter.bad;
    alpha = Double.NaN;
  }

  public int size() {
    return good + bad;
  }
}
