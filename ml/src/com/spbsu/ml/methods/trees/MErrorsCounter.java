package com.spbsu.ml.methods.trees;

import com.spbsu.commons.math.MathTools;

import static java.lang.Math.*;

/** Key idea is to find minimal expectation of errors count: \min_s \sum_i \frac{1}{1 + e^{(x_i + s})y_i}, where x_i -- current score, y_i \in \{-1,1\} -- category
 * for this we need to get solution for \sum_i \frac{y_i}{1 + e^{y_i(x_i + s}}. This equation is difficult to solve in closed form so
 * we use Taylor series approximation. For this we need to make substitution s = log(1-v) - log(1+v) so that Maclaurin series in terms of
 * v were limited. Final optimization looks like:
 * \arg \min \sum_i \frac{1}{1 + \left(1 - t \over 1 + t\right)^y_i e^{y_i x_i}}
 * Maclaurin series (a = e^{x_i y_i}, b = y_i):
 * 1   \frac{1}{a + 1} +
 * x   \frac{2ab}}{(a + 1)^2} +
 * x^2 \frac{2ab^2(a - 1)}{(a + 1)^3} +
 * x^3 \frac{2ab(a^2(2b^2 + 1) + a(2 - 8b^2) + 2b^2 + 1)}{3(a + 1)^4} +
 * x^4 \frac{2ab^2(a^2(b^2 + 2) + a(4 - 10b^2) + b^2 + 2)}{3(a + 1)^5} +
 * MErrorCounter calculates Maclaurin coefficients.
 */
public class MErrorsCounter {
  public volatile int good = 0;
  public volatile int bad = 0;
  public volatile double m0;
  public volatile double m1;
  public volatile double m2;
  public volatile double m3;
  public volatile double m4;

  public double alpha() {
    double y = optimal();
    return log((1. - y) / (1. + y));// * (1 - log(2.)/log(n + 2.));
  }

  public double optimal() {
    if (good == 0 || bad == 0)
      return 0;
    final double[] x = new double[3];

    int cnt = MathTools.cubic(x, 4 * m4, 3 * m3, 2 * m2, m1);
    double y = 0.;
    double bestLL = m0;
    for (int i = 0; i < cnt; i++) {
      final double normX = signum(x[i]) * min(abs(x[i]), 0.9);
      final double score = scoreInner(normX);
      if (score < bestLL) {
        y = normX;
        bestLL = score;
      }
    }

    return y;
  }

  private double scoreInner(double x) {
    return m0 + m1 * x + m2 * x * x + m3 * x * x * x + m4 * x * x * x * x;
  }

  public double score() {
    return score(0);
  }

  public double score(double R) {
    final double optimal = optimal();
//    final double l = log((1. - optimal) / (1. + optimal));
    return scoreInner(optimal);// * R;
  }

  public void found(double current, double target, double weight) {
    final double b = target > 0 ? 1. : -1.;
    final double a = exp(current*b);
    /*
     * x   \frac{2ab}}{(a + 1)^2} +
     * x^2 \frac{2ab^2(a - 1)}{(a + 1)^3} +
     * x^3 \frac{2ab(a^2(2b^2 + 1) + a(2 - 8b^2) + 2b^2 + 1)}{3(a + 1)^4} +
     * x^4 \frac{2ab^2(a^2(b^2 + 2) + a(4 - 10b^2) + b^2 + 2)}{3(a + 1)^5} +
     * MErrorCounter calculates Maclaurin coefficients.
     */
    // \frac{1}{a + 1}
    m0 += 1/(a + 1);
    // \frac{2ab}}{(a + 1)^2}
    final double m1 = 2 * a * b / (1 + a) / (1 + a);
    // \frac{2ab^2(a - 1)}{(a + 1)^3}
    final double m2 = 2 * a * b * b * (a - 1) / (1 + a) / (1 + a) / (1 + a);
    // \frac{2ab(a^2(2b^2 + 1) + a(2 - 8b^2) + 2b^2 + 1)}{3(a + 1)^4}
    final double m3 = 2 * a * b * (a * a * (2 * b * b + 1) + a * (2 - 8 * b * b) + 2 * b * b + 1) / 3 / (1 + a) / (1 + a) / (1 + a) / (1 + a);
    // \frac{2ab^2(a^2(b^2 + 2) + a(4 - 10b^2) + b^2 + 2)}{3(a + 1)^5}
    final double m4 = 2 * a * b * b * (a * a * (b * b + 2) + a * (4 - 10 * b * b) + b * b + 2) / 3 / (1 + a) / (1 + a) / (1 + a) / (1 + a) / (1 + a);
    this.m1 += m1;
    this.m2 += m2;
    this.m3 += m3;
    this.m4 += m4;
    if (b > 0)
      good++;
    else
      bad++;
  }

  public void add(MErrorsCounter counter) {
    m0 += counter.m0;
    m1 += counter.m1;
    m2 += counter.m2;
    m3 += counter.m3;
    m4 += counter.m4;
    good += counter.good;
    bad += counter.bad;
  }

  public void sub(MErrorsCounter counter) {
    m0 -= counter.m0;
    m1 -= counter.m1;
    m2 -= counter.m2;
    m3 -= counter.m3;
    m4 -= counter.m4;
    good -= counter.good;
    bad -= counter.bad;
  }

  /**
   * Combine two parts of $LL=-\sum_{\{a_c\}, b} \sum_c log(1+e^{-1^{I(b=c)}(-a-m(c)x)})$ depending on $m: C \to \{-1,1\}$
   * actually they differs in sign of odd derivations so we need only to properly sum them :)
   */
  public void invert(MErrorsCounter counter, double sign) {
    m1 += sign * 2 * counter.m1;
    m3 += sign * 2 * counter.m3;
  }

  public int size() {
    return good + bad;
  }
}
