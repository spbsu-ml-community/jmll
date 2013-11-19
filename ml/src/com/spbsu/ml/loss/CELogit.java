package com.spbsu.ml.loss;

import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.commons.math.vectors.impl.ArrayVec;
import com.spbsu.ml.Oracle1;

import static java.lang.Math.exp;

/**
 * We use value representation = \frac{e^x}{e^x + 1}.
 * User: solar
 * Date: 21.12.2010
 * Time: 22:37:55
 */
public class CELogit implements Oracle1 {
  private final Vec target;

  public CELogit(Vec target) {
    this.target = target;
  }

  @Override
  public Vec gradient(Vec point) {
    Vec result = new ArrayVec(point.dim());
    for (int i = 0; i < point.dim(); i++) {
      double b = target.get(i) > 0 ? 1 : -1;
      double a = exp(point.get(i) * b);
      result.set(i, 2 * a * b / (1 + a) / (1 + a));
    }
    return result;
  }

  @Override
  public int dim() {
    return target.dim();
  }

  public double value(Vec point) {
    double result = 0;
    for (int i = 0; i < point.dim(); i++) {
      double expMX = exp(-point.get(i));
      double pX = 1. / (1. + expMX);
      if (target.get(i) > 0) // positive example
        result += 1 - pX;
      else // negative
        result += pX;
    }

    return result / point.dim();
  }
}
