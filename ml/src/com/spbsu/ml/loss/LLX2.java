package com.spbsu.ml.loss;

import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.commons.math.vectors.impl.ArrayVec;
import com.spbsu.ml.FuncC1;

import static java.lang.Math.exp;
import static java.lang.Math.log;

/**
 * We use value representation = e^{-(x + 1)^2}.
 * User: solar
 * Date: 21.12.2010
 * Time: 22:37:55
 */
public class LLX2 extends FuncC1.Stub {
  private final Vec target;

  public LLX2(Vec target) {
    this.target = target;
  }

  public Vec gradient(Vec point) {
    Vec result = new ArrayVec(point.dim());
    for (int i = 0; i < point.dim(); i++) {
      double x = point.get(i) + 1;
      if (target.get(i) > 0) // positive example
        result.set(i, -2 * x);
      else // negative
        result.set(i, 2 * x/(exp(x*x) - 1));
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
      double x = point.get(i) + 1;
      double pX = exp(-x * x);
      if (target.get(i) > 0) // positive example
        result -= log(pX);
      else // negative
        result -= log(1 - pX);
    }

    return result;
  }
}
