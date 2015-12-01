package com.spbsu.ml.loss;

import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.commons.math.vectors.impl.vectors.ArrayVec;
import com.spbsu.commons.math.FuncC1;
import com.spbsu.ml.TargetFunc;
import com.spbsu.ml.data.set.DataSet;

import static java.lang.Math.exp;
import static java.lang.Math.log;

/**
 * We use value representation = e^{-(x + 1)^2}.
 * User: solar
 * Date: 21.12.2010
 * Time: 22:37:55
 */
public class LLX2 extends FuncC1.Stub implements TargetFunc {
  private final Vec target;
  private final DataSet<?> owner;

  public LLX2(final Vec target, final DataSet<?> owner) {
    this.target = target;
    this.owner = owner;
  }

  @Override
  public Vec gradient(final Vec point) {
    final Vec result = new ArrayVec(point.dim());
    for (int i = 0; i < point.dim(); i++) {
      final double x = point.get(i) + 1;
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

  @Override
  public double value(final Vec point) {
    double result = 0;
    for (int i = 0; i < point.dim(); i++) {
      final double x = point.get(i) + 1;
      final double pX = exp(-x * x);
      if (target.get(i) > 0) // positive example
        result -= log(pX);
      else // negative
        result -= log(1 - pX);
    }

    return result;
  }

  @Override
  public DataSet<?> owner() {
    return owner;
  }
}
