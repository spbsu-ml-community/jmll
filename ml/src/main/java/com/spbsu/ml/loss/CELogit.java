package com.spbsu.ml.loss;

import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.commons.math.vectors.impl.vectors.ArrayVec;
import com.spbsu.commons.math.FuncC1;
import com.spbsu.ml.TargetFunc;
import com.spbsu.ml.data.set.DataSet;

import static java.lang.Math.exp;

/**
 * We use value representation = \frac{e^x}{e^x + 1}.
 * User: solar
 * Date: 21.12.2010
 * Time: 22:37:55
 */
public class CELogit extends FuncC1.Stub implements TargetFunc{
  private final Vec target;
  private final DataSet<?> owner;

  public CELogit(final Vec target, final DataSet<?> owner) {
    this.target = target;
    this.owner = owner;
  }

  @Override
  public Vec gradient(final Vec x) {
    final Vec result = new ArrayVec(x.dim());
    for (int i = 0; i < x.dim(); i++) {
      final double b = target.get(i) > 0 ? 1 : -1;
      final double a = exp(x.get(i) * b);
      result.set(i, 2 * a * b / (1 + a) / (1 + a));
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
      final double expMX = exp(-point.get(i));
      final double pX = 1. / (1. + expMX);
      if (target.get(i) > 0) // positive example
        result += 1 - pX;
      else // negative
        result += pX;
    }

    return result / point.dim();
  }

  @Override
  public DataSet<?> owner() {
    return owner;
  }
}
