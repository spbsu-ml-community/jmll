package com.spbsu.ml.loss;

import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.commons.math.vectors.impl.vectors.ArrayVec;
import com.spbsu.commons.math.FuncC1;
import com.spbsu.ml.TargetFunc;
import com.spbsu.ml.data.set.DataSet;

import static java.lang.Math.exp;
import static java.lang.Math.log;

/**
 * User: solar
 * Date: 19.03.15
 * Time: 13:43
 */
public class ComplementLLLogit extends FuncC1.Stub implements TargetFunc {
  protected final Vec target;
  private final Vec complimentProbab;
  private final DataSet<?> owner;

  public ComplementLLLogit(final Vec target, final Vec complimentProbab, final DataSet<?> owner) {
    this.target = target;
    this.complimentProbab = complimentProbab;
    this.owner = owner;
  }

  @Override
  public Vec gradient(final Vec x) {
    final Vec result = new ArrayVec(x.dim());
    for (int i = 0; i < x.dim(); i++) {
      final double expX = exp(x.get(i));
      final double pX = expX / (1 + expX);
      final double complimentP = complimentProbab.get(i);
      if (target.get(i) > 0) // positive example
        result.set(i, -pX * (1 - complimentP)/(complimentP + expX));
      else // negative
        result.set(i, pX);
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
      final double complimentP = complimentProbab.get(i);
      final double pGen = complimentP + pX * (1 - complimentP);
      if (target.get(i) > 0) // positive example
        result += log(pGen);
      else // negative
        result += log(1 - pGen);
    }

    return exp(result / point.dim());
  }

  public int label(final int idx) {
    return (int)target.get(idx);
  }

  public Vec labels() {
    return target;
  }

  @Override
  public DataSet<?> owner() {
    return owner;
  }

}
