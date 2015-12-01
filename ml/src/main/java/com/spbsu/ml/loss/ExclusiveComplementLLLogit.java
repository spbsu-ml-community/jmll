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
public class ExclusiveComplementLLLogit extends FuncC1.Stub implements TargetFunc {
  private final double alpha;
  protected final Vec target;
  private final Vec complimentProbab;
  private final DataSet<?> owner;

  public ExclusiveComplementLLLogit(double alpha, final Vec target, final Vec complimentProbab, final DataSet<?> owner) {
    this.alpha = alpha;
    this.target = target;
    this.complimentProbab = complimentProbab;
    this.owner = owner;
  }

  @Override
  public Vec gradient(final Vec x) {
    final Vec result = new ArrayVec(x.dim());
    final double a = alpha;
    for (int i = 0; i < x.dim(); i++) {
      final double expX = exp(x.get(i));
      final double pX = expX / (1 + expX);
      double c = complimentProbab.get(i) ;
      if (target.get(i) > 0)  // positive example
        result.set(i, (a - 1) * pX / (a + expX));
      else
        result.set(i, c * (a - 1) * pX / (a * c + (c - 1) * expX - 1));
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
      final double pGen = complimentP * (alpha  + (1 - alpha) * pX);
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
