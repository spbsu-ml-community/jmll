package com.spbsu.ml.loss;

import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.commons.math.vectors.impl.vectors.ArrayVec;
import com.spbsu.commons.seq.IntSeq;
import com.spbsu.ml.BlockedTargetFunc;
import com.spbsu.commons.math.FuncC1;
import com.spbsu.ml.data.set.DataSet;
import com.spbsu.ml.func.generic.Logit;

import static java.lang.Math.exp;
import static java.lang.Math.log;

/**
 * We use value representation = \frac{e^x}{e^x + 1}.
 * User: solar
 * Date: 21.12.2010
 * Time: 22:37:55
 */
public class LLLogit extends FuncC1.Stub implements BlockedTargetFunc {
  protected final Vec target;
  private final DataSet<?> owner;

  public LLLogit(final Vec target, final DataSet<?> owner) {
    this.target = target;
    this.owner = owner;
  }

  public LLLogit(final IntSeq target, final DataSet<?> owner) {
    this.target = new ArrayVec(target.length());
    for (int i = 0; i < target.length(); i++) {
      this.target.set(i, target.at(i));
    }
    this.owner = owner;
  }

  @Override
  public Vec gradient(final Vec x) {
    final Vec result = new ArrayVec(x.dim());
    for (int i = 0; i < x.dim(); i++) {
      final double expX = exp(x.get(i));
      final double pX = expX / (1 + expX);
      if (target.get(i) > 0) // positive example
        result.set(i, pX - 1);
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
      if (target.get(i) > 0) // positive example
        result += log(pX);
      else // negative
        result += log(1 - pX);
    }

    return exp(-result/point.dim());
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

  @Override
  public Logit block(int index) {
    return new Logit(target.get(index) > 0 ? -1. : 1., 0.);
  }

  @Override
  public int blocksCount() {
    return target.dim();
  }
}
