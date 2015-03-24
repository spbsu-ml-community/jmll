package com.spbsu.ml.loss;

import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.commons.math.vectors.VecTools;
import com.spbsu.commons.math.vectors.impl.vectors.ArrayVec;
import com.spbsu.ml.data.set.DataSet;

import static java.lang.Math.exp;
import static java.lang.Math.log;

/**
 * Created by irlab on 13.02.2015.
 */
public class WeightedLLLogit extends LLLogit {
  private Vec weights;
  private double sumWeights;

  public WeightedLLLogit(final Vec target, final DataSet<?> owner) {
    this(target, owner, VecTools.fill(new ArrayVec(target.dim()), 1));
  }

  public WeightedLLLogit(final Vec target, final DataSet<?> owner, final Vec weights) {
    super(target, owner);
    this.weights = weights;
    this.sumWeights = VecTools.sum(weights);
  }

  public void setWeights(final Vec weights) {
    this.weights = weights;
    this.sumWeights = VecTools.sum(weights);
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
    VecTools.scale(result, weights);
    return result;
  }

  @Override
  public double value(final Vec point) {
    double result = 0;
    for (int i = 0; i < point.dim(); i++) {
      final double expMX = exp(-point.get(i));
      final double pX = 1. / (1. + expMX);
      if (target.get(i) > 0) // positive example
        result += log(pX) * weights.get(i);
      else // negative
        result += log(1 - pX) * weights.get(i);
    }

    return exp(result / sumWeights);
  }
}
