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
public class ShiftedLLLogit extends LLLogit {
  private Vec step1Scores;

  public ShiftedLLLogit(final Vec target, final DataSet<?> owner) {
    this(target, owner, new ArrayVec(target.dim()));
  }

  public ShiftedLLLogit(final Vec target, final DataSet<?> owner, final Vec step1Scores) {
    super(target, owner);
    this.step1Scores = step1Scores;
  }

  public void setStep1Scores(final Vec step1Scores) {
    this.step1Scores = step1Scores;
  }

  @Override
  public Vec gradient(final Vec x) {
    final Vec result = new ArrayVec(x.dim());
    for (int i = 0; i < x.dim(); i++) {
      final double X = step1Scores.get(i) + x.get(i);
      final double expX = exp(X);
      final double pX = expX / (1 + expX);
      if (target.get(i) > 0) // positive example
        result.set(i, pX - 1);
      else // negative
        result.set(i, pX);
    }
    return result;
  }

  @Override
  public double value(final Vec point) {
    double result = 0;
    for (int i = 0; i < point.dim(); i++) {
      final double X = step1Scores.get(i) + point.get(i);
      final double expMX = exp(-X);
      final double pX = 1. / (1. + expMX);
      if (target.get(i) > 0) // positive example
        result += log(pX);
      else // negative
        result += log(1 - pX);
    }

    return exp(result / point.dim());
  }
}
