package com.spbsu.ml.func.generic;

import com.spbsu.commons.math.MathTools;
import com.spbsu.commons.math.vectors.SingleValueVec;
import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.commons.math.FuncC1;

/**
* User: solar
* Date: 26.05.15
* Time: 11:45
*/
public class Logit extends FuncC1.Stub {
  private final double a, b;

  public Logit(double a, double b) {
    this.a = a;
    this.b = b;
  }

  @Override
  public Vec gradient(Vec x) {
    return new SingleValueVec(a * (prob(x) - 1));
  }

  @Override
  public double value(Vec x) {
    return Math.log(prob(x));
  }

  private double prob(Vec x) {
    return 1. / (1. + Math.exp(a * x.get(0) + b));
  }

  @Override
  public int dim() {
    return 1;
  }

  @Override
  public String toString() {
    return "-log(1+exp(" + a + "x" + (Math.abs(b) > MathTools.EPSILON ? " + " + b : "") + "))";
  }
}
