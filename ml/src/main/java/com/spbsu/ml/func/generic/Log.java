package com.spbsu.ml.func.generic;

import com.spbsu.commons.math.vectors.SingleValueVec;
import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.ml.FuncC1;

/**
* User: solar
* Date: 26.05.15
* Time: 11:45
*/
public class Log extends FuncC1.Stub {
  private final double a, b;

  public Log(double a, double b) {
    this.a = a;
    this.b = b;
  }

  @Override
  public Vec gradient(Vec x) {
    final double denom = a * x.get(0) + b;
    return new SingleValueVec(a / denom);
  }

  @Override
  public double value(Vec x) {
    return Math.log(a * x.get(0) + b);
  }

  @Override
  public int dim() {
    return 1;
  }
}
