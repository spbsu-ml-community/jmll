package com.spbsu.ml.func.generic;

import com.spbsu.commons.math.FuncC1;
import com.spbsu.commons.math.MathTools;
import com.spbsu.commons.math.vectors.SingleValueVec;
import com.spbsu.commons.math.vectors.Vec;

public class Tanh extends FuncC1.Stub{
  @Override
  public Vec gradient(Vec x) {
    final double exp = Math.exp(x.get(0));
    return new SingleValueVec(4 / MathTools.sqr(1 / exp + exp));
  }

  @Override
  public double value(Vec x) {
    final double exp = Math.exp(2 * x.get(0));
    return (exp - 1) / (exp + 1);
  }

  @Override
  public int dim() {
    return 1;
  }
}
