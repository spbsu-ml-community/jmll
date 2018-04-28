package com.expleague.ml.func;

import com.expleague.commons.math.FuncC1;
import com.expleague.commons.math.vectors.Vec;

public class LogLikelihood extends FuncC1.Stub {
  @Override
  public double value(Vec x) {
    double sum = 0.;
    for (int i = 0; i < x.dim(); i++) {
      sum += Math.log(x.get(i));
    }
    return sum;
  }

  @Override
  public Vec gradientTo(Vec x, Vec to) {
    for (int i = 0; i < x.dim(); i++) {
      to.set(i, 1. / x.get(i));
    }
    return to;
  }

  @Override
  public int dim() {
    throw new UnsupportedOperationException();
  }
}
