package com.expleague.ml.func;

import com.expleague.commons.math.vectors.Vec;
import com.expleague.commons.math.vectors.VecTools;
import com.expleague.commons.math.vectors.impl.vectors.ArrayVec;

public class WeightedLL extends LogLikelihood {
  private final Vec weights;

  public WeightedLL(Vec weights) {
    this.weights = new ArrayVec(weights.dim());
    VecTools.assign(this.weights, weights);
  }

  @Override
  public double value(Vec x) {
    double sum = 0.;
    for (int i = 0; i < x.dim(); i++) {
      sum += Math.log(x.get(i)) * weights.get(i);
    }
    return sum;
  }

  @Override
  public Vec gradientTo(Vec x, Vec to) {
    for (int i = 0; i < x.dim(); i++) {
      to.set(i, weights.get(i) / x.get(i));
    }
    return to;
  }

  @Override
  public int dim() {
    return weights.dim();
  }
}
