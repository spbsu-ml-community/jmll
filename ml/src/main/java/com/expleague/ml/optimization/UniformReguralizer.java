package com.expleague.ml.optimization;

import com.expleague.commons.math.vectors.Vec;
import com.expleague.ml.func.RegularizerFunc;

public class UniformReguralizer extends RegularizerFunc.Stub {
  private final int dim;

  public UniformReguralizer(int dim) {
    this.dim = dim;
  }

  @Override
  public Vec project(Vec x) {
    return x;
  }

  @Override
  public double value(Vec x) {
    return 0;
  }

  @Override
  public int dim() {
    return dim;
  }
}
