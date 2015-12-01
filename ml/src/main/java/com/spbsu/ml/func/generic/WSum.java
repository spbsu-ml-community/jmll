package com.spbsu.ml.func.generic;

import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.commons.math.vectors.VecTools;
import com.spbsu.commons.math.FuncC1;

/**
* User: solar
* Date: 26.05.15
* Time: 11:45
*/
public class WSum extends FuncC1.Stub {
  public final Vec weights;

  public WSum(Vec weights) {
    this.weights = weights;
  }

  @Override
  public Vec gradientTo(Vec x, Vec to) {
    VecTools.assign(to, weights);
    return to;
  }

  @Override
  public double value(Vec x) {
    return VecTools.multiply(weights, x);
  }

  @Override
  public int dim() {
    return weights.length();
  }
}
