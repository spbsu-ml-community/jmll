package com.spbsu.ml.func.generic;

import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.commons.math.vectors.VecTools;
import com.spbsu.commons.math.FuncC1;

/**
* User: solar
* Date: 26.05.15
* Time: 11:45
*/
public class WSumSigmoid extends FuncC1.Stub {
  public final Vec weights;

  public WSumSigmoid(Vec weights) {
    this.weights = weights;
  }

  public Vec gradientTo(Vec x, Vec to) {
    final double exp = Math.exp(-VecTools.multiply(x, weights));
    final double value = exp / (1 + exp) / (1 + exp);
    VecTools.assign(to, weights);
    VecTools.scale(to, value);
    return to;
  }

  @Override
  public double value(Vec x) {
    final double exp = Math.exp(-VecTools.multiply(x, weights));
    return 1./(1. + exp);
  }

  @Override
  public int dim() {
    return -1;
  }
}
