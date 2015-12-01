package com.spbsu.ml.func.generic;

import com.spbsu.commons.math.vectors.SingleValueVec;
import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.commons.math.FuncC1;

/**
* User: solar
* Date: 27.05.15
* Time: 17:59
*/
public class ReLU extends FuncC1.Stub {
  @Override
  public Vec gradient(Vec x) {
    final double v = x.get(0);
    return new SingleValueVec(v > 0 ? 1 : 0);
  }

  @Override
  public double value(Vec x) {
    final double v = x.get(0);
    return v > 0 ? v : 0;
  }

  @Override
  public int dim() {
    return 0;
  }
}
