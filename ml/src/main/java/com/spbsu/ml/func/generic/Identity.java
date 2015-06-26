package com.spbsu.ml.func.generic;

import com.spbsu.commons.math.vectors.SingleValueVec;
import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.ml.FuncC1;

/**
* User: solar
* Date: 26.05.15
* Time: 11:45
*/
public class Identity extends FuncC1.Stub {
  private static final SingleValueVec one = new SingleValueVec(1);
  @Override
  public Vec gradient(Vec x) {
    return one;
  }

  @Override
  public double value(Vec x) {
    return x.get(0);
  }

  @Override
  public int dim() {
    return 1;
  }
}
