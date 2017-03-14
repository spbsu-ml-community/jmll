package com.spbsu.ml.func;

import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.commons.math.Func;

/**
* User: qdeee
* Date: 26.12.14
*/
public class ScaledFunc extends Func.Stub {
  private final double w;
  private final Func func;

  public ScaledFunc(final double w, final Func func) {
    this.w = w;
    this.func = func;
  }

  @Override
  public double value(final Vec x) {
    return w * func.value(x);
  }

  @Override
  public int dim() {
    return func.dim();
  }
}
