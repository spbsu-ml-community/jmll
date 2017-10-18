package com.expleague.ml.func;

import com.expleague.commons.math.Func;
import com.expleague.commons.math.vectors.Vec;

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
