package com.spbsu.ml.loss.multiclass;

import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.ml.Func;

/**
 * User: qdeee
 * Date: 09.04.14
 */
public class MCMicroPrecision extends Func.Stub {
  protected final Vec target;

  public MCMicroPrecision(final Vec target) {
    this.target = target;
  }

  @Override
  public double value(final Vec x) {
    int tp = 0;
    int fp = 0;
    for (int i = 0; i < x.dim(); i++) {
      int expected = (int) target.get(i);
      int actual = (int) x.get(i);

      //skip unrecognized class
      if (actual == -1)
        continue;

      if (actual == expected)
        tp += 1;
      else
        fp += 1;
    }
    return tp / (tp + fp + 0.);
  }

  @Override
  public int dim() {
    return target.dim();
  }
}
