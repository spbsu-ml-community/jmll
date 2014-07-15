package com.spbsu.ml.loss.multiclass;

import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.ml.Func;
import com.spbsu.ml.TargetFunc;
import com.spbsu.ml.data.set.DataSet;

/**
 * User: qdeee
 * Date: 09.04.14
 */
public class MCMicroPrecision extends Func.Stub implements TargetFunc {
  protected final Vec target;
  private final DataSet<?> owner;

  public MCMicroPrecision(final Vec target, DataSet<?> owner) {
    this.target = target;
    this.owner = owner;
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

  @Override
  public DataSet<?> owner() {
    return owner;
  }
}
