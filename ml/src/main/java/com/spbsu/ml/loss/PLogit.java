package com.spbsu.ml.loss;

import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.ml.data.set.DataSet;

/**
 * User: starlight
 * Date: 25.12.13
 */
public class PLogit extends LLLogit {
  private final Vec target;

  public PLogit(final Vec target, final DataSet<?> base) {
    super(target, base);
    this.target = target;
  }

  @Override
  public double value(final Vec point) {
    int truePositive = 0;
    int falsePositive = 0;

    for (int i = 0; i < point.dim(); i++) {
      if (point.get(i) > 0 && target.get(i) > 0) {
        truePositive++;
      } else if (point.get(i) > 0 && target.get(i) <= 0) {
        falsePositive++;
      }
    }

    return truePositive / (0. + truePositive + falsePositive);
  }
}
