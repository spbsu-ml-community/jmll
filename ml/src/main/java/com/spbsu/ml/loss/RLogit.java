package com.spbsu.ml.loss;

import com.spbsu.commons.math.vectors.Vec;

/**
 * User: starlight
 * Date: 25.12.13
 */
public class RLogit extends LLLogit {
  private final Vec target;

  public RLogit(final Vec target) {
    super(target);
    this.target = target;
  }

  @Override
  public double value(final Vec point) {
    int truePositive = 0;
    int falseNegative = 0;

    for (int i = 0; i < point.dim(); i++) {
      if (point.get(i) > 0 && target.get(i) > 0) {
        truePositive++;
      } else if (point.get(i) <= 0 && target.get(i) > 0) {
        falseNegative++;
      }
    }

    return truePositive / (0. + truePositive + falseNegative);
  }
}
