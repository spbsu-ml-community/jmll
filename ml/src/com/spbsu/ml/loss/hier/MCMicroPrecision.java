package com.spbsu.ml.loss.hier;

import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.ml.data.DataSet;
import com.spbsu.ml.data.impl.Hierarchy;

/**
 * User: qdeee
 * Date: 04.04.14
 */
public class MCMicroPrecision extends HierLoss {
  public MCMicroPrecision(final Hierarchy unfilledHierarchy, final DataSet dataSet, final int minEntries) {
    super(unfilledHierarchy, dataSet, minEntries);
  }

  public MCMicroPrecision(final HierLoss learningLoss, final Vec testTarget) {
    super(learningLoss, testTarget);
  }

  @Override
  public double value(final Vec x) {
    int tp = 0;
    int fp = 0;
    for (int i = 0; i < x.dim(); i++) {
      int expected = (int) target.get(i);
      int actual = (int) x.get(i);
      if (actual == expected)
        tp += 1;
      else
        fp += 1;
    }

    return tp / (tp + fp + 0.);
  }
}
