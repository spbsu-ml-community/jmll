package com.spbsu.ml.loss.hier;

import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.ml.data.DataSet;
import com.spbsu.ml.data.impl.Hierarchy;

/**
 * User: qdeee
 * Date: 07.04.14
 */
public class MCMicroF1Score extends HierLoss {
  private MCMicroPrecision precision;
  private MCMicroRecall recall;

  public MCMicroF1Score(final Hierarchy unfilledHierarchy, final DataSet dataSet, final int minEntries) {
    super(unfilledHierarchy, dataSet, minEntries);
    precision = new MCMicroPrecision(this, target);
    recall = new MCMicroRecall(this, target);
  }

  public MCMicroF1Score(final HierLoss learningLoss, final Vec testTarget) {
    super(learningLoss, testTarget);
    precision = new MCMicroPrecision(this, target);
    recall = new MCMicroRecall(this, target);
  }

  @Override
  public double value(Vec x) {
    double p = precision.value(x);
    double r = recall.value(x);
    return (p + r) > 0 ? 2 * p * r / (p + r)
        : 0.;
  }
}
