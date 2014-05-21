package com.spbsu.ml.loss.multiclass.hier.impl;

import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.ml.data.DataSet;
import com.spbsu.ml.data.impl.HierarchyTree;
import com.spbsu.ml.loss.multiclass.MCMicroPrecision;
import com.spbsu.ml.loss.multiclass.hier.HierLoss;

/**
 * User: qdeee
 * Date: 04.04.14
 */
public class HMCMicroPrecision extends HierLoss {
  private final MCMicroPrecision precision;

  public HMCMicroPrecision(final HierarchyTree unfilledHierarchy, final DataSet dataSet, final int minEntries) {
    super(unfilledHierarchy, dataSet, minEntries);
    precision = new MCMicroPrecision(target);
  }

  public HMCMicroPrecision(final HierLoss learningLoss, final Vec testTarget) {
    super(learningLoss, testTarget);
    precision = new MCMicroPrecision(target);
  }

  @Override
  public double value(final Vec x) {
    return precision.value(x);
  }
}
