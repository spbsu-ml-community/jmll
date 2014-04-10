package com.spbsu.ml.loss.multiclass.hier.impl;

import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.ml.data.DataSet;
import com.spbsu.ml.data.impl.Hierarchy;
import com.spbsu.ml.loss.multiclass.MCMacroPrecision;
import com.spbsu.ml.loss.multiclass.hier.HierLoss;
import gnu.trove.iterator.TIntIterator;
import gnu.trove.map.hash.TIntIntHashMap;

/**
 * User: qdeee
 * Date: 07.03.14
 * Info: precision loss for hierarchical classification
 */
public class HMCMacroPrecision extends HierLoss {
  private final MCMacroPrecision precision;

  public HMCMacroPrecision(Hierarchy unfilledHierarchy, DataSet dataSet, int minEntries) {
    super(unfilledHierarchy, dataSet, minEntries);
    precision = new MCMacroPrecision(target);
  }

  public HMCMacroPrecision(HierLoss learningLoss, Vec testTarget) {
    super(learningLoss, testTarget);
    precision = new MCMacroPrecision(target);
  }

  @Override
  public double value(Vec x) {
    return precision.value(x);
  }
}
