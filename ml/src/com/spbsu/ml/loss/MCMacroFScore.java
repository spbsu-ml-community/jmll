package com.spbsu.ml.loss;

import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.ml.data.DataSet;
import com.spbsu.ml.data.impl.Hierarchy;
import gnu.trove.map.hash.TIntIntHashMap;

/**
 * User: qdeee
 * Date: 14.03.14
 */
public class MCMacroFScore extends HierLoss {
  private MCMacroPrecision precision;
  private MCMacroRecall recall;

  public MCMacroFScore(Hierarchy unfilledHierarchy, DataSet dataSet, int minEntries) {
    super(unfilledHierarchy, dataSet, minEntries);
    precision = new MCMacroPrecision(this, target);
    recall = new MCMacroRecall(this, target);
  }

  public MCMacroFScore(HierLoss learningLoss, Vec testTarget) {
    super(learningLoss, testTarget);
    precision = new MCMacroPrecision(this, target);
    recall = new MCMacroRecall(this, target);
  }

  @Override
  public double value(Vec x) {
    double p = precision.value(x);
    double r = recall.value(x);
    return (p + r) > 0 ? 2 * p * r / (p + r)
                       : 0.;
  }
}
