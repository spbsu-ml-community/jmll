package com.spbsu.ml.loss.hier;

import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.ml.data.DataSet;
import com.spbsu.ml.data.impl.Hierarchy;

/**
 * User: qdeee
 * Date: 14.03.14
 */
public class MCMacroF1Score extends HierLoss {
  private MCMacroPrecision precision;
  private MCMacroRecall recall;

  public MCMacroF1Score(Hierarchy unfilledHierarchy, DataSet dataSet, int minEntries) {
    super(unfilledHierarchy, dataSet, minEntries);
    precision = new MCMacroPrecision(this, target);
    recall = new MCMacroRecall(this, target);
  }

  public MCMacroF1Score(HierLoss learningLoss, Vec testTarget) {
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
