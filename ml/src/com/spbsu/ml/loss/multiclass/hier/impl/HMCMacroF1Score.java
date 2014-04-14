package com.spbsu.ml.loss.multiclass.hier.impl;

import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.ml.data.DataSet;
import com.spbsu.ml.data.impl.Hierarchy;
import com.spbsu.ml.loss.multiclass.MCMacroF1Score;
import com.spbsu.ml.loss.multiclass.MCMacroPrecision;
import com.spbsu.ml.loss.multiclass.MCMacroRecall;
import com.spbsu.ml.loss.multiclass.hier.HierLoss;

/**
 * User: qdeee
 * Date: 14.03.14
 */
public class HMCMacroF1Score extends HierLoss {
  private final MCMacroF1Score f1Score;

  public HMCMacroF1Score(Hierarchy unfilledHierarchy, DataSet dataSet, int minEntries) {
    super(unfilledHierarchy, dataSet, minEntries);
    f1Score = new MCMacroF1Score(target);
  }

  public HMCMacroF1Score(HierLoss learningLoss, Vec testTarget) {
    super(learningLoss, testTarget);
    f1Score = new MCMacroF1Score(target);
  }

  @Override
  public double value(Vec x) {
    return f1Score.value(x);
  }
}
