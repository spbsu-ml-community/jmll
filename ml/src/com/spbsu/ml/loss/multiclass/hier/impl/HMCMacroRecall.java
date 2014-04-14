package com.spbsu.ml.loss.multiclass.hier.impl;

import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.ml.data.DataSet;
import com.spbsu.ml.data.impl.Hierarchy;
import com.spbsu.ml.loss.multiclass.MCMacroRecall;
import com.spbsu.ml.loss.multiclass.hier.HierLoss;
import gnu.trove.iterator.TIntIterator;
import gnu.trove.map.hash.TIntIntHashMap;

/**
 * User: qdeee
 * Date: 14.03.14
 */
public class HMCMacroRecall extends HierLoss {
  private final MCMacroRecall recall;

  public HMCMacroRecall(Hierarchy unfilledHierarchy, DataSet dataSet, int minEntries) {
    super(unfilledHierarchy, dataSet, minEntries);
    recall = new MCMacroRecall(target);
  }

  public HMCMacroRecall(HierLoss learningLoss, Vec testTarget) {
    super(learningLoss, testTarget);
    recall = new MCMacroRecall(target);
  }

  @Override
  public double value(Vec x) {
    return recall.value(x);
  }
}
