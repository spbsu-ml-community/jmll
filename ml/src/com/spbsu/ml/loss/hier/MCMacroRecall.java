package com.spbsu.ml.loss.hier;

import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.ml.data.DataSet;
import com.spbsu.ml.data.impl.Hierarchy;
import gnu.trove.iterator.TIntIterator;
import gnu.trove.map.hash.TIntIntHashMap;

/**
 * User: qdeee
 * Date: 14.03.14
 */
public class MCMacroRecall extends HierLoss {
  public MCMacroRecall(Hierarchy unfilledHierarchy, DataSet dataSet, int minEntries) {
    super(unfilledHierarchy, dataSet, minEntries);
  }

  public MCMacroRecall(HierLoss learningLoss, Vec testTarget) {
    super(learningLoss, testTarget);
  }

  @Override
  public double value(Vec x) {
    TIntIntHashMap id2tp = new TIntIntHashMap();
    TIntIntHashMap id2fn = new TIntIntHashMap();
    for (int i = 0; i < target.dim(); i++) {
      int expected = (int) target.get(i);
      int actual = (int) x.get(i);
      if (actual == expected)
        id2tp.adjustOrPutValue(expected, 1, 1);
      else
        id2fn.adjustOrPutValue(expected, 1, 1);
    }

    double result = 0.;
    int nonEmpty = 0;
    for (TIntIterator iter = nodesClasses.iterator(); iter.hasNext(); ) {
      int cls = iter.next();
      int tp = id2tp.get(cls);
      int fn = id2fn.get(cls);
      if (tp + fn != 0) {
        nonEmpty++;
        result += tp / (0. + tp + fn);
      }
    }
    return result / nonEmpty;
  }
}
