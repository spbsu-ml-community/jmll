package com.spbsu.ml.loss.hier;

import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.ml.data.DataSet;
import com.spbsu.ml.data.impl.Hierarchy;
import gnu.trove.iterator.TIntIterator;
import gnu.trove.map.hash.TIntIntHashMap;

/**
 * User: qdeee
 * Date: 07.03.14
 * Info: precision loss for hierarchical classification
 */
public class MCMacroPrecision extends HierLoss {
  public MCMacroPrecision(Hierarchy unfilledHierarchy, DataSet dataSet, int minEntries) {
    super(unfilledHierarchy, dataSet, minEntries);
  }

  public MCMacroPrecision(HierLoss learningLoss, Vec testTarget) {
    super(learningLoss, testTarget);
  }

  @Override
  public double value(Vec x) {
    TIntIntHashMap id2tp = new TIntIntHashMap();
    TIntIntHashMap id2fp = new TIntIntHashMap();
    for (int i = 0; i < x.dim(); i++) {
      int expected = (int) target.get(i);
      int actual = (int) x.get(i);
      if (actual == expected)
        id2tp.adjustOrPutValue(actual, 1, 1);
      else
        id2fp.adjustOrPutValue(actual, 1, 1);
    }

    double result = 0.;
    int nonEmpty = 0;
    for (TIntIterator iter = nodesClasses.iterator(); iter.hasNext(); ) {
      int cls = iter.next();
      int tp = id2tp.get(cls);
      int fp = id2fp.get(cls);
      if (tp + fp != 0) {
        nonEmpty++;
        result += tp / (0. + tp + fp);
      }
    }
    return result / nonEmpty;
  }
}
