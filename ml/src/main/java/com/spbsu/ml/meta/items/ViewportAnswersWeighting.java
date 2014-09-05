package com.spbsu.ml.meta.items;

import com.spbsu.commons.math.vectors.VecIterator;
import com.spbsu.commons.math.vectors.VecTools;
import com.spbsu.commons.math.vectors.impl.vectors.CommonBasisVec;
import com.spbsu.commons.math.vectors.impl.vectors.DVector;
import com.spbsu.commons.util.ArrayTools;
import com.spbsu.ml.meta.DSItem;

/**
 * User: solar
 * Date: 18.07.14
 * Time: 17:01
 */
public class ViewportAnswersWeighting extends DSItem.Stub implements DSItem {
  public String reqId;
  public String vpName;
  public String[] answers;
  public double[] weights;

  public ViewportAnswersWeighting(final String reqId, String vpName, final CommonBasisVec<String> answers) {
    this.vpName = vpName;
    final VecIterator nzIt = answers.nonZeroes();
    int[] order = new int[VecTools.l0(answers)];
    this.weights = new double[order.length];
    int index = 0;
    while (nzIt.advance()) {
      order[index] = nzIt.index();
      this.weights[index] = nzIt.value();
      index++;
    }
    ArrayTools.parallelSort(weights, order);
    this.answers = new String[order.length];
    for (int i = 0; i < order.length; i++) {
      this.answers[i] = answers.basis().fromIndex(order[i]);
    }
    this.reqId = reqId;
  }

  public ViewportAnswersWeighting() {
  }

  @Override
  public String id() {
    return reqId;
  }
}
