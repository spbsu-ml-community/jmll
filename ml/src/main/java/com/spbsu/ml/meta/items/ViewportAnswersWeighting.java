package com.spbsu.ml.meta.items;

import com.spbsu.commons.math.vectors.VecIterator;
import com.spbsu.commons.math.vectors.VecTools;
import com.spbsu.commons.math.vectors.impl.vectors.DVector;
import com.spbsu.ml.meta.DSItem;

/**
 * User: solar
 * Date: 18.07.14
 * Time: 17:01
 */
public class ViewportAnswersWeighting implements DSItem {
  public String reqId;
  public String vpName;
  public String[] answers;
  public double[] weights;

  public ViewportAnswersWeighting(final String reqId, String vpName, final DVector<String> answers) {
    this.vpName = vpName;
    final VecIterator nzIt = answers.nonZeroes();
    this.answers = new String[VecTools.l0(answers)];
    this.weights = new double[this.answers.length];
    int index = 0;
    while (nzIt.advance()) {
      this.answers[index] = answers.basis().fromIndex(nzIt.index());
      this.weights[index] = nzIt.value();
      index++;
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
