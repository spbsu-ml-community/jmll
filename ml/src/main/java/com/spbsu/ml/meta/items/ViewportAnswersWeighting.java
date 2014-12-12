package com.spbsu.ml.meta.items;

import java.util.Arrays;
import java.util.Comparator;


import com.spbsu.commons.math.vectors.VecIterator;
import com.spbsu.commons.math.vectors.VecTools;
import com.spbsu.commons.math.vectors.impl.vectors.CommonBasisVec;
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
    final Integer[] order = new Integer[VecTools.l0(answers)];
    int index = 0;
    while (nzIt.advance()) {
      order[index] = nzIt.index();
      index++;
    }

    //    ArrayTools.parallelSort(weights, order);
    Arrays.sort(order, new Comparator<Integer>() {
      @Override
      public int compare(final Integer o1, final Integer o2) {
        final double weight1 = answers.get(o1);
        final double weight2 = answers.get(o2);
        if (weight1 < weight2) {
          return -1;
        } else if (weight1 > weight2) {
          return 1;
        } else {
          final String name1 = answers.basis().fromIndex(o1);
          final String name2 = answers.basis().fromIndex(o2);
          return name1.compareTo(name2);
        }
      }
    });

    this.answers = new String[order.length];
    this.weights = new double[order.length];
    for (int i = 0; i < order.length; i++) {
      final Integer index1 = order[i];
      this.answers[i] = answers.basis().fromIndex(index1);
      this.weights[i] = answers.get(index1);
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
