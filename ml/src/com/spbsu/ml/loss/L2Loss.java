package com.spbsu.ml.loss;

import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.commons.math.vectors.impl.ArrayVec;
import com.spbsu.ml.Model;
import com.spbsu.ml.data.DSIterator;
import com.spbsu.ml.data.DataSet;

/**
 * User: solar
 * Date: 21.12.2010
 * Time: 22:37:55
 */
public class L2Loss implements LossFunction {
  @Override
  public Vec gradient(Vec point, DataSet learn) {
    double[] result = new double[learn.power()];
    final DSIterator it = learn.iterator();
    int index = 0;
    while(it.advance()) {
      result[index] = it.y() - point.get(index);
      index++;
    }
    return new ArrayVec(result);
  }

  public double value(Model model, DataSet set) {
        double loss = 0;
        final int count = set.power();
        final DSIterator it = set.iterator();
        while(it.advance()) {
            double v = model.value(it.x()) - it.y();
            loss += v * v;
        }
        return Math.sqrt(loss/count);
    }
}
