package com.spbsu.ml.func.generic;

import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.commons.math.FuncC1;

/**
* User: solar
* Date: 26.05.15
* Time: 11:45
*/
public class ChainProbSum extends FuncC1.Stub {
  @Override
  public Vec gradientTo(Vec x, Vec to) {
    double prod = 1;
    for (int i = 0; i < x.length(); i++) {
      prod *= (1 - x.get(i));
    }
    double probTail = 0;
    for (int i = x.length() - 1; i >= 0; i--) {
      final double prob = x.get(i);
      prod /= (1 - prob);
      to.set(i, prod - probTail);
      probTail = prob + (1 - prob) * probTail;
    }
    return to;
  }

  @Override
  public double value(Vec x) {
    double result = 0;
    for (int i = 0; i < x.length(); i++) {
      result += (1 - result) * x.at(i);
    }
    return result;
  }

  @Override
  public int dim() {
    return 1;
  }
}
