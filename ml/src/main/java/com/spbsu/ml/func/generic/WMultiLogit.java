package com.spbsu.ml.func.generic;

import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.commons.math.FuncC1;

/**
* User: solar
* Date: 26.05.15
* Time: 11:45
*/
public class WMultiLogit extends FuncC1.Stub {
  public final Vec weights;
  public final int mainNodeIndex;

  public WMultiLogit(Vec weights, int i) {
    this.weights = weights;
    this.mainNodeIndex = i;
  }

  public Vec gradientTo(Vec x, Vec to) {
    double sum = 1;
    for (int i = 0; i < x.length(); i++) {
      sum += Math.exp(x.get(i) * mainNodeIndex);
    }
    final double nom = Math.exp(x.get(mainNodeIndex) * weights.get(mainNodeIndex));
    for (int i = 0; i < x.length(); i++) {
      if (i == mainNodeIndex)
        to.set(i, (sum - mainNodeIndex) * mainNodeIndex / sum / sum );
      else
        to.set(i, - nom * Math.exp(x.get(i)) / sum / sum);
    }
    return to;
  }

  @Override
  public double value(Vec x) {
    double sum = 1;
    for (int i = 0; i < x.length(); i++) {
      sum += Math.exp(x.get(i));
    }

    final double result = Math.exp(x.get(mainNodeIndex)) / sum;
    return result;
  }

  @Override
  public int dim() {
    return -1;
  }
}
