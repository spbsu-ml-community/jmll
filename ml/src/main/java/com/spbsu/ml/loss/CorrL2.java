package com.spbsu.ml.loss;

import com.spbsu.commons.math.vectors.Vec;

/**
 * User: solar
 * Date: 21.12.2010
 * Time: 22:37:55
 */
public class CorrL2 extends L2 {
  public CorrL2(Vec target) {
    super(target);
  }

  @Override
  public double bestIncrement(MSEStats stats) {
    return stats.weight > 1 ? stats.sum/stats.weight : 0;
  }

  @Override
  public double score(MSEStats stats) {
    final double n = stats.weight;
    return n > 2 ? n/(n-1) * (stats.sum2 - stats.sum * stats.sum / n) : stats.sum2;
  }
}