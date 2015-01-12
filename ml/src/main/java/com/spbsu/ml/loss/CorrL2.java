package com.spbsu.ml.loss;

import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.ml.data.set.DataSet;

/**
 * User: solar
 * Date: 21.12.2010
 * Time: 22:37:55
 */
public class CorrL2 extends L2 {
  public CorrL2(final Vec target, final DataSet<?> owner) {
    super(target, owner);
  }

  @Override
  public double bestIncrement(final MSEStats stats) {
    return stats.weight > 1 ? stats.sum/stats.weight : 0;
  }

  @Override
  public double score(final MSEStats stats) {
    final double n = stats.weight;
    return n > 2 ? n/(n-1) * (stats.sum2 - stats.sum * stats.sum / n) : stats.sum2;
  }
}
