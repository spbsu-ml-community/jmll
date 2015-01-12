package com.spbsu.ml.loss.blockwise;

import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.ml.data.set.DataSet;

/**
 * User: qdeee
 * Date: 21.07.14
 */
public class BlockwiseSatL2 extends BlockwiseL2 {
  public BlockwiseSatL2(final Vec target, final DataSet<?> owner) {
    super(target, owner);
  }

  @Override
  public double bestIncrement(final BlockwiseL2.MSEStats stats) {
    return stats.weight > 2 ? stats.sum/stats.weight : 0;
  }

  @Override
  public double score(final BlockwiseL2.MSEStats stats) {
    final double n = stats.weight;
    return n > 2 ? n*(n-2)/(n * n - 3 * n + 1) * (stats.sum2 - stats.sum * stats.sum / n) : stats.sum2;
  }
}
