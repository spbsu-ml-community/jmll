package com.spbsu.ml.loss;

import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.ml.data.set.DataSet;

import static com.spbsu.commons.math.MathTools.sqr;

/**
 * User: solar
 * Date: 21.12.2010
 * Time: 22:37:55
 */
public class L2GreedyTDRegion extends L2 {

  public L2GreedyTDRegion(final Vec target, final DataSet<?> base) {
    super(target, base);
  }

  @Override
  public double value(final MSEStats stats) {
    return stats.weight >= 1 ? stats.sum / stats.weight : 0;
  }

  @Override
  public double score(final MSEStats stats) {
    return stats.weight > 1 ? (-stats.sum * stats.sum / stats.weight) *sqr(stats.weight / (stats.weight - 1))
            : 0;
  }
}
