package com.expleague.ml.loss;

import com.expleague.commons.math.vectors.Vec;
import com.expleague.ml.data.set.DataSet;

import static com.expleague.commons.math.MathTools.sqr;

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
  public double value(final Stat stats) {
    return stats.weight >= 1 ? stats.sum / stats.weight : 0;
  }

  @Override
  public double score(final Stat stats) {
    return stats.weight > 1 ? (-stats.sum * stats.sum / stats.weight) *sqr(stats.weight / (stats.weight - 1))
            : 0;
  }
}
