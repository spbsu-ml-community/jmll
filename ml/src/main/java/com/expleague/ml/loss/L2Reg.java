package com.expleague.ml.loss;

import com.expleague.commons.math.MathTools;
import com.expleague.commons.math.vectors.Vec;
import com.expleague.ml.data.set.DataSet;

/**
 * User: solar
 * Date: 21.12.2010
 * Time: 22:37:55
 */
public class L2Reg extends L2 {
  public L2Reg(final Vec target, final DataSet<?> base) {
    super(target, base);
  }

  @Override
  public double value(final MSEStats stats) {
    return stats.weight >= 1 ? stats.sum / (stats.weight + 1): 0;
  }

  @Override
  public double bestIncrement(MSEStats stats) {
    return stats.weight > MathTools.EPSILON ? stats.sum / (stats.weight + 1) : 0;
  }

  @Override
  public double score(final MSEStats stats) {
    final double weight = stats.weight;
    return stats.weight > 1 ? (- stats.sum * stats.sum / stats.weight) * (1 + 2 * Math.log(weight + 1)) * MathTools.sqr(stats.weight / (stats.weight - 1.)) : 0;
  }
}
