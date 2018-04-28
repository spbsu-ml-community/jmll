package com.expleague.ml.loss;

import com.expleague.commons.math.MathTools;
import com.expleague.commons.math.vectors.Vec;
import com.expleague.ml.data.set.DataSet;

import static java.lang.Math.log;

/**
 * User: solar
 * Date: 21.12.2010
 * Time: 22:37:55
 */
public class CombL2 extends L2 {
  public CombL2(final Vec target, final DataSet<?> owner) {
    super(target, owner);
  }

  @Override
  public double bestIncrement(Stat stats) {
    return stats.weight > 2 ? stats.sum/stats.weight : 0;
  }

  @Override
  public double score(Stat stats) {
    final double n = stats.weight;
    final double deltaD = stats.weight > 1 ? (-stats.sum * stats.sum / stats.weight) * MathTools.sqr(stats.weight / (stats.weight - 1.)) : 0;
    return deltaD * (1 + 2 * log(n + 1));
  }
}
