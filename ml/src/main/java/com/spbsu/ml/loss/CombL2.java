package com.spbsu.ml.loss;

import com.spbsu.commons.math.MathTools;
import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.ml.data.set.DataSet;

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
  public double bestIncrement(MSEStats stats) {
    return stats.weight > 2 ? stats.sum/stats.weight : 0;
  }

  @Override
  public double score(MSEStats stats) {
    final double n = stats.weight;
    final double deltaD = stats.weight > 1 ? (-stats.sum * stats.sum / stats.weight) * MathTools.sqr(stats.weight / (stats.weight - 1.)) : 0;
    return deltaD * (1 + 2 * log(n + 1));
  }
}
