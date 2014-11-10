package com.spbsu.ml.loss;

import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.commons.math.vectors.VecTools;
import com.spbsu.ml.data.set.DataSet;

/**
 * User: solar
 * Date: 21.12.2010
 * Time: 22:37:55
 */
public class L2GreedyTDRegion extends L2 {
  final double norm;

  public L2GreedyTDRegion(Vec target, DataSet<?> base) {
    super(target, base);
    norm = VecTools.norm(target);
  }

  @Override
  public double value(MSEStats stats) {
    return stats.weight >= 1 ? stats.sum / stats.weight : 0;
  }

  @Override
  public double score(MSEStats stats) {
    return stats.weight > 1 ? (-stats.sum * stats.sum / stats.weight) * stats.weight * (stats.weight - 2) / (stats.weight * stats.weight - 3 * stats.weight + 1) : stats.sum2;
  }


}
