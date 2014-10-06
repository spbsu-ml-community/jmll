package com.spbsu.ml.loss;

import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.ml.data.set.DataSet;

/**
 * User: solar
 * Date: 21.12.2010
 * Time: 22:37:55
 */
public class L2GreedyRegion extends L2 {
  public L2GreedyRegion(Vec target, DataSet<?> base) {
    super(target, base);
  }

  @Override
  public double value(MSEStats stats) {
    return stats.weight >= 1 ? stats.sum / stats.weight : 0;
  }

  @Override
  public double score(MSEStats stats) {
    final double n = stats.weight;
    if (n <= 6)
      return Double.POSITIVE_INFINITY;

//    return n*(n-2)/(n * n - 3 * n + 1) * (- stats.sum * stats.sum * n);
    return (-stats.sum * stats.sum * n / (n - 1) / (n - 1));

//    return stats.weight > 1 ? (-stats.sum * stats.sum / stats.weight) * stats.weight * (stats.weight - 2)
//            / (stats.weight * stats.weight - 3 * stats.weight + 1) * (1 + 2 * Math.log(stats.weight + 1)) : 0;
  }
}
