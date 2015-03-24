package com.spbsu.ml.loss;

import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.ml.data.set.DataSet;

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
    return stats.weight >= 1 ? stats.sum / stats.weight : 0;
  }

  @Override
  public double score(final MSEStats stats) {
    final double weight = stats.weight;
    final double sum = stats.sum;
    return weight > 2 ? (-sum * sum / weight) * weight * (weight - 2) / (weight * weight - 3 * weight + 1) * (1 + 2 * Math.log(weight + 1)) : 0;
  }
}
