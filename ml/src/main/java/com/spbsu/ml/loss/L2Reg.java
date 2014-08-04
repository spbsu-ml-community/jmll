package com.spbsu.ml.loss;

import com.spbsu.commons.math.MathTools;
import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.ml.data.set.DataSet;

/**
 * User: solar
 * Date: 21.12.2010
 * Time: 22:37:55
 */
public class L2Reg extends L2 {
  public L2Reg(Vec target, DataSet<?> base) {
    super(target, base);
  }

  @Override
  public double value(MSEStats stats) {
    return stats.weight >= 1 ? stats.sum / stats.weight : 0;
  }

  @Override
  public double score(MSEStats stats) {
    return stats.weight > 1 ? (-stats.sum * stats.sum / stats.weight) * MathTools.sqr(stats.weight / (stats.weight - 1.)) * (1 + 2 * Math.log(stats.weight + 1)) : 0;
  }
}
