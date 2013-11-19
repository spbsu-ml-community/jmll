package com.spbsu.ml.loss;

import com.spbsu.commons.math.vectors.Vec;

/**
 * User: solar
 * Date: 21.12.2010
 * Time: 22:37:55
 */
public class LOOL2W extends L2W {
  public LOOL2W(Vec target, Vec weight) {
    super(target, weight);
  }

  @Override
  public double score(MSEStats stats) {
    return Math.sqrt(stats.sum2 - stats.sum * stats.sum / stats.weight) / (stats.weight - 1);
  }
}
