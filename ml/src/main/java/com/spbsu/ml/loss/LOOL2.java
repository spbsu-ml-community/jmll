package com.spbsu.ml.loss;

import com.spbsu.commons.func.Computable;
import com.spbsu.commons.math.MathTools;
import com.spbsu.commons.math.vectors.Vec;

/**
 * User: solar
 * Date: 21.12.2010
 * Time: 22:37:55
 */
public class LOOL2 extends L2 {
  public static final Computable<Vec, LOOL2> FACTORY = new Computable<Vec, LOOL2>() {
    @Override
    public LOOL2 compute(Vec argument) {
      return new LOOL2(argument);
    }
  };

  public LOOL2(Vec target) {
    super(target);
  }

  @Override
  public double value(MSEStats stats) {
    return stats.weight > 1 ? stats.sum/stats.weight : 0;
  }

  @Override
  public double score(MSEStats stats) {
    return stats.weight > 1 ? (stats.sum2 - stats.sum * stats.sum / stats.weight) * MathTools.sqr(stats.weight / (stats.weight - 1.)) : stats.sum2;
  }
}
