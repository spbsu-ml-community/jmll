package com.spbsu.ml.loss;

import com.spbsu.commons.func.Computable;
import com.spbsu.commons.math.vectors.Vec;

/**
 * User: solar
 * Date: 21.12.2010
 * Time: 22:37:55
 */
public class LogL2 extends L2 {
  public static final Computable<Vec, LogL2> FACTORY = new Computable<Vec, LogL2>() {
    @Override
    public LogL2 compute(Vec argument) {
      return new LogL2(argument);
    }
  };

  public LogL2(Vec target) {
    super(target);
  }

  @Override
  public double value(MSEStats stats) {
    return stats.weight >= 1 ? stats.sum/stats.weight : 0;
  }

  @Override
  public double score(MSEStats stats) {
    return stats.weight > 1 ? (- stats.sum * stats.sum / stats.weight) * Math.log(stats.weight + 1) : 0;
  }
}
