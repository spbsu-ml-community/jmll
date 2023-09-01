package com.expleague.ml.loss;

import com.expleague.commons.func.AdditiveStatistics;
import com.expleague.commons.math.FuncC1;
import com.expleague.commons.math.MathTools;
import com.expleague.commons.math.vectors.Vec;
import com.expleague.commons.math.vectors.VecTools;
import com.expleague.commons.math.vectors.impl.vectors.ArrayVec;
import com.expleague.ml.TargetFunc;
import com.expleague.ml.data.set.DataSet;

import java.util.function.IntFunction;

/**
 * Created by irlab on 22.02.2015.
 */
public class WeightedSatL2 extends WeightedL2 {
  public WeightedSatL2(final DataSet<?> owner, final Vec targets, final Vec weights) {
    super(owner, targets, weights);
  }

  @Override
  public double bestIncrement(final Stat stats) {
    return stats.weight > 2 ? stats.sum/stats.weight : 0;
  }

  @Override
  public double score(final Stat stats) {
    final double n = stats.weight;
    return n > 2 ? n*(n-2)/(n * n - 3 * n + 1) * (stats.sum2 - stats.sum * stats.sum / n) : stats.sum2;
  }
}
