package com.spbsu.ml.loss;

import com.spbsu.commons.math.MathTools;
import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.ml.data.set.DataSet;

import static com.spbsu.commons.math.MathTools.EPSILON;
import static java.lang.Math.log;

/**
 * User: solar
 * Date: 21.12.2010
 * Time: 22:37:55
 */
public class JNL2 extends L2 {
  public JNL2(final Vec target, final DataSet<?> base) {
    super(target, base);
  }

  @Override
  public double value(final MSEStats stats) {
    return stats.weight > 1 ? stats.sum/stats.weight : 0;
  }

  @Override
  public double score(final MSEStats stats) {
    final double D = stats.sum2 * MathTools.sqr((stats.weight + 1) / stats.weight) - MathTools.sqr(stats.sum / stats.weight) * (stats.weight + 2);
    final double oldD = stats.sum2;// * MathTools.sqr(stats.weight / (stats.weight - 1));
    return stats.weight > EPSILON ? (D - oldD) : Double.POSITIVE_INFINITY;
  }

  @Override
  public double bestIncrement(MSEStats stats) {
    return stats.sum / stats.weight * (stats.weight - 1) / stats.weight;
  }
}
