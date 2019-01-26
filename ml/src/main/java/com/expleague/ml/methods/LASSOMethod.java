package com.expleague.ml.methods;

import com.expleague.commons.math.Trans;
import com.expleague.commons.math.vectors.Mx;
import com.expleague.commons.math.vectors.Vec;
import com.expleague.commons.math.vectors.VecTools;
import com.expleague.commons.math.vectors.impl.vectors.ArrayVec;
import com.expleague.ml.data.set.VecDataSet;
import com.expleague.ml.func.Linear;
import com.expleague.ml.loss.L2;

/**
 * User: solar
 * Date: 27.12.10
 * Time: 18:04
 */
public class LASSOMethod extends VecOptimization.Stub<L2> {
  private final int iterations;
  private final double step;

  public LASSOMethod(final int iterations, final double step) {
    this.iterations = iterations;
    this.step = step;
  }

  @Override
  public Linear fit(final VecDataSet ds, final L2 loss) {
    final Mx learn = ds.data();
    final Vec betas = new ArrayVec(learn.columns());
    final Vec values = VecTools.copy(loss.target);

    for (int t = 0; t < iterations; t++) {
      int bestDirection = 0;
      double sign = 0;
      {
        final double[] correlations = new double[learn.columns()];
        for (int i = 0; i < learn.rows(); i++) {
          for (int j = 0; j < correlations.length; j++) {
            correlations[j] += learn.get(i, j) * values.get(i);
          }
        }
        double corr = Math.abs(correlations[0]);
        for (int i = 1; i < correlations.length; i++) {
          final double current = Math.abs(correlations[i]);
          if (corr < current) {
            corr = current;
            bestDirection = i;
            sign = Math.signum(correlations[i]);
          }
        }
      }
      final double signedStep = step * sign;
      betas.adjust(bestDirection, signedStep);
      {
        for (int i = 0; i < learn.rows(); i++) {
          values.adjust(i, -signedStep * learn.get(i, bestDirection));
        }
      }
    }
    return new Linear(betas);
  }
}
