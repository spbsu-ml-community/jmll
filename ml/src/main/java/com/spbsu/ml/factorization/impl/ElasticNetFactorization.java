package com.spbsu.ml.factorization.impl;

import com.spbsu.commons.func.impl.WeakListenerHolderImpl;
import com.spbsu.commons.math.vectors.Mx;
import com.spbsu.commons.math.vectors.MxTools;
import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.commons.math.vectors.VecTools;
import com.spbsu.commons.math.vectors.impl.vectors.ArrayVec;
import com.spbsu.commons.util.Pair;
import com.spbsu.ml.factorization.OuterFactorization;

/**
 * User: qdeee
 * Date: 25.02.15
 */
public class ElasticNetFactorization extends WeakListenerHolderImpl<Pair<Vec, Vec>> implements OuterFactorization {
  private final int iters;
  private final double tolerance;
  private final double alpha;
  private final double lambda;

  public ElasticNetFactorization(final int iters, final double tolerance, final double alpha, final double lambda) {
    this.iters = iters;
    this.tolerance = tolerance;
    this.alpha = alpha;
    this.lambda = lambda;
  }

  @Override
  public Pair<Vec, Vec> factorize(final Mx X) {
    final Vec currentU = VecTools.fill(new ArrayVec(X.rows()), 1.0);
    final Vec currentV = VecTools.fill(new ArrayVec(X.columns()), 1.0);
    final Mx transposeX = MxTools.transpose(X);

    for (int iter = 0; iter < iters; iter++) {
      //fix v, search u:
      VecTools.assign(currentU, findElasticNetSolution(currentV, X, tolerance, alpha, lambda, 2));

      //fix u, search v:
      VecTools.assign(currentV, findElasticNetSolution(currentU, transposeX, tolerance, alpha, lambda, 2));

      invoke(Pair.create(currentU, currentV));
    }
    return Pair.create(currentU, currentV);
  }

  private static Vec findElasticNetSolution(
      final Vec vec2fix,
      final Mx rightSideMx,
      final double tolerance,
      final double alpha,
      final double lambda,
      final int checkIterations
  ) {
    final int columns = rightSideMx.dim() / vec2fix.dim();
    final double selfFeatureProduct = VecTools.multiply(vec2fix, vec2fix);
    final double[] gradient = new double[columns];
    for (int j = 0; j < columns; j++) {
      gradient[j] = VecTools.multiply(vec2fix, rightSideMx.row(j));
    }

    boolean anyUpdated = true;
    Vec prev = new ArrayVec(columns);
    Vec betas = new ArrayVec(columns);
    while (anyUpdated) {
      anyUpdated = false;
      VecTools.assign(prev, betas);
      for (int i = 0; i < checkIterations; i++) {
        for (int k = 0; k < columns; k++) {
          final int N = rightSideMx.dim();
          double newBeta = gradient[k];
          newBeta = softThreshold(newBeta, N * lambda * alpha);
          newBeta /= (selfFeatureProduct + N * lambda * (1 - alpha));
          if (Math.abs(newBeta - betas.get(k)) > 1e-15) {
            betas.set(k, newBeta);
            anyUpdated = true;
          }
        }

        if (!anyUpdated) {
          break;
        }
      }

      if (VecTools.distance(betas, prev) < tolerance) {
        break;
      }
    }

    return betas;
  }

  private static double softThreshold(final double z, final double j) {
    final double sgn = Math.signum(z);
    return sgn * Math.max(sgn * z - j, 0);
  }
}
