package com.spbsu.ml.methods;

import com.spbsu.commons.math.vectors.Mx;
import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.commons.math.vectors.VecTools;
import com.spbsu.commons.math.vectors.impl.vectors.ArrayVec;
import com.spbsu.ml.Trans;
import com.spbsu.ml.data.set.VecDataSet;
import com.spbsu.ml.func.Linear;
import com.spbsu.ml.loss.L2;

import static com.spbsu.commons.math.vectors.VecTools.copy;

/**
 * User: noxoomo
 * Date: 5.02.2015
 * Time: 13:58
 * Pathwise coordinate descent by (for more details see articles by Friedman, Hastie, Tibshirani)
 */


public class ElasticNetMethod extends VecOptimization.Stub<L2> {

  private double softThreshold(final double z, final double j) {
    final double sgn = Math.signum(z);
    return sgn * Math.max(sgn * z - j, 0);
  }

  private final double tolerance;
  private final double alpha;
  private final double lambda;
  private final int iterations = 5;
  private  Vec init;

  public ElasticNetMethod(final double tolerance, final double alpha, final double lambda) {
    this.tolerance = tolerance;
    this.alpha = alpha;
    this.lambda = lambda;
    this.init = null;
  }
  public ElasticNetMethod(final double tolerance, double alpha, double lambda, Vec init) {
    this.tolerance = tolerance;
    this.alpha = alpha;
    this.lambda = lambda;
    this.init = init;
  }

  @Override
  public Trans fit(final VecDataSet ds, final L2 loss) {
    final Mx learn = ds.data();
    final Vec betas = init != null ? copy(init) : new ArrayVec(learn.columns());
    final Vec target = loss.target;
    double[][] featuresProducts = new double[betas.dim()][];
    for (int i=0; i < betas.dim();++i) {
        featuresProducts[i] = new double[i + 1];
      featuresProducts[i][i] = VecTools.multiply(learn.col(i), learn.col(i));
    }
    boolean[] cached = new boolean[featuresProducts.length];

    double[] targetProduct = new double[learn.columns()];
    {
      for (int i = 0; i < learn.columns(); ++i) {
        targetProduct[i] = VecTools.multiply(target, learn.col(i));
      }
    }
    Vec prev = copy(betas);

    for (int k=0; k < betas.dim();++k) {
      if (betas.get(k) != 0 && !cached[k]) {
        for (int j = 0; j < k + 1; ++j) {
          featuresProducts[k][j] = VecTools.multiply(learn.col(k), learn.col(j));
        }
        for (int j = k+1; j < betas.dim(); ++j) {
          if (!cached[j])
            featuresProducts[j][k] = VecTools.multiply(learn.col(k), learn.col(j));
        }
        cached[k] = true;
      }
    }

    double[] cachedGradient = new double[betas.dim()];
    {
      for (int i = 0; i < betas.dim(); ++i) {
        double res = targetProduct[i];
        for (int j = 0; j < i; ++j) {
          res -= betas.get(j) * featuresProducts[i][j];
        }
        for (int j = i + 1; j < betas.dim(); ++j) {
          res -= betas.get(j) * featuresProducts[j][i];
        }
        cachedGradient[i] = res;
      }
    }

    while (true) {
      final int N = learn.columns();
      for (int t = 0; t < iterations; t++) {
        for (int k = 0; k < betas.dim(); ++k) {
          double newBeta = cachedGradient[k];
          newBeta = softThreshold(newBeta, N * lambda * alpha);
          if (newBeta != 0) {
            if (!cached[k]) {
                for (int j = 0; j < k + 1; ++j) {
                  featuresProducts[k][j] = VecTools.multiply(learn.col(k), learn.col(j));
                }
              for (int j = k+1; j < betas.dim(); ++j) {
                if (!cached[j])
                  featuresProducts[j][k] = VecTools.multiply(learn.col(k), learn.col(j));
              }
                cached[k] = true;
            }
            newBeta /= (featuresProducts[k][k] + N * lambda * (1 - alpha));
            final double beta = betas.get(k);
            for (int i = 0; i < k; ++i) {
              cachedGradient[i] -= (newBeta-beta) * featuresProducts[k][i];
            }
            for (int i = k + 1; i < betas.dim(); ++i) {
              cachedGradient[i] -=(newBeta-beta) * featuresProducts[i][k];
            }
            betas.set(k, newBeta);
          }
        }
      }
      if (VecTools.distance(betas, prev) < tolerance) {
        break;
      }
      VecTools.copyTo(betas, prev, 0);
    }
    return new Linear(betas);
  }
}
