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
 * Pathwise coordinate descent (for more details see articles by Friedman, Hastie, Tibshirani)
 */


public class ElasticNetMethod extends VecOptimization.Stub<L2> {
  private final double tolerance;
  private double alpha;
  private double lambda;

  public ElasticNetMethod(final double tolerance, final double alpha, final double lambda) {
    this.tolerance = tolerance;
    this.lambda = lambda;
    this.alpha = alpha;
  }

  @Override
  public Trans fit(final VecDataSet ds, final L2 loss) {
    final ElasticNetCache cache = new ElasticNetCache(ds.data(), loss.target, alpha, lambda);
    return fit(cache);
  }

  public Trans fit(final VecDataSet ds, final L2 loss, final Vec init) {
    final ElasticNetCache cache = new ElasticNetCache(ds.data(), loss.target, init, alpha, lambda);
    return fit(cache);
  }


  public int checkIterations = 2;

  public Linear fit(ElasticNetCache cache) {
    boolean updated = true;
    Vec prev;
    Vec betas = cache.betas();
    while (updated) {
      updated = false;
      prev = betas;
      for (int i = 0; i < checkIterations; ++i) {
        for (int k = 0; k < cache.dim(); ++k) {
          updated = cache.update(k) || updated;
        }
        if (!updated)
          break;
      }
      betas = cache.betas();
      if (VecTools.distance(betas, prev) < tolerance) {
        break;
      }
    }
    return new Linear(betas);
  }


  public static class ElasticNetCache {
    private final Mx data;
    private final Vec target;

    private final boolean[][] isFeaturesProductCached;
    private final boolean[] isTargetCached;
    private final double[] gradient;
    private final double[][] featureProducts;
    private final double[] targetProducts;
    private final Vec betas;
    private int dim;
    private double alpha;
    private double lambda;

    public ElasticNetCache(final Mx data, final Vec target, final Vec init, int dim, double alpha, double lambda) {
      this.alpha = alpha;
      this.lambda = lambda;
      this.data = data;
      this.target = target;
      this.betas = init;
      this.dim = 0;
      isFeaturesProductCached = new boolean[betas.dim()][];
      isTargetCached = new boolean[betas.dim()];
      featureProducts = new double[betas.dim()][];
      targetProducts = new double[betas.dim()];
      gradient = new double[betas.dim()];
      for (int i = 0; i < betas.dim(); ++i) {
        featureProducts[i] = new double[i + 1];
        isFeaturesProductCached[i] = new boolean[i + 1];
      }
      this.updateDim(dim);
    }

    public ElasticNetCache(final Mx data, final Vec target, final Vec init, double alpha, double lambda) {
      this(data, target, init, init.dim(), alpha, lambda);
    }

    public ElasticNetCache(final Mx data, final Vec target, double alpha, double lambda) {
      this(data, target, new ArrayVec(data.columns()), alpha, lambda);
    }

    public ElasticNetCache(final Mx data, final Vec target,int dim, double alpha, double lambda) {
      this(data, target, new ArrayVec(data.columns()),dim, alpha, lambda);
    }


    public double beta(int i) {
      return betas.get(i);
    }

    public int dim() {
      return dim;
    }

    public void updateDim(int newDim) {
      final int oldDim = dim;
      dim = newDim;
      for (int i = oldDim; i < dim; ++i) {
        double res = targetProduct(i);
        for (int j = 0; j < i; ++j) {
          final double beta = betas.get(j);
          res -= beta != 0 ? beta * featureProduct(i, j) : 0;
        }
        for (int j = i + 1; j < dim; ++j) {
          final double beta = betas.get(j);
          res -= beta != 0 ? beta * featureProduct(i, j) : 0;
        }
        gradient[i] = res;
      }
      for (int i=0; i < oldDim;++i)  {
        for (int j=oldDim; j < dim;++j) {
          final double beta = betas.get(j);
          gradient[i] -= beta != 0 ? beta * featureProduct(i, j) : 0;
        }
      }
    }

    public double gradient(int k) {
      return gradient[k];
    }

    private double featureProduct(int i, int j) {
      if (i < j) {
        return featureProduct(j, i);
      }
      if (!isFeaturesProductCached[i][j]) {
        featureProducts[i][j] = VecTools.multiply(data.col(i), data.col(j));
        isFeaturesProductCached[i][j] = true;
      }
      return featureProducts[i][j];
    }

    private double targetProduct(int k) {
      if (!isTargetCached[k]) {
        targetProducts[k] = VecTools.multiply(target, data.col(k));
        isTargetCached[k] = true;
      }
      return targetProducts[k];
    }

    public void setLambda(double lambda) {
      this.lambda = lambda;
    }

    public void setAlpha(double alpha) {
      this.alpha = alpha;
    }

    public boolean update(int k) {
      final int N = data.rows();
      double newBeta = gradient(k);
      newBeta = softThreshold(newBeta, N * lambda * alpha);
      newBeta /= (featureProduct(k, k) + N * lambda * (1 - alpha));
      if (Math.abs(newBeta - betas.get(k)) > 1e-9f) {
        update(k, newBeta);
        return true;
      }
      return false;
    }

    private void update(int k, double newBeta) {
      final double beta = betas.get(k);
      for (int i = 0; i < k; ++i) {
        gradient[i] -= (newBeta - beta) * featureProduct(k, i);
      }
      for (int i = k + 1; i < dim; ++i) {
        gradient[i] -= (newBeta - beta) * featureProduct(k, i);
      }
      betas.set(k, newBeta);
    }

    public Vec betas() {
      return copy(betas);
    }

    private double softThreshold(final double z, final double j) {
      final double sgn = Math.signum(z);
      return sgn * Math.max(sgn * z - j, 0);
    }
  }

}
