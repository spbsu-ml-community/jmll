package com.spbsu.ml.methods;

import com.spbsu.commons.math.vectors.Mx;
import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.commons.math.vectors.VecTools;
import com.spbsu.commons.math.vectors.impl.vectors.ArrayVec;
import com.spbsu.commons.math.Trans;
import com.spbsu.ml.data.set.VecDataSet;
import com.spbsu.ml.func.Linear;
import com.spbsu.ml.loss.L2;
import com.spbsu.ml.models.ShifftedTrans;
import org.apache.commons.math3.util.FastMath;

import java.util.ArrayList;
import java.util.List;

import static com.spbsu.commons.math.vectors.VecTools.adjust;
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
    double intercept  = 0;
    Vec target = loss.target;
    for (int i=0; i < target.dim();++i) {
      intercept += target.get(i);
    }
    intercept /= target.dim();
    Vec localTarget = copy(target);
    adjust(localTarget,-intercept);
    final ElasticNetCache cache = new ElasticNetCache(ds.data(), localTarget, alpha, lambda);
    Trans result = fit(cache);
    return new ShifftedTrans(result,intercept);
  }

  public Trans fit(final VecDataSet ds, final L2 loss, final Vec init) {
    final ElasticNetCache cache = new ElasticNetCache(ds.data(), loss.target, init, alpha, lambda);
    return fit(cache);
  }

  public final List<Linear> fit(final Mx data, final Vec target, int nlambda, double lambdaEps) {
    final ElasticNetCache cache = new ElasticNetCache(data, target, alpha, lambda);
    double lambdaMax = Double.NEGATIVE_INFINITY;
    for (int i=0; i < data.columns();++i) {
      lambdaMax = FastMath.max(FastMath.abs(cache.targetProduct(i)), lambdaMax);
    }
    lambdaMax *= 1.0  / (alpha *  data.rows());
    double lambdaMin = lambdaMax * lambdaEps;
    double step = (lambdaMax - lambdaMin) / nlambda;
    List<Linear> path = new ArrayList<>(nlambda);

    for (double lambda = lambdaMax; lambda > lambdaMin; lambda -= step) {
      cache.setLambda(lambda);
      path.add(fit(cache));
    }
    return path;
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
    private final double equalsTolerance = 1e-10;
    private final boolean[] isFeaturesProductCached;
    private final boolean[] isTargetCached;
    private final double[] gradient;
    private final double[] featureProducts;
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
      isFeaturesProductCached = new boolean[betas.dim()*betas.dim()];
      isTargetCached = new boolean[betas.dim()];
      featureProducts = new double[betas.dim() * betas.dim()];
      targetProducts = new double[betas.dim()];
      gradient = new double[betas.dim()];
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
          res -= beta != 0 ? beta * featureProduct(j,i) : 0;
        }
        for (int j = i + 1; j < dim; ++j) {
          final double beta = betas.get(j);
          res -= beta !=0 ? beta * featureProduct(i, j) : 0;
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



    private double dot(Mx data, int i, int j) {
      final int rows = data.rows();
      final int length = 4*(rows / 4);
      double result = 0;
      final double[] cache = new double[4];
      for (int k=0; k < length; k+=4) {
        final double l1 = data.get(k,i);
        final double l2 = data.get(k+1,i);
        final double l3 = data.get(k+2,i);
        final double l4 = data.get(k+3,i);

        final double r1 = data.get(k,j);
        final double r2 = data.get(k+1,j);
        final double r3 = data.get(k+2,j);
        final double r4 = data.get(k+3,j);

        cache[0] = l1 * r1;
        cache[1] = l2 * r2;
        cache[2] = l3 * r3;
        cache[3] = l4 * r4;
        cache[0] += cache[2];
        cache[1] += cache[3];
        cache[0] += cache[1];
        result += cache[0];
      }
      for (int k=length; k < rows;++k) {
        result += data.get(k,i) * data.get(k,j);
      }
      return result;
    }
//jvm vectorization http://hg.openjdk.java.net/hsx/hotspot-comp/hotspot/rev/006050192a5a
    private double targetDot(Mx data, int i, Vec target) {
      final int rows = data.rows();
      final int length = 4*(rows / 4);
      double result = 0;
      final double[] cache = new double[4];
      for (int k=0; k < length; k+=4) {
        final double l1 = data.get(k,i);
        final double l2 = data.get(k+1,i);
        final double l3 = data.get(k+2,i);
        final double l4 = data.get(k+3,i);

        final double r1 = target.get(k);
        final double r2 = target.get(k+1);
        final double r3 = target.get(k+2);
        final double r4 = target.get(k+3);

        cache[0] = l1 * r1;
        cache[1] = l2 * r2;
        cache[2] = l3 * r3;
        cache[3] = l4 * r4;
        cache[0] += cache[2];
        cache[1] += cache[3];
        cache[0] += cache[1];
        result += cache[0];
      }
      for (int k=length; k < rows;++k) {
        result += data.get(k,i) * target.get(k);
      }
      return result;
    }

    private double featureProduct(int i, int j) {
      if (i > j) {
        return featureProduct(j, i);
      }
      if (!isFeaturesProductCached[i*betas.dim() + j]) {
        featureProducts[i*betas.dim() + j] = dot(data, i, j);
        isFeaturesProductCached[i*betas.dim() + j] = true;
      }
      return featureProducts[i*betas.dim() + j];
    }

    private double targetProduct(int k) {
      if (!isTargetCached[k]) {
        targetProducts[k] = targetDot(data, k, target);
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
      if (Math.abs(newBeta - betas.get(k)) > equalsTolerance) {
        update(k, newBeta);
        return true;
      }
      return false;
    }

    private void update(final int k,final double newBeta) {
      final double beta = betas.get(k);
      final double diff = newBeta - beta;
      {
        final int length = 4 * (k / 4);
        final double[] gradientLocal = gradient;
        for (int i = 0; i < length; i += 4) {
          final int ind = i;
          final int localK = k;
          final double dot1 = diff * featureProduct(ind,localK);
          final double dot2 = diff * featureProduct(ind + 1,localK);
          final double dot3 = diff * featureProduct(ind + 2,localK);
          final double dot4 = diff * featureProduct(ind + 3,localK);
          gradientLocal[ind] -= dot1;
          gradientLocal[ind + 1] -= dot2;
          gradientLocal[ind + 2] -= dot3;
          gradientLocal[ind + 3] -= dot4;
        }
        for (int i = length; i < k; ++i) {
          gradientLocal[i] -= diff * featureProduct(i,k);
        }
      }

      {
        final int offset = k +1;
        final int size = dim - offset;
        final int end = 4 * (size / 4) + offset;
        final double[] gradientLocal = gradient;
        for (int i = offset; i < end; i += 4) {
          final int ind = i;
          final int localK = k;
          final double dot1 = diff * featureProduct(localK, ind);
          final double dot2 = diff * featureProduct(localK, ind + 1);
          final double dot3 = diff * featureProduct(localK, ind + 2);
          final double dot4 = diff * featureProduct(localK, ind + 3);
          gradientLocal[ind] -= dot1;
          gradientLocal[ind + 1] -= dot2;
          gradientLocal[ind + 2] -= dot3;
          gradientLocal[ind + 3] -= dot4;
        }
        for (int i = end; i < dim; ++i) {
          gradientLocal[i] -= diff * featureProduct(k,i);
        }
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

