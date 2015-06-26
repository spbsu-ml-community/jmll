package com.spbsu.ml.methods.linearRegressionExperiments;

import com.spbsu.commons.math.MathTools;
import com.spbsu.commons.math.vectors.Mx;
import com.spbsu.commons.math.vectors.MxTools;
import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.commons.math.vectors.VecTools;
import com.spbsu.commons.math.vectors.impl.mx.VecBasedMx;
import com.spbsu.commons.math.vectors.impl.vectors.ArrayVec;
import com.spbsu.commons.util.ThreadTools;
import com.spbsu.ml.func.Linear;
import gnu.trove.iterator.TDoubleIterator;

import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ThreadPoolExecutor;

/**
 * Created by noxoomo on 10/06/15.
 */


public class EmpericalBayesRidgeRegression {
  private final Mx[] datas;
  private final double tolerance = 1e-14;
  private final Vec[] targets;
  private final int featuresCount;
  private final EmpericalBayesRidgeRegressionCache[] cache;
  private double alpha = 1e-12;
  //  private double beta = 1;
  double diff = Double.POSITIVE_INFINITY;
  private Vec[] means;


  private static ThreadPoolExecutor exec = ThreadTools.createBGExecutor("bayesian linear model executor", -1);

  public EmpericalBayesRidgeRegression(final Mx[] datas,final Vec[] targets) {
    this.datas = datas;
    this.targets = targets;
    featuresCount = datas[0].columns();

    for (int i = 1; i < datas.length; ++i) {
      if (datas[i].columns() != featuresCount)
        throw new IllegalArgumentException("tasks should use common set of features");
    }

    means = new Vec[datas.length];
    cache = new EmpericalBayesRidgeRegressionCache[targets.length];
    final CountDownLatch latch = new CountDownLatch(targets.length);
    for (int i = 0; i < targets.length; ++i) {
      final int ind = i;
      exec.submit(new Runnable() {
        @Override
        public void run() {
          cache[ind] = new EmpericalBayesRidgeRegressionCache(datas[ind], targets[ind]);
          latch.countDown();
        }
      });
    }
    try {
      latch.await();
    } catch (InterruptedException e) {
      e.printStackTrace();
    }
  }

  Linear[] fit() {
    int iter = 0;
    while (diff > tolerance && iter++ < 30) {
      fillMeans();
      final double[] gammas = calcGammas();
      final double gamma;
      {
        double res = 0;
        for (double g : gammas)
          res += g;
        gamma = res;
      }

      double newAlpha = calcNewAlpha(gamma);
//      final double newBeta = calcNewBeta(gamma);
      final double newBetas[] = calcNewBetas(gammas);
//      final double d = der(newAlpha);
//      if (Math.abs(d) < tolerance)
//        break;
//      System.out.println("alpha: " + d);
//      System.out.println("beta " + derBeta(newBetas));

      diff = Math.abs(alpha - newAlpha);
      for (int i = 0; i < cache.length; ++i) {
        cache[i].update(newAlpha, newBetas[i]);
      }
      alpha = newAlpha;

//      if (diff < tolerance  || iter > 19) {
//        for (double beta : newBetas) {
//          System.out.println("Derivative " + der(newAlpha));
//          System.out.println(newAlpha / beta);
//        }
//      }
//      beta = newBetas;
    }

    fillMeans();
    Linear[] result = new Linear[datas.length];
    for (int i = 0; i < result.length; ++i) {
      result[i] = new Linear(cache[i].getMean());
    }
    return result;
  }

  void fillMeans() {
    for (int i = 0; i < cache.length; ++i)
      means[i] = cache[i].getMean();
  }

  double[] calcGammas() {
    double[] gamma = new double[cache.length];
    for (int i = 0; i < cache.length; ++i) {
      TDoubleIterator eigenValuesIterator = cache[i].getEigenvaluesIterator();
      while (eigenValuesIterator.hasNext()) {
        final double lambda = eigenValuesIterator.next();
        gamma[i] += lambda / (lambda + alpha);
      }
    }
    return gamma;
  }

  double der(double alpha) {
    double der = 0;
    for (int i = 0; i < cache.length; ++i) {
      der += 0.5 * featuresCount / alpha;
      der -= 0.5 * VecTools.multiply(means[i], means[i]);
      TDoubleIterator eigenValuesIterator = cache[i].getEigenvaluesIterator();
      while (eigenValuesIterator.hasNext()) {
        final double lambda = eigenValuesIterator.next();
        der -= 0.5 / (lambda + alpha);
      }
    }
    return der;
  }

  double derBeta(double[] betas) {
    double der = 0;
    for (int i = 0; i < cache.length; ++i) {
      der += 0.5 * datas[i].rows() / betas[i];
      TDoubleIterator eigenValuesIterator = cache[i].getEigenvaluesIterator();
      while (eigenValuesIterator.hasNext()) {
        final double lambda = eigenValuesIterator.next();
        der -= 0.5 * lambda / (lambda + alpha) / betas[i];
      }
      for (int j = 0; j < datas[i].rows(); ++j)
        der -= 0.5 * (MathTools.sqr(targets[i].get(j) - VecTools.multiply(means[i], datas[i].row(j))));
    }
    return der;
  }

  double calcNewAlpha(final double gamma) {
    double denum = 0;
    for (int i = 0; i < cache.length; ++i) {
      final Vec mean = means[i];
      denum += VecTools.multiply(mean, mean);
    }
    return gamma / denum;
  }

  double[] calcNewBetas(final double[] gammas) {
    final double[] betas = new double[gammas.length];

    final CountDownLatch latch = new CountDownLatch(cache.length);
    for (int i = 0; i < betas.length; ++i) {
      final int ind = i;
      exec.submit(new Runnable() {
        @Override
        public void run() {
          double error = 0;
          final Vec mean = means[ind];
          final Mx data = datas[ind];
          final Vec target = targets[ind];
          for (int i = 0; i < data.rows(); ++i) {
            error += MathTools.sqr(target.get(i) - VecTools.multiply(mean, data.row(i)));
          }
//          betas[ind] = (data.rows() - gammas[ind]) / error;
          betas[ind] = (data.rows() - data.columns()) / error;
          latch.countDown();
        }
      });
    }

    try {
      latch.await();
    } catch (InterruptedException e) {
      e.printStackTrace();
    }
    return betas;
  }
}


// t = Xw + \varepsilon
// w_i \sim N(0, 1.0 / alpha)
// \varepsilon \sim N(0,1.0 / beta)
// if alpha known — equals to ridge regression
//this is cache class, no VecOptimization

class EmpericalBayesRidgeRegressionCache {
  private double alpha = 1e-15;
  private double beta;
  private final Mx sigma;
  private final Vec eigenValues;
  private Mx A; // cache for ( alpha I + beta Sigma)
  private Mx invA; // cache for ( alpha I + beta Sigma)^-1
  private final Vec covFeatureWithTarget;

  public EmpericalBayesRidgeRegressionCache(Mx data, Vec target) {
    sigma = new VecBasedMx(data.columns(), data.columns());
    covFeatureWithTarget = new ArrayVec(data.columns());
    for (int i = 0; i < data.columns(); ++i) {
      final Vec feature = data.col(i);
      sigma.set(i, i, VecTools.multiply(feature, feature));
      covFeatureWithTarget.set(i, VecTools.multiply(feature, target));
      for (int j = i + 1; j < data.columns(); ++j) {
        final double cov = VecTools.multiply(feature, data.col(j));
        sigma.set(i, j, cov);
        sigma.set(j, i, cov);
      }
    }
    A = new VecBasedMx(sigma.columns(), sigma.columns());
    Mx eigenValuesMx = new VecBasedMx(sigma.columns(), sigma.columns());
    Mx Q = new VecBasedMx(sigma.columns(), sigma.columns());
    MxTools.eigenDecomposition(sigma, Q, eigenValuesMx);
    this.eigenValues = new ArrayVec(sigma.columns());
    for (int i = 0; i < sigma.columns(); ++i) {
      this.eigenValues.set(i, eigenValuesMx.get(i, i));
    }
    beta = 1.0;
    update(alpha, beta);
  }


  void update(double alpha, double beta) {
    this.alpha = alpha;
    this.beta = beta;

    for (int i = 0; i < sigma.columns(); ++i) {
      A.set(i, i, alpha + beta * sigma.get(i, i));
      for (int j = i + 1; j < sigma.columns(); ++j) {
        final double val = beta * sigma.get(i, j);
        A.set(i, j, val);
        A.set(j, i, val);
      }
    }
    invA = MxTools.inverse(A);
  }

  Vec getMean() {
    Vec result = MxTools.multiply(invA, covFeatureWithTarget);
    return VecTools.scale(result, beta);
  }

  TDoubleIterator getEigenvaluesIterator() {
    return new TDoubleIterator() {
      int i = 0;

      @Override
      public double next() {
        return beta * eigenValues.get(i++);
      }

      @Override
      public boolean hasNext() {
        return i < eigenValues.dim();
      }

      @Override
      public void remove() {
        throw new UnsupportedOperationException();
      }
    };
  }
}

