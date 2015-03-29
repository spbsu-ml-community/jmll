package com.spbsu.bernulli.betaBinomialMixture;

import com.spbsu.bernulli.EM;
import com.spbsu.bernulli.caches.BetaCache;
import com.spbsu.bernulli.caches.Digamma1Cache;
import com.spbsu.bernulli.caches.DigammaCache;
import com.spbsu.commons.math.vectors.Mx;
import com.spbsu.commons.math.vectors.impl.mx.VecBasedMx;
import com.spbsu.commons.random.FastRandom;

import java.util.Arrays;

import static java.lang.Double.isNaN;


//reparametrezation of beta-distribution for solve singularities
// alpha + beta <= MaxN — maximim number of "prior" observations
//reparametrization of beta — \mu \in [0,1] and N.
//optimizaion — EM for mixture, after E — newton with alternating mu and N, shrinking N on maxN
public class RegularizedBetaBinomialMixtureEM extends EM<BetaBinomialMixture> {
  final int k;
  final int[] sums;
  final int n;
  final Mx dummy;
  final BetaBinomialMixture model;
  final FastRandom random;
  final SpecialFunctionCache funcs[];
  final MeanOptimization mean;
  final PrecisionOptimization precision;
  final int N; //it's regularized — maximum number of pseudo-observations

  final double mu[];
  final double precisions[];


  public RegularizedBetaBinomialMixtureEM(int k, final int[] sums, final int n, final int N, FastRandom random) {
    this.k = k; //components count
    this.sums = sums;
    this.n = n;
    this.dummy = new VecBasedMx(sums.length, k);
    this.model = new BetaBinomialMixture(k, n, random);
    this.random = random;
    this.funcs = new SpecialFunctionCache[k];
    this.mu = new double[k];
    this.precisions = new double[k];
    for (int i = 0; i < k; ++i) {
      precisions[i] = (this.model.alphas[i] + this.model.betas[i]);
      mu[i] = this.model.alphas[i] / precisions[i];
      precisions[i] = Math.min(precisions[i], N);
      this.funcs[i] = new SpecialFunctionCache(mu[i], precisions[i], n);
    }
    updateModel();
    this.N = N;
    this.mean = new MeanOptimization();
    precision = new PrecisionOptimization(N);
  }


  private void updateCache() {
    for (int i = 0; i < k; ++i) {
      funcs[i].update(model.alphas[i], model.betas[i]);
    }
  }


  @Override
  protected void expectation() {
    double[] probs = new double[k];
    updateCache();
    for (int i = 0; i < sums.length; ++i) {
      final int m = sums[i];
      double denum = 0;
      for (int j = 0; j < k; ++j) {
        probs[j] = model.q[j] * funcs[j].calculate(m, n);
        denum += probs[j];
      }
      for (int j = 0; j < k; ++j) {
        dummy.set(i, j, probs[j] /= denum);
      }
    }
  }

  private final int newtonIters = 3;
  private final double gradientStep = 0.05;
  private final double newtonStep = 0.01;
  private final int gradientIters = 20;

  private final int iterations = 3;
  boolean first = true;

  @Override
  protected void maximization() {
    final double probs[] = new double[k];
    for (int i = 0; i < sums.length; ++i) {
      for (int j = 0; j < k; ++j) {
        probs[j] += dummy.get(i, j);
      }
    }
    double total = 0;
    for (int i = 0; i < k; ++i) {
      total += probs[i];
    }
    for (int i = 0; i < k; ++i)
      model.q[i] = probs[i] / total;

    for (int i = 0; i < iterations; ++i) {
      mean.maximize();
      precision.maximize();
    }
  }


  int count = 300;
  double oldLikelihood = Double.NEGATIVE_INFINITY;

  @Override
  protected boolean stop() {
    final double currentLL = likelihood();
    if (Math.abs(oldLikelihood - currentLL) < 1e-1) {
      return true;
    }
    oldLikelihood = currentLL;
    return --count < 0;
  }

  @Override
  public BetaBinomialMixture model() {
    return model;
  }

  @Override
  protected double likelihood() {
    updateCache();
    double ll = 0;
    for (int i = 0; i < sums.length; ++i) {
      double p = 0;
      final int m = sums[i];
      for (int j = 0; j < model.alphas.length; ++j) {
        p += model.q[j] * funcs[j].calculate(m, n);
      }
      ll += Math.log(p);
    }

    return ll;
  }


  //emperical bayes estimation with fitted beta-mixture
  public double[] estimate(boolean fit) {
    if (fit) {
      fit();
    }
    expectation();
    double result[] = new double[sums.length];
    for (int i = 0; i < sums.length; ++i) {
      for (int j = 0; j < k; ++j)
        result[i] += dummy.get(i, j) * (sums[i] + model.alphas[j]) * 1.0 / (n + model.betas[j] + model.alphas[j]);
    }
    return result;
  }


  class MeanOptimization {
    final double gradientCache[];
    final double newtonCache[];

    MeanOptimization() {
      this.gradientCache = new double[2 * model.alphas.length];
      this.newtonCache = new double[3 * model.alphas.length];
    }

    private boolean newtonStep(double step) {
      updateCache();
      Arrays.fill(newtonCache, 0.0);
      fillGradient();

      final double cache[] = new double[k];
      final double cache2[] = new double[k];
      for (int i = 0; i < k; ++i) {
        final double beta = model.betas[i];
        final double alpha = model.alphas[i];
        cache[i] = -2 * (funcs[i].digamma(Type.Alpha, 0) - funcs[i].digamma(Type.Beta, 0));
        cache[i] += beta * (funcs[i].digamma1(Type.Alpha, 0) + funcs[i].digamma1(Type.Beta, 0));

        cache2[i] = alpha * beta * (funcs[i].digamma1(Type.Alpha, 0) + funcs[i].digamma1(Type.Beta, 0));
        cache2[i] += (beta - alpha) * (funcs[i].digamma(Type.Alpha, 0) - funcs[i].digamma(Type.Beta, 0));
      }
      for (int i = 0; i < sums.length; ++i) {
        final int m = sums[i];
        for (int j = 0; j < k; ++j) {
          final SpecialFunctionCache func = funcs[j];
          final double prob = dummy.get(i, j);
          final double alpha = model.alphas[j];
          final double beta = model.betas[j];


          final int idx0 = 3 * j; //a
          final int idx1 = 3 * j + 1;//b
          final int idx2 = 3 * j + 2;//d

          final double dpp;
          final double dqq;
          {
            final double tmp1 = -2 * func.digamma(Type.Beta, n - m) - beta * func.digamma1(Type.Beta, n - m);
            final double tmp2 = 2 * func.digamma(Type.Alpha, m) - beta * func.digamma1(Type.Alpha, m);
            final double tmp = prob * (tmp1 + tmp2 + cache[j]);

            dpp = -beta * tmp;
            dqq = alpha * tmp;
          }

          final double dpq;
          {
            final double tmp1 = (beta - alpha) * (func.digamma(Type.Beta, n - m) - func.digamma(Type.Alpha, m));
            final double tmp2 = -(alpha * beta) * (func.digamma1(Type.Beta, n - m) + func.digamma1(Type.Alpha, m));
            final double tmp = prob * (tmp1 + tmp2 + cache2[j]);
            dpq = tmp;
          }

          newtonCache[idx0] += dpp;
          newtonCache[idx1] += dpq;
          newtonCache[idx2] += dqq;
        }
      }

      for (int i = 0; i < k; ++i) {
        //gradient
        final double dp = gradientCache[2 * i];
        final double dq = gradientCache[2 * i + 1];
        //hessian
        final double a = newtonCache[3 * i];
        final double b = newtonCache[3 * i + 1];
        final double d = newtonCache[3 * i + 2];
        final double det = a * d - b * b;

        final double dirp = (d * dp - b * dq) / det;
        final double dirq = (a * dq - b * dp) / det;


        while (mu[i] - step * dirp < 1e-3 || 1 - mu[i] - step * dirq < 1e-3) {
          step *= step;
          if (step < 1e-15)
            return true;
        }

        double p = mu[i] - step * dirp;
        double q = 1 - mu[i] - step * dirq;
        p /= (p + q);
        mu[i] = p;
      }
      updateModel();
      return false;
    }

    //don't use it before update
    private void fillGradient() {
      Arrays.fill(gradientCache, 0.0);
      final double cache[] = new double[k];
      for (int i = 0; i < k; ++i) {
        cache[i] = funcs[i].digamma(Type.Alpha, 0) - funcs[i].digamma(Type.Beta, 0);
      }
      for (int i = 0; i < sums.length; ++i) {
        final int m = sums[i];
        for (int j = 0; j < k; ++j) {
          final double p = dummy.get(i, j);
          final double tmp = p * (cache[j] - funcs[j].digamma(Type.Alpha, m) + funcs[j].digamma(Type.Beta, n - m));
          final double grad1 = -model.betas[j] * tmp / precisions[j];
          final double grad2 = model.alphas[j] * tmp / precisions[j];
          gradientCache[2 * j] += grad1;
          gradientCache[2 * j + 1] += grad2;
        }
      }
    }

    private boolean gradientStep(double step) {
      updateCache();
      fillGradient();
      for (int i = 0; i < gradientCache.length; ++i) {
        if (isNaN(gradientCache[i])) {
          return true;
        }
      }
      for (int i = 0; i < k; ++i) {
        final double alpha = model.alphas[i];
        final double dalpha = gradientCache[2 * i];
        final double beta = model.betas[i];
        final double dbeta = gradientCache[2 * i + 1];
        while (alpha + step * dalpha < 1e-3 || beta + step * dbeta < 1e-3) {
          step *= step;
          if (step < 1e-15)
            return true;
        }
        final double newAlpha = alpha + step * dalpha;
        final double newBeta = beta + step * dbeta;
        mu[i] = newAlpha / (newAlpha + newBeta);
      }
      updateModel();
      return false;
    }

    boolean first = true;

    boolean maximize() {
//      if (first) {
      for (int i = 0; i < gradientIters; ++i)
        gradientStep(gradientStep);
//        first = false;
//      }
//      for (int i = 0; i < newtonIters; ++i)
//        newtonStep(newtonStep);
      return true;
    }
  }


  class PrecisionOptimization {
    final double gradientCache[];
    final double newtonCache[];
    final int maxPrecision;
    boolean[] stopped;
    int stoppedCount;

    PrecisionOptimization(int N) {
      this.gradientCache = new double[model.alphas.length];
      this.maxPrecision = N;
      this.newtonCache = new double[model.alphas.length];
      this.stopped = new boolean[model.alphas.length];
      stoppedCount = 0;
    }

    private boolean newtonStep(double step) {
      updateCache();
      Arrays.fill(newtonCache, 0.0);
      fillGradient();

      final double cache[] = new double[k];
      for (int i = 0; i < k; ++i) {
        if (stopped[i])
          continue;
        final double p = mu[i];
        cache[i] = -p * p * funcs[i].digamma1(Type.Alpha, 0) - (1 - p) * (1 - p) * funcs[i].digamma1(Type.Beta, 0);
        cache[i] += -funcs[i].digamma1(Type.AlphaBeta, n) + funcs[i].digamma1(Type.AlphaBeta, 0);
      }
      for (int i = 0; i < sums.length; ++i) {
        final int m = sums[i];
        for (int j = 0; j < k; ++j) {
          if (stopped[j])
            continue;
          final double prob = dummy.get(i, j);
          final double p = mu[j];
          final double dgrad = prob * (cache[j] + (1 - p) * (1 - p) * funcs[j].digamma1(Type.Beta, n - m) + p * p * funcs[j].digamma1(Type.Alpha, m));
          newtonCache[j] += dgrad;
        }
      }

      for (int i = 0; i < k; ++i) {
        if (stopped[i])
          continue;
        //matrix
        double N = precisions[i] - step * (gradientCache[i] / newtonCache[i]);
        if (N > 0) {
          if (precisions[i] > maxPrecision) {
            precisions[i] = maxPrecision;
            stopped[i] = true;
            stoppedCount++;
          } else {
            precisions[i] = N;
          }
        }
      }
      updateModel();
      return false;
    }

    //don't use it before update()
    private void fillGradient() {
      Arrays.fill(gradientCache, 0.0);
      final double cache[] = new double[k];
      for (int i = 0; i < k; ++i) {
        if (stopped[i])
          continue;
        cache[i] = -mu[i] * funcs[i].digamma(Type.Alpha, 0) - (1 - mu[i]) * funcs[i].digamma(Type.Beta, 0)
                + funcs[i].digamma(Type.AlphaBeta, 0) - funcs[i].digamma(Type.AlphaBeta, n);
      }
      for (int i = 0; i < sums.length; ++i) {
        final int m = sums[i];
        for (int j = 0; j < k; ++j) {
          if (stopped[j])
            continue;
          final double prob = dummy.get(i, j);
          final double p = mu[j];
          final SpecialFunctionCache func = funcs[j];
          final double grad = (1 - p) * func.digamma(Type.Beta, n - m) + p * func.digamma(Type.Alpha, m) + cache[j];
          gradientCache[j] += prob * grad;
        }
      }
    }

    private boolean gradientStep(double step) {
      updateCache();
      fillGradient();
      for (int i = 0; i < gradientCache.length; ++i) {
        if (isNaN(gradientCache[i])) {
          return true;
        }
      }
      for (int i = 0; i < k; ++i) {
        if (stopped[i])
          continue;
        while (precisions[i] + step * gradientCache[i] < 1e-3) {
          step *= 0.5;
          if (step < 1e-10)
            return false;
        }
        precisions[i] += step * gradientCache[i];
        if (precisions[i] > maxPrecision) {
          precisions[i] = maxPrecision;
          stopped[i] = true;
          stoppedCount++;
        }

      }
      updateModel();
      return false;
    }

    boolean first = true;

    boolean maximize() {
      if (stoppedCount == k)
        return false;
      if (first) {
        for (int i = 0; i < gradientIters; ++i)
          gradientStep(gradientStep);
        first = false;
      }

      for (int i = 0; i < 3; ++i)
        gradientStep(0.01);

      for (int i = 0; i < newtonIters; ++i)
        newtonStep(newtonStep);
      return true;
//    }
    }
  }

  private void updateModel() {
    for (int i = 0; i < model.alphas.length; ++i) {
      model.alphas[i] = mu[i] * precisions[i];
      model.betas[i] = (1 - mu[i]) * precisions[i];
    }
  }

  private enum Type {
    Alpha,
    Beta,
    AlphaBeta
  }

  private double alpha(double mu, double N) {
    return mu * N;
  }

  private double beta(double mu, double N) {
    return (1 - mu) * N;
  }

  private class SpecialFunctionCache {
    DigammaCache head;
    DigammaCache tail;
    DigammaCache alphabeta;
    Digamma1Cache dalphabeta;
    Digamma1Cache dhead;
    Digamma1Cache dtail;
    BetaCache betaCache;


    public SpecialFunctionCache(double mu, double N, int n) {
      betaCache = new BetaCache(alpha(mu, N), beta(mu, N), n);
      head = new DigammaCache(mu * N, n);
      tail = new DigammaCache((1 - mu) * N, n);
      dhead = new Digamma1Cache(mu * N, n);
      dtail = new Digamma1Cache((1 - mu) * N, n);
      dalphabeta = new Digamma1Cache(N, n);
      alphabeta = new DigammaCache(N, n);
    }


    public double calculate(int m, int n) {
      return betaCache.calculate(m, n);
    }


    final public double digamma(Type type, int offset) {
      if (type == Type.Alpha) {
        return head.calculate(offset);
      } else if (type == Type.Beta) {
        return tail.calculate(offset);
      }

      return alphabeta.calculate(offset);
    }

    public double digamma1(Type type, int offset) {
      if (type == Type.Alpha) {
        return dhead.calculate(offset);
      } else if (type == Type.Beta) {
        return dtail.calculate(offset);
      }
      return dalphabeta.calculate(offset);
    }


    final public void update(final double alpha, final double beta) {
      betaCache.update(alpha, beta);
      head.update(alpha);
      tail.update(beta);
      dhead.update(alpha);
      dtail.update(beta);
      dalphabeta.update(alpha + beta);
      alphabeta.update(alpha + beta);
    }
  }
}










