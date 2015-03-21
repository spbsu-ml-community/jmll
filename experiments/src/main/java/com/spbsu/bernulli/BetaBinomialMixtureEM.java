package com.spbsu.bernulli;

import com.spbsu.commons.math.vectors.Mx;
import com.spbsu.commons.math.vectors.impl.mx.VecBasedMx;
import com.spbsu.commons.random.FastRandom;

import java.util.Arrays;

import static com.spbsu.commons.math.MathTools.sqr;
import static java.lang.Double.isNaN;

public class BetaBinomialMixtureEM extends EM<BetaBinomialMixture> {
  final int k;
  final int[] sums;
  final int n;
  final Mx dummy;
  final BetaBinomialMixture model;
  final FastRandom random;
  final SpecialFunctionCache funcs[];
  final double gradientCache[];
  final double newtonCache[];

  public BetaBinomialMixtureEM(int k, final int[] sums, final int n, FastRandom random) {
    this.k = k; //components count
    this.sums = sums;
    this.n = n;
    this.dummy = new VecBasedMx(sums.length, k);
    this.model = new BetaBinomialMixture(k, random);
    this.random = random;
    this.funcs = new SpecialFunctionCache[k];
    for (int i = 0; i < k; ++i) {
      this.funcs[i] = new SpecialFunctionCache(this.model.alphas[i], this.model.betas[i], n);
    }
    this.gradientCache = new double[this.model.alphas.length * 2];
    // it's k matrix, 3 elements block per component mixture
    this.newtonCache = new double[this.model.alphas.length * 3];
    this.oldPoint = new double[this.model.alphas.length * 2];
  }


  final private void updateCache() {
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

  private final int iterations = 3;
  private final double gradientStartStep = 0.01;
  private final double newtonStartStep = 0.01;
  private final int gradientIters = 25;

  private boolean newtonStep(double step) {
    updateCache();
    Arrays.fill(newtonCache, 0.0);
    fillGradient();

    final double psi1abdiff[] = new double[k];
    final double psi1a[] = new double[k];
    final double psi1b[] = new double[k];
    for (int i = 0; i < k; ++i) {
      psi1abdiff[i] = funcs[i].digamma1(SpecialFunctionCache.Type.AlphaBeta, 0) - funcs[i].digamma1(SpecialFunctionCache.Type.AlphaBeta, n);
      psi1a[i] = funcs[i].digamma1(SpecialFunctionCache.Type.Alpha, 0);
      psi1b[i] = funcs[i].digamma1(SpecialFunctionCache.Type.Beta, 0);
    }
    for (int i = 0; i < sums.length; ++i) {
      final int m = sums[i];
      for (int j = 0; j < k; ++j) {
        final double psi1am = funcs[j].digamma1(SpecialFunctionCache.Type.Alpha, m);
        final double psi1bm = funcs[j].digamma1(SpecialFunctionCache.Type.Beta, n - m);
        final int idx0 = 3 * j;
        final int idx1 = 3 * j + 1;
        final int idx2 = 3 * j + 2;
        final double p = dummy.get(i, j);
        //
        final double val0 = p * (-psi1a[j] + psi1am + psi1abdiff[j]);
        final double val1 = p * (psi1abdiff[j]);
        final double val2 = p * (-psi1b[j] + psi1bm + psi1abdiff[j]);
        //
        newtonCache[idx0] += val0;
        newtonCache[idx1] += val1;
        newtonCache[idx2] += val2;
      }
    }
    boolean status = false;

    for (int i = 0; i < k; ++i) {
      final double dalpha = gradientCache[2 * i];
      final double dbeta = gradientCache[2 * i + 1];
      //matrix
      final double a = newtonCache[3 * i]; //d/dalpha^2
      final double b = newtonCache[3 * i + 1];//d/alphabeta
      final double d = newtonCache[3 * i + 2];//d/betaallpha
      final double det = a * d - b * b;

      final double stepAlpha = (d * dalpha - b * dbeta) / det;
      final double stepBeta = (a * dbeta - b * dalpha) / det;

      model.alphas[i] -= step * stepAlpha;
      model.betas[i] -= step * stepBeta;
      if (model.betas[i] < 0.001) {
        model.betas[i] = 0.001;
        status = true;
      } else if (model.betas[i] > 10000) {
        status = true;
      }
      if (model.alphas[i] < 0.001) {
        model.alphas[i] = 0.001;
        status = true;
      } else if (model.alphas[i] > 10000) {
        status = true;
      }
    }
    return status;
  }


  private void fillGradient() {
    Arrays.fill(gradientCache, 0.0);
    final double psiasum[] = new double[k];
    final double psibsum[] = new double[k];
    for (int i = 0; i < k; ++i) {
      final double psiab = funcs[i].digamma(SpecialFunctionCache.Type.AlphaBeta, 0);
      final double psiabn = funcs[i].digamma(SpecialFunctionCache.Type.AlphaBeta, n);
      final double psia = funcs[i].digamma(SpecialFunctionCache.Type.Alpha, 0);
      final double psib = funcs[i].digamma(SpecialFunctionCache.Type.Beta, 0);
      psiasum[i] = -psia + psiab - psiabn;
      psibsum[i] = -psib + psiab - psiabn;
    }
    for (int i = 0; i < sums.length; ++i) {
      final int m = sums[i];
      for (int j = 0; j < k; ++j) {
        final double alphaGrad = dummy.get(i, j) * (psiasum[j] + funcs[j].digamma(SpecialFunctionCache.Type.Alpha, m));
        final double betaGrad = dummy.get(i, j) * (psibsum[j] + funcs[j].digamma(SpecialFunctionCache.Type.Beta, n - m));
        gradientCache[2 * j] += alphaGrad;
        gradientCache[2 * j + 1] += betaGrad;
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
    boolean status = false;
    for (int i = 0; i < k; ++i) {
      model.alphas[i] += step * gradientCache[2 * i];
      model.betas[i] += step * gradientCache[2 * i + 1];
      if (model.betas[i] < 0.001) {
        model.betas[i] = 0.001;
        status = true;
      } else if (model.betas[i] > 10000) {
        status = true;
      }
      if (model.alphas[i] < 0.001) {
        model.alphas[i] = 0.001;
        status = true;
      } else if (model.alphas[i] > 10000) {
        status = true;
      }
    }
    return status;
  }

//  @Override
//  protected void maximization() {
//    double probs[] = new double[k];
//    for (int i = 0; i < sums.length; ++i) {
//      for (int j = 0; j < k; ++j) {
//        probs[j] += dummy.get(i, j);
//      }
//    }
//    double total = 0;
//    for (int i = 0; i < k; ++i) {
//      total += probs[i];
//    }
//    for (int i = 0; i < k; ++i)
//      model.q[i] = probs[i] / total;
//
//    double step = startStep;
//    double ll = likelihood();
//    int iters = iterations;
//    while (true) {
//      for (int i = 0; i < iters; ++i)
//        if (gradientStep(step))
//          break;
//      double gradientNorm = 0;
//      for (int i = 0; i < gradientCache.length; ++i) {
//        gradientNorm += sqr(gradientCache[i]);
//      }
//      if (gradientNorm / gradientCache.length < 1e-2)
//        return;
//
//      double currentLL = likelihood();
//      if (currentLL + 0.01 >= ll || step < 1e-6) {
//        return;
//      }
//      step *= 0.5;
//    }
//  }

  boolean first = true;
  final int maxIterations = 10;

  @Override
  protected void maximization() {
    double probs[] = new double[k];
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

    double gradientStep = gradientStartStep;
    double ll = likelihood();
    double currentLL = Double.NEGATIVE_INFINITY;

    int iters = 0;
    while (currentLL < ll || iters < maxIterations) {
//      if (first) {
        for (int i = 0; i < gradientIters; ++i) {
          if (gradientStep(gradientStep)) {
            return;
          }
        }
//        first = false;
//      }

//      for (int i = 0; i < iterations; ++i) {
//        if (newtonStep(newtonStartStep)) {
//          return;
//        }
//      }
      double gradientNorm = 0;
      for (int i = 0; i < gradientCache.length; ++i) {
        gradientNorm += sqr(gradientCache[i]);
      }
      if (gradientNorm / gradientCache.length < 1e-2)
        return;
      currentLL = likelihood();
      ++iters;
    }
  }

  int count = 200;
  double[] oldPoint;

  @Override
  protected boolean stop() {
    double dist = 0;
    for (int i = 0; i < model.alphas.length; ++i) {
      if (model.alphas[i] > 10000) {
        return true;
      }
      if (model.betas[i] > 10000) {
        return true;
      }
      final double diff1 = oldPoint[2 * i] - model.alphas[i];
      final double diff2 = oldPoint[2 * i + 1] - model.betas[i];
      dist += diff1 * diff1 + diff2 * diff2;
      oldPoint[2 * i] = model.alphas[i];
      oldPoint[2 * i + 1] = model.betas[i];
    }
    count--;
    if (dist < 1e-2 || count < 0)
      return true;
    return false;
  }

  @Override
  protected BetaBinomialMixture model() {
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

}

