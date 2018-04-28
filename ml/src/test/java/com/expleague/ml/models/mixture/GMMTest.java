package com.expleague.ml.models.mixture;

import com.expleague.commons.math.vectors.Mx;
import com.expleague.commons.math.vectors.Vec;
import com.expleague.commons.math.vectors.impl.mx.VecBasedMx;
import com.expleague.commons.random.FastRandom;
import com.expleague.ml.func.LogLikelihood;
import com.expleague.ml.optimization.EM;
import com.expleague.ml.optimization.Optimize;
import org.junit.Test;

import java.util.stream.Collectors;
import java.util.stream.IntStream;

import static junit.framework.TestCase.assertEquals;
import static junit.framework.TestCase.assertTrue;

public class GMMTest {
  private static final FastRandom rng = new FastRandom();
  private static final double EPS = 1e-4;

  private static class Stat {
    final double[] mu;
    final double[] sigma;
    final double[] w;
    private final int numComponents;

    Stat(int numComponents) {
      mu = new double[numComponents];
      sigma = new double[numComponents];
      w = new double[numComponents];
      this.numComponents = numComponents;
    }

    void calculate(Vec samples, int[] sampleIdx) {
      for (int i = 0; i < samples.dim(); i++) {
        final double value = samples.get(i);
        w[sampleIdx[i]]++;
        mu[sampleIdx[i]] += value;
        sigma[sampleIdx[i]] += value * value;
      }

      double sumW = 0.;
      for (int i = 0; i < mu.length; i++) {
        mu[i] /= w[i];
        sigma[i] /= w[i];
        sigma[i] -= mu[i] * mu[i];
        sumW += w[i];
      }

      for (int i = 0; i < w.length; i++) {
        w[i] /= sumW;
      }
    }

    @Override
    public String toString() {
      return IntStream.range(0, numComponents)
          .mapToObj(i -> String.format("(%.5f, %.5f, %.5f)", w[i], mu[i], sigma[i]))
          .collect(Collectors.joining(","));
    }
  }

  private void testGMMParam(Mixture gmm, Stat stat) {
    final int numComponents = gmm.numComponents();
    final Vec parameters = gmm.getParameters();

    final Vec wEst = parameters.sub(0, numComponents);
    final Vec muEst = parameters.sub(numComponents, numComponents);
    final Vec sEst = parameters.sub(2 * numComponents, numComponents);

    for (int i = 0; i < numComponents; i++) {
      final double mu_i = stat.mu[i];

      boolean found = false;
      for (int j = 0; j < numComponents; j++) {
        if (Math.abs(mu_i - muEst.get(j)) < EPS) {
          assertEquals(stat.w[i], wEst.get(j), EPS);
          assertEquals(stat.sigma[i], sEst.get(j), EPS);
          found = true;
        }
      }
      assertTrue("Not found suitable mean for " + mu_i, found);
    }
  }

  @Test
  public void twoGausssiansTest() {
    final int numComponents = 2;
    final int numSamples = 100_000;
    final int numIters = 100;

    final GaussianMixture1D gmm = new GaussianMixture1D(numComponents, rng);
    final Mx samples = new VecBasedMx(numSamples, 1);

    final double mu1 = 0.;
    final double sigma1 = 1;

    final double mu2 = 10.;
    final double sigma2 = 1;

    int[] sampleIdx = new int[numSamples];

    for (int i = 0; i < numSamples; i++) {
      double value = rng.nextGaussian();
      sampleIdx[i] = rng.nextBoolean() ? 1 : 0;
      value = sampleIdx[i] == 0 ? value * sigma1 + mu1 : value * sigma2 + mu2;
      samples.set(i, 0, value);
    }

    final Stat stat = new Stat(numComponents);
    stat.calculate(samples, sampleIdx);

    System.out.println(stat);

    final Optimize<LogLikelihood> optimizer = new EM(samples, gmm, numIters);
    optimizer.optimize(new LogLikelihood());

    testGMMParam(gmm, stat);
  }
}
