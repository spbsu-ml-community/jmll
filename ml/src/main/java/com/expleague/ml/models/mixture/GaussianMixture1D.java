package com.expleague.ml.models.mixture;

import com.expleague.commons.math.FuncC1;
import com.expleague.commons.math.MathTools;
import com.expleague.commons.math.vectors.Mx;
import com.expleague.commons.math.vectors.Vec;
import com.expleague.commons.math.vectors.VecTools;
import com.expleague.commons.math.vectors.impl.vectors.ArrayVec;
import com.expleague.commons.random.FastRandom;
import com.expleague.ml.func.Gaussian1D;
import org.jetbrains.annotations.NotNull;

import java.util.stream.IntStream;

public class GaussianMixture1D extends FuncC1.Stub implements Mixture {
  private final int numComponents;
  private final Vec mu;
  private final Vec sigma2;
  private final Vec weights;
  private final Vec parameters;

  public GaussianMixture1D(int numComponents, FastRandom rng) {
    if (numComponents < 2) {
      throw new IllegalArgumentException("Number of components must be >= 2");
    }
    this.numComponents = numComponents;

    parameters = new ArrayVec(numComponents * 3);
    weights = parameters.sub(0, numComponents);
    mu = parameters.sub(numComponents, numComponents);
    sigma2 = parameters.sub(numComponents * 2, numComponents);

    VecTools.fillUniform(mu, rng);
    VecTools.fill(sigma2, 1.);

    for (int i = 0; i < weights.dim(); i++) {
      weights.set(i, rng.nextInt(numComponents) + 1);
    }
    VecTools.scale(weights, 1. / VecTools.sum(weights));
  }

  private double mixtureValue(double x) {
    double prob = 0.;
    for (int i = 0; i < numComponents; i++) {
      prob += weights.get(i) * Gaussian1D.value(x, mu.get(i), sigma2.get(i));
    }
    return prob;
  }

  private double mixtureValue(double x, int compIdx) {
    return weights.get(compIdx) *
        Gaussian1D.value(x, mu.get(compIdx), sigma2.get(compIdx));
  }

  @Override
  public void setParameters(Vec parameters) {
    VecTools.assign(weights, parameters.sub(0, weights.dim()));
    VecTools.assign(mu, parameters.sub(weights.dim(), mu.dim()));
    VecTools.assign(sigma2, parameters.sub(weights.dim() + mu.dim(), sigma2.dim()));
  }

  @NotNull
  @Override
  public Vec getParameters() {
    return parameters;
  }

  @Override
  public Vec probAll(Mx samples, Vec to) {
    for (int i = 0; i < samples.rows(); i++) {
      to.set(i, prob(samples.row(i)));
    }

    return to;
  }

  @Override
  public Vec byComponentProb(Vec sample, Vec to) {
    assert(to.dim() == numComponents);
    final double x = sample.get(0);
    final double norm = mixtureValue(sample.get(0));
    for (int i = 0; i < numComponents; i++) {
      final double prob = mixtureValue(x, i) / norm;
      to.set(i, prob);
    }

    return to;
  }

  @Override
  public Mx byComponentProb(Mx samples, Mx to) {
    IntStream.range(0, samples.rows())
        .forEach(i -> byComponentProb(samples.row(i), to.row(i)));
    return to;
  }

  @Override
  public void upgradeParameters(Mx probBySample, Mx samples) {
    final int numSamples = samples.length();
    VecTools.fill(weights, 0.);
    VecTools.fill(mu, 0.);
    VecTools.fill(sigma2, 0.);

    for (int t = 0; t < numSamples; t++) {
      final double x = samples.get(t, 0);
      for (int i = 0; i < numComponents; i++) {
        final double prob = probBySample.get(t, i);

        assert(prob >= - MathTools.EPSILON && prob <= 1. + MathTools.EPSILON);

        weights.adjust(i, prob);
        mu.adjust(i, prob * x);
        sigma2.adjust(i, prob * x * x);
      }
    }

    for (int i = 0; i < numComponents; i++) {
      mu.set(i, mu.get(i) / weights.get(i));
      sigma2.set(i, sigma2.get(i) / weights.get(i));
    }

    VecTools.scale(sigma2, numSamples / (numSamples - 1));
    VecTools.scale(weights, 1. / numSamples);

    for (int i = 0; i < numComponents; i++) {
      final double mu_i = mu.get(i);
      sigma2.adjust(i, - mu_i * mu_i);
    }
  }

  @Override
  public int numComponents() {
    return numComponents;
  }

  @Override
  public int wdim() {
    return parameters.dim();
  }

  @Override
  public double value(Vec x) {
    return mixtureValue(x.get(0));
  }

  @Override
  public int dim() {
    return 1;
  }
}
