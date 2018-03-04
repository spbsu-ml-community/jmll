package com.expleague.ml.distributions.parametric.impl;

import com.expleague.commons.random.FastRandom;
import com.expleague.ml.distributions.parametric.BetaDistribution;

public class BetaDistributionImpl implements BetaDistribution {
  double alpha;
  double beta;

  public BetaDistributionImpl(final double alpha, final double beta) {
    this.alpha = alpha;
    this.beta = beta;
  }

  @Override
  public double cdf(final double value) {
    return BetaDistribution.Stub.cumulativeProbability(value, alpha, beta);
  }

  @Override
  public double logDensity(double value) {
    throw new RuntimeException("Unimplemented");
  }

  @Override
  public double sample(FastRandom random) {
    return BetaDistribution.Stub.instance(random, alpha, beta);
  }

  public double mean() {
    return BetaDistribution.Stub.expectation(alpha, beta);
  }


  @Override
  public double alpha() {
    return alpha;
  }

  @Override
  public double beta() {
    return beta;
  }


  public BetaDistribution update(final double alpha, final double beta) {
    this.alpha = alpha;
    this.beta = beta;
    return this;
  }

  @Override
  public boolean equals(final Object o) {
    if (this == o) return true;
    if (!(o instanceof BetaDistribution)) return false;
    return BetaDistribution.Stub.equals(this, (BetaDistribution) o);
  }

  @Override
  public int hashCode() {
    return BetaDistribution.Stub.hashCode(this);
  }
}
