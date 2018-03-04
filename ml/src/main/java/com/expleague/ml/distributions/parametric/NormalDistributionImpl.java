package com.expleague.ml.distributions.parametric;

import com.expleague.commons.random.FastRandom;

import static com.expleague.commons.math.MathTools.sqr;

public class NormalDistributionImpl implements NormalDistribution {
  double mu;
  double sd;

  public NormalDistributionImpl(final double mu,
                                final double sd) {
    this.mu = mu;
    this.sd = sd;
  }


  public double mean() {
    return mu;
  }

  @Override
  public double mu() {
    return mu;
  }

  @Override
  public double sd() {
    return sd;
  }

  @Override
  public NormalDistribution add(NormalDistribution other) {
    mu += other.mu();
    sd = Math.sqrt(sqr(sd) + other.sd());
    return this;
  }

  @Override
  public NormalDistribution add(final NormalDistribution other,
                                final double scale) {
    mu += other.mu() * scale;
    sd = Math.sqrt(sqr(sd) + sqr(other.sd() * scale));
    return this;
  }

  @Override
  public NormalDistribution scale(final double scale) {
    sd *= scale;
    return this;
  }
}
