package com.expleague.ml.distributions.parametric;

import com.expleague.ml.distributions.RandomVariable;
import com.expleague.ml.distributions.RandomVecBuilder;
import com.expleague.ml.distributions.samplers.RandomVariableSampler;

import static com.expleague.commons.math.MathTools.sqr;
import static java.lang.StrictMath.sqrt;

/**
 * Created by noxoomo on 06/11/2017.
 */
public interface NormalDistribution extends RandomVariable<NormalDistribution> {
  double mu();
  double precision();

  class Impl implements NormalDistribution {
    final double mu;
    final double sd;

    public Impl(final double mu, final double precision) {
      this.mu = mu;
      this.sd = sqrt(1.0 / precision);
    }

    @Override
    public double cdf(final double value) {
      throw new RuntimeException("unimplemented");
    }

    @Override
    public double mean() {
      return mu;
    }

    @Override
    public RandomVariableSampler sampler() {
      return random -> mu + random.nextGaussian() * sd;
    }

    @Override
    public RandomVecBuilder<NormalDistribution> vecBuilder() {
      throw new RuntimeException("unimplemented");
    }

    @Override
    public double mu() {
      return mu;
    }

    @Override
    public double precision() {
      return sqr(1.0 / sd);
    }
  }
}
