package com.expleague.ml.distributions.parametric;

import com.expleague.commons.random.FastRandom;
import com.expleague.ml.distributions.RandomVariable;

import static java.lang.StrictMath.sqrt;

/**
 * Created by noxoomo on 06/11/2017.
 */
public interface StudentDistribution extends RandomVariable {

  double degreesOfFreedom();

  double mu();

  double scale();

  class Impl implements StudentDistribution {
    private final double degrees;
    private final double mu;
    private final double scale;

    public Impl(final double degrees,
                final double mu,
                final double scale) {
      this.degrees = degrees;
      this.mu = mu;
      this.scale = scale;
    }


    @Override
    public double cdf(final double value) {
      throw new RuntimeException("unimplemented");
    }

    @Override
    public double logDensity(double value) {
      throw new RuntimeException("Unimplemented");
    }

    @Override
    public double sample(FastRandom random) {
      return random.nextGaussian() * sqrt(1.0 / random.nextBayessianGamma(degrees / 2, degrees / 2));
    }

    @Override
    public double degreesOfFreedom() {
      return degrees;
    }


    @Override
    public double mu() {
      return mu;
    }

    @Override
    public double scale() {
      return scale;
    }

  }
}
