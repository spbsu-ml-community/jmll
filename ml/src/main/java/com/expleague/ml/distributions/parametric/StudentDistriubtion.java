package com.expleague.ml.distributions.parametric;

import com.expleague.commons.random.FastRandom;
import com.expleague.ml.distributions.RandomVariable;
import com.expleague.ml.distributions.RandomVecBuilder;
import com.expleague.ml.distributions.samplers.RandomVariableSampler;
import org.apache.commons.math3.distribution.TDistribution;
import org.apache.commons.math3.random.RandomGenerator;

import java.util.Map;
import java.util.WeakHashMap;

import static java.lang.StrictMath.sqrt;

/**
 * Created by noxoomo on 06/11/2017.
 */
public interface StudentDistriubtion extends RandomVariable<StudentDistriubtion> {
  double degreesOfFreedom();

  double mu();

  double scale();

  class Impl implements StudentDistriubtion {
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
    public double mean() {
      return mu;
    }

    @Override
    public RandomVariableSampler sampler() {
      return random -> {
        final double instance = generate(random);
        return mu + scale * instance;
      };
    }

    @Override
    public RandomVecBuilder<StudentDistriubtion> vecBuilder() {
      throw new RuntimeException("unimplemented");
    }

    @Override
    public double degreesOfFreedom() {
      return degrees;
    }
//


    @Override
    public double mu() {
      return mu;
    }

    @Override
    public double scale() {
      return scale;
    }

    private double generate(final FastRandom random) {
      return random.nextGaussian() * sqrt(1.0 / random.nextBayessianGamma(degrees / 2, degrees / 2));
    }
  }
}
