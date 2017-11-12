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
    private TDistribution distribution;

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
        final double instance;
        if (degrees < 100) {
          final TDistribution randomGenerator = getRng(random);
          instance = randomGenerator.sample();
        } else {
          instance = random.nextGaussian();
        }
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
    private TDistribution getRng(final FastRandom random) {
      if (distribution == null) {
        final RandomGenerator randomGenerator = new RandomGenerator() {
          @Override
          public void setSeed(final int i) {
          }

          @Override
          public void setSeed(final int[] ints) {
          }

          @Override
          public void setSeed(final long l) {
          }

          @Override
          public void nextBytes(final byte[] bytes) {
            random.nextBytes(bytes);
          }

          @Override
          public int nextInt() {
            return random.nextInt();
          }

          @Override
          public int nextInt(final int i) {
            return random.nextInt(i);
          }

          @Override
          public long nextLong() {
            return random.nextLong();
          }

          @Override
          public boolean nextBoolean() {
            return random.nextBoolean();
          }

          @Override
          public float nextFloat() {
            return random.nextFloat();
          }

          @Override
          public double nextDouble() {
            return random.nextDouble();
          }

          @Override
          public double nextGaussian() {
            return random.nextGaussian();
          }
        };
        synchronized (this) {
          if (distribution == null) {
            distribution = new TDistribution(randomGenerator, degreesOfFreedom(), 1e-4D);
          }
        }
      }
      return distribution;
    }

    @Override
    public double mu() {
      return mu;
    }

    @Override
    public double scale() {
      return scale;
    }

    private double generate(final FastRandom random) {
      return random.nextGaussian() * sqrt(random.nextGamma(degrees / 2, degrees / 2));
    }
  }
}
