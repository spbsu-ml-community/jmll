package com.expleague.ml.distributions.parametric;

import com.expleague.commons.random.FastRandom;
import com.expleague.ml.distributions.RandomVariable;
import com.expleague.ml.distributions.RandomVecBuilder;
import com.expleague.ml.distributions.samplers.RandomVariableSampler;

/**
 * Created by noxoomo on 06/11/2017.
 */
public interface NormalGammaDistribution extends RandomVariable<NormalGammaDistribution> {
  double mu();
  double lambda();
  double alpha();
  double beta();

  default double mean() {
    return mu();
  }
  NormalGammaDistribution update(double mu, double lambda, double alpha, double beta);

  default double cdf(double x) {
    return Stub.cdf(x, mu(), lambda(), alpha(), beta());
  }

  public static class Stub {
    public static double cdf(double x, double mu, double lambda, double alpha, double beta) {
      throw new RuntimeException("unimplemented");
    }

    public static double expectation(double mu, double lambda, double alpha, double beta) {
      return mu;
    }

    public static boolean equals(final NormalGammaDistribution left,
                                 final NormalGammaDistribution right) {
      return left.mu() == right.mu() && left.lambda() == right.lambda() && left.alpha() == right.alpha() && left.beta() == right.beta();
    }

    public static int hashCode(final NormalGammaDistribution dist) {
      int result;
      long temp;
      temp = Double.doubleToLongBits(dist.mu());
      result = (int) (temp ^ (temp >>> 32));
      temp = Double.doubleToLongBits(dist.lambda());
      result = 31 * result + (int) (temp ^ (temp >>> 32));
      temp = Double.doubleToLongBits(dist.alpha());
      result = 31 * result + (int) (temp ^ (temp >>> 32));
      temp = Double.doubleToLongBits(dist.beta());
      result = 31 * result + (int) (temp ^ (temp >>> 32));
      return result;
    }

    public static double instance(final FastRandom random, double mu, double lambda, double alpha, double beta) {
      final double tau = random.nextBayessianGamma(alpha, beta);
      return mu + random.nextGaussian() * Math.sqrt(1.0 / lambda / tau);
    }

  }

  class MeanImpl implements RandomVariable<NormalGammaDistribution> {
    final double mu;
    final double lambda;
    final double alpha;
    final double beta;
    public MeanImpl(final double mu, final double lambda,
                    final double alpha, final double beta) {
      this.mu = mu;
      this.lambda = lambda;
      this.alpha = alpha;
      this.beta = beta;

    }

    @Override
    public double cdf(final double value) {
      throw new RuntimeException("Unimplemented");
    }

    @Override
    public double mean() {
      return mu;
    }

    @Override
    public RandomVariableSampler sampler() {
      return random -> {
        final double tau = random.nextGamma(alpha, 1.0 / beta);
        return mu + random.nextGaussian() * 1.0 / Math.sqrt(tau);
      };
    }

    @Override
    public RandomVecBuilder<NormalGammaDistribution> vecBuilder() {
      throw new RuntimeException("Unimplemented");
    }
  }
}
