package com.expleague.ml.distributions.parametric;

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
