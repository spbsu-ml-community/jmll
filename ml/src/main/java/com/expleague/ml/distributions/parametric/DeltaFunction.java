package com.expleague.ml.distributions.parametric;

import com.expleague.ml.distributions.RandomVariable;
import com.expleague.ml.distributions.RandomVec;
import com.expleague.ml.distributions.RandomVecBuilder;
import com.expleague.ml.distributions.parametric.impl.DeltaDistributionVec;
import com.expleague.ml.distributions.samplers.RandomVariableSampler;

public interface DeltaFunction extends RandomVariable<DeltaFunction> {

  double value();

  default double cdf(final double value) {
    return value < value() ? 0 : 1;
  }

  default double mean() {
    return value();
  }

  default RandomVariableSampler sampler() {
    return rng -> value();
  }

  default RandomVecBuilder<DeltaFunction> vecBuilder() {
    return new VecBuilder();
  }

  class VecBuilder implements RandomVecBuilder<DeltaFunction> {
    final com.expleague.commons.math.vectors.impl.vectors.VecBuilder vecBuilder = new com.expleague.commons.math.vectors.impl.vectors.VecBuilder();

    @Override
    public RandomVecBuilder<DeltaFunction> add(final DeltaFunction distribution) {
      vecBuilder.append(distribution.value());
      return this;
    }

    @Override
    public RandomVec<DeltaFunction> build() {
      return new DeltaDistributionVec(vecBuilder.build());
    }
  }
}

