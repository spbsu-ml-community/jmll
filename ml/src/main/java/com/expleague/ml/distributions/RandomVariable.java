package com.expleague.ml.distributions;

import com.expleague.ml.distributions.samplers.RandomVariableSampler;

public interface RandomVariable<U extends RandomVariable<U>> extends Distribution<Double> {

  double cdf(final double value);

//  default double inverseCdf(final double p) {
//    throw new RuntimeException("Unimplemented");
//  }

//  double logDensity(final double x);

  double mean();

//  default double logLikelihood(final Double x) {
//    return logDensity(x);
//  }

  default Double expectation() {
    return mean();
  }

  RandomVariableSampler sampler();

  RandomVecBuilder<U> vecBuilder();

}


