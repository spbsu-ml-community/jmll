package com.expleague.ml.distributions;

import com.expleague.commons.random.FastRandom;

public interface RandomVariable extends Distribution<Double> {

  //P(x < value)
  default double cdf(final double value) {
    throw new RuntimeException("Unimplemented");
  }

  default double logProb(final Double value) {
    return logDensity(value);
  }

  double logDensity(final double value);

  double sample(final FastRandom random);

  default Double instance(final FastRandom random) {
    return sample(random);
  }

  default double expectation() {
    throw new RuntimeException("Unimplemented");
  }

}


