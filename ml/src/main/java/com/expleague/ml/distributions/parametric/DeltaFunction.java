package com.expleague.ml.distributions.parametric;

import com.expleague.commons.random.FastRandom;
import com.expleague.ml.distributions.RandomVariable;
import com.expleague.ml.distributions.RandomVecBuilder;
import com.expleague.ml.distributions.parametric.impl.DeltaDistributionVec;

public interface DeltaFunction extends RandomVariable {

  double value();

//  P(x <= V) = [1,
  default double cdf(final double value) {
    return value < value() ? 0 : 1;
  }

  default double mean() {
    return value();
  }

  default double logDensity(final double value) {
    return value == value() ? 0.0 : Double.NEGATIVE_INFINITY;
  }

  default double sample(final FastRandom random) {
    return value();
  }

  default double expectation() {
    return mean();
  }


}

