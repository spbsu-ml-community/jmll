package com.expleague.ml.distributions.parametric;

import com.expleague.commons.random.FastRandom;
import com.expleague.ml.distributions.AdditiveFamilyDistribution;
import com.expleague.ml.distributions.RandomVariable;
import com.expleague.ml.distributions.RandomVecBuilder;

import static com.expleague.commons.math.MathTools.sqr;
import static java.lang.StrictMath.sqrt;

/**
 * Created by noxoomo on 06/11/2017.
 */
public interface NormalDistribution extends RandomVariable, AdditiveFamilyDistribution<NormalDistribution> {
  double mu();

  default double  precision() {
    return sqr(1.0 / sd());
  }

  double sd();

  default double logDensity(double value) {
    return Stub.logDensity(mu(), sd(), value);
  }

  default double cdf(double value) {
    return Stub.cdf(mu(), sd(), value);
  }

  default double expectation() {
    return mu();
  }

  class Stub {
    static double logDensity(double mu, double sd, double x) {
      throw new RuntimeException("unimplemented");
    }

    static double cdf(double mu, double sd, double x) {
      throw new RuntimeException("unimplemented");
    }
  }

  default double sample(final FastRandom random) {
    return random.nextGaussian(mu(), sd());
  }
}


