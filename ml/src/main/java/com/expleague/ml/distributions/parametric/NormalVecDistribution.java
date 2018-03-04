package com.expleague.ml.distributions.parametric;

import com.expleague.commons.random.FastRandom;
import com.expleague.ml.distributions.AdditiveFamilyDistribution;
import com.expleague.ml.distributions.RandomVec;

/**
 * Created by noxoomo on 12/02/2018.
 */
public interface NormalVecDistribution extends RandomVec, AdditiveFamilyDistribution<NormalVecDistribution> {

  NormalDistribution at(final int idx);

  NormalVecDistribution add(final int idx, double scale, NormalDistribution var);

  double mu(int idx);

  double sd(int idx);

  default double cdf(int idx, double x) {
    return NormalDistribution.Stub.cdf(mu(idx), sd(idx), x);
  }

  default double logDensity(int idx, double x) {
    return NormalDistribution.Stub.logDensity(mu(idx), sd(idx), x);
  }

  default double instance(int idx, FastRandom random) {
    return random.nextGaussian(mu(idx), sd(idx));
  }

}
