package com.expleague.ml.bayesianEstimation.impl;

import com.expleague.ml.bayesianEstimation.ConjugateBayesianEstimator;
import com.expleague.ml.distributions.RandomVec;
import com.expleague.ml.distributions.parametric.BetaDistribution;
import com.expleague.ml.distributions.parametric.BetaVecDistribution;
import com.expleague.ml.distributions.parametric.impl.BetaDistributionImpl;

public class BetaConjugateBayesianEstimator implements ConjugateBayesianEstimator<BetaDistribution> {

  @Override
  public BetaDistribution clone(final BetaDistribution dist) {
    return new BetaDistributionImpl(dist.alpha(), dist.beta());
  }

  @Override
  public BetaDistribution improperPrior() {
    return new BetaDistributionImpl(0, 0);
  }

  @Override
  public int dim() {
    return 2;
  }

  @Override
  public RandomVec<BetaDistribution> update(final int idx,
                                            final double observation,
                                            final RandomVec<BetaDistribution> distribution) {
    if (distribution instanceof BetaVecDistribution) {
      BetaVecDistribution dist = (BetaVecDistribution) distribution;
      dist.update(idx, dist.alpha(idx) + observation, dist.beta(idx) + 1.0 - observation);
    } else {
      final BetaDistribution coordinate = distribution.randomVariable(idx);
      distribution.setRandomVariable(idx, coordinate.update(coordinate.alpha() + observation, coordinate.beta() + 1.0 - observation));
    }
    return distribution;
  }

  @Override
  public BetaDistribution update(final double observation, final BetaDistribution dist) {
    return dist.update(dist.alpha() + observation, dist.beta() + 1.0 - observation);
  }

}
