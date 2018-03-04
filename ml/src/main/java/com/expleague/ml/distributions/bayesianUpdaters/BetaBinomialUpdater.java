package com.expleague.ml.distributions.bayesianUpdaters;

import com.expleague.commons.math.vectors.Vec;
import com.expleague.ml.distributions.*;
import com.expleague.ml.distributions.parametric.BetaDistribution;
import com.expleague.ml.distributions.parametric.impl.BetaDistributionImpl;

public class BetaBinomialUpdater implements NumericBayesianUpdater<BetaDistribution, VecSufficientStat> {

  private VecStatisticsAggregator.CombineFunction[] combineFuncs = new VecStatisticsAggregator.CombineFunction[]{
      VecStatisticsAggregator.sumCombine()
  };

  public BetaDistribution clone(final BetaDistribution dist) {
    return new BetaDistributionImpl(dist.alpha(), dist.beta());
  }

  @Override
  public BetaDistribution improperPrior() {
    return new BetaDistributionImpl(0, 0);
  }

  @Override
  public NumericAggregator<VecSufficientStat> aggregator() {
    return new VecStatisticsAggregator(combineFuncs);
  }


  @Override
  public BetaDistribution posteriorTo(final BetaDistribution prior,
                                      final VecSufficientStat statistic,
                                      final BetaDistribution to) {
    final Vec stats = statistic.stats();
    final double total = stats.get(0);
    final double firstClassCount = stats.get(1);
    final double newAlpha = prior.alpha() + firstClassCount;
    final double newBeta = prior.beta() + total - firstClassCount;
    return to.update(newAlpha, newBeta);
  }

}
