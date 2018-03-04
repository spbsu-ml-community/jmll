package com.expleague.ml.distributions;

public interface NumericBayesianUpdater<U extends Distribution<?>, Stat extends SufficientStatistic> extends BayesianUpdater<U, Double, Stat > {

  NumericAggregator<Stat> aggregator();

  U posteriorTo(final U prior,
                final Stat statistic,
                final U posterior);

  default U posteriorTo(final U prior,
                        final double value,
                        final double weight,
                        final U to) {
    final Stat stat = aggregator().create();
    aggregator().update(value, weight, stat);
    return posteriorTo(prior, stat, to);
  }

  default U posteriorTo(final U prior,
                        final Double observation,
                        final double weight,
                        final U to) {
    return posteriorTo(prior, observation.doubleValue(), weight, to);
  }

}
