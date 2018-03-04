package com.expleague.ml.distributions;

import com.expleague.commons.math.vectors.Vec;
import com.expleague.commons.seq.Seq;

/**
 * Created by noxoomo on 29/10/2017.
 */
public interface BayesianUpdater<U extends Distribution<?>, V, Stat extends SufficientStatistic> {

  U improperPrior();

  Aggregator<V, Stat> aggregator();


  U posteriorTo(final U prior,
                final Stat statistic,
                final U to);

  //TODO(noxoomo): weights
  default U posteriorTo(final U prior,
                        final V observation,
                        final double w,
                        final U to
                        ) {
    final Stat statistic = aggregator().create();
    aggregator().update(observation, w, statistic);
    return posteriorTo(prior,
                        statistic,
                        to);
  }

  default U posteriorTo(final U prior,
                        final V observation,
                        final U to) {
    return posteriorTo(prior, observation, 1.0, to);
  }


  default U posteriorTo(final U prior,
                        final Seq<V> observations,
                        final Vec weights, U to) {
    final Stat statistic = aggregator().create();
    aggregator().update(observations, weights, statistic);
    return posteriorTo(prior, statistic, to);
  }

  default U posteriorTo(final U prior,
                        final Seq<V> observations, U to) {
    return posteriorTo(prior, observations, null, to);
  }
}

