package com.expleague.ml.distributions;

import com.expleague.commons.math.vectors.Vec;
import com.expleague.commons.seq.Seq;

public interface Aggregator<U, V> {

  V create();

  Aggregator<U, V> update(final U observation,
                          final double weight,
                          V dest);

  default Aggregator<U, V> update(final Seq<U> observation,
                                  final Vec weights,
                                  V dest) {
    Aggregator<U, V> aggregator = this;
    for (int i = 0; i < observation.length(); ++i) {
      final double w=  weights !=  null ? weights.get(i) : 1.0;
      aggregator = update(observation.at(i), w, dest);
    }
    return aggregator;
  }

  default Aggregator<U, V> update(final Seq<U> observation, V dest) {
    return update(observation, null, dest);
  }
}


