package com.expleague.ml.distributions;

import com.expleague.commons.math.vectors.Vec;
import com.expleague.commons.seq.Seq;

public interface NumericAggregator<V> extends Aggregator<Double, V> {

  NumericAggregator<V> update(final double observation,
                              final double w,
                              V dest);

  default NumericAggregator<V> update(final Vec observation,
                                      final Vec weights,
                                      V dest) {
    NumericAggregator<V> aggregator = this;
    for (int i = 0; i < observation.length(); ++i) {
      double w = weights != null ? weights.get(i) : 1.0;
      aggregator.update(observation.get(i), w, dest);
    }
    return aggregator;
  }


  @Override
  default NumericAggregator<V> update(final Double observation, final double w, V dest) {
    return update(observation.doubleValue(), w, dest);
  }



}
