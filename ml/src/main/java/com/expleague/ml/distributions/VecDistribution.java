package com.expleague.ml.distributions;

import com.expleague.commons.seq.Seq;

public interface VecDistribution extends RandomSeq<Double> {

  RandomVariable at(final int idx);

  default double cdf(final int idx, final double value) {
    return at(idx).cdf(value);
  }

}
