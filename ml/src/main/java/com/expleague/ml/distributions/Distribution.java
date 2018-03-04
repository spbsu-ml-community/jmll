package com.expleague.ml.distributions;

import com.expleague.commons.random.FastRandom;

/**
 * Created by noxoomo on 22/10/2017.
 */
public interface Distribution<T> {

  double logProb(final T object);

  default double prob(final T object) {
    return Math.exp(logProb(object));
  }

  T instance(final FastRandom random);
}



