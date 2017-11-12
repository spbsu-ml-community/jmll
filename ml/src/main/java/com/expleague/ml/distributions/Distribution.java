package com.expleague.ml.distributions;

import com.expleague.ml.distributions.samplers.DistributionSampler;

/**
 * Created by noxoomo on 22/10/2017.
 */
public interface Distribution<T> {

//  double logLikelihood(final T object);

  T expectation();

  DistributionSampler<T> sampler();
}


