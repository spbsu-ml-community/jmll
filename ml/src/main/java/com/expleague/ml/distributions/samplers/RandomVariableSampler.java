package com.expleague.ml.distributions.samplers;

import com.expleague.commons.random.FastRandom;
import com.expleague.ml.distributions.samplers.DistributionSampler;

import java.util.Random;

/**
 * Created by noxoomo on 26/10/2017.
 */
public interface RandomVariableSampler extends DistributionSampler<Double> {

  double instance(final FastRandom random);

  default Double sample(final FastRandom random) {
    return instance(random);
  }

}

