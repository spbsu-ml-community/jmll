package com.expleague.ml.distributions.samplers;

import com.expleague.commons.random.FastRandom;

import java.util.Random;

/**
 * Created by noxoomo on 26/10/2017.
 */
public interface DistributionSampler<T> {

  T sample(final FastRandom random);

}




