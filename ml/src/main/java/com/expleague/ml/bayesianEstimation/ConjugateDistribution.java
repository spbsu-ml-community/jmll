package com.expleague.ml.bayesianEstimation;

import com.expleague.ml.distributions.RandomVariable;

/**
 * Created by noxoomo on 29/10/2017.
 */
public interface ConjugateDistribution {

  <V extends RandomVariable> Class<? extends RandomVariable> get(V var);
}
