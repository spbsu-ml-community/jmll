package com.expleague.ml.bayesianEstimation;

import com.expleague.ml.distributions.Distribution;

/**
 * Created by noxoomo on 29/10/2017.
 */
public interface ConjugateUpdaterFactory {
  ConjugateBayesianEstimator<?> create(Class<Distribution> clazz);
}
