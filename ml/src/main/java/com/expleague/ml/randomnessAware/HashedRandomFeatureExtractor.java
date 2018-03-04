package com.expleague.ml.randomnessAware;

import com.expleague.commons.math.vectors.Mx;
import com.expleague.commons.math.vectors.Vec;
import com.expleague.ml.distributions.RandomVariable;
import com.expleague.ml.distributions.RandomVec;

public interface HashedRandomFeatureExtractor<U extends RandomVariable> extends VecRandomFeatureExtractor<U> {

  int hash(final Vec vec);

  U variable(int hash);

  default U compute(final Vec vec) {
    return variable(hash(vec));
  }

  RandomVec randomVecForBins(final int[] bins);

  default RandomVec applyAll(final Mx dataSet) {
    final int[] bins = new int[dataSet.length()];
    for (int i = 0; i < dataSet.length(); ++i) {
      bins[i] = hash(dataSet.row(i));
    }
    return randomVecForBins(bins);
  }

}
