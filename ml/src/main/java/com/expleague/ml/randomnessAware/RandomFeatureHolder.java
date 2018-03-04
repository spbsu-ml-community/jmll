package com.expleague.ml.randomnessAware;

import com.expleague.ml.data.set.VecDataSet;
import com.expleague.ml.distributions.RandomVariable;
import com.expleague.ml.distributions.RandomVec;

public interface RandomFeatureHolder<U extends RandomVariable> {

  RandomVec feature(final VecDataSet dataSet);

  VecRandomFeatureExtractor<U> extractor();

}
