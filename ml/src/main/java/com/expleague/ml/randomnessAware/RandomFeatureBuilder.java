package com.expleague.ml.randomnessAware;

import com.expleague.ml.data.set.BinomialTargetHolder;
import com.expleague.ml.data.set.VecDataSet;

public interface RandomFeatureBuilder<Ds extends VecDataSet> {

  VecRandomFeatureExtractor build(final BinomialTargetHolder<Ds> data);

}

