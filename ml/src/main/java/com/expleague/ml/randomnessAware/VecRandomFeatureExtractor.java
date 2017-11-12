package com.expleague.ml.randomnessAware;

import com.expleague.commons.func.Computable;
import com.expleague.commons.math.vectors.Mx;
import com.expleague.commons.math.vectors.Vec;
import com.expleague.ml.data.set.VecDataSet;
import com.expleague.ml.distributions.DynamicRandomVec;
import com.expleague.ml.distributions.RandomVariable;
import com.expleague.ml.distributions.RandomVec;
import com.expleague.ml.distributions.RandomVecBuilder;

/**
 * Created by noxoomo on 26/10/2017.
 */
public interface VecRandomFeatureExtractor<U extends RandomVariable<U>> extends Computable<Vec, U> {

  U compute(final Vec featuresVec);

  RandomVec<U> apply(final VecDataSet dataSet);

  int dim();

  RandomVecBuilder<U> randomVecBuilder();
}



