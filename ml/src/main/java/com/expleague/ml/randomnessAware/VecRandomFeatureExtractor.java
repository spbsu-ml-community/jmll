package com.expleague.ml.randomnessAware;

import com.expleague.commons.func.Computable;
import com.expleague.commons.math.vectors.Vec;
import com.expleague.ml.data.set.VecDataSet;
import com.expleague.ml.distributions.RandomVariable;
import com.expleague.ml.distributions.RandomVec;
import com.expleague.ml.distributions.RandomVecBuilder;

/**
 * Created by noxoomo on 26/10/2017.
 */
public interface VecRandomFeatureExtractor<U extends RandomVariable> extends Computable<Vec, U> {

  U compute(final Vec featuresVec);

  RandomVec computeAll(final VecDataSet dataSet);

  int dim();

}



