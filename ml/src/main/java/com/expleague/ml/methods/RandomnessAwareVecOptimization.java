package com.expleague.ml.methods;

import com.expleague.commons.math.vectors.Vec;
import com.expleague.ml.TargetFunc;
import com.expleague.ml.data.set.VecDataSet;
import com.expleague.ml.distributions.RandomVariable;
import com.expleague.ml.distributions.RandomVec;
import com.expleague.ml.randomnessAware.RandomFunc;

public interface RandomnessAwareVecOptimization<Loss extends TargetFunc> extends AnyOptimization<Loss, VecDataSet, Vec, RandomFunc> {

  RandomVec emptyVec(int dim);

  RandomVariable emptyVar();

  RandomFunc fit(final VecDataSet learn,
                 final Loss loss);

  abstract class Stub<Loss extends TargetFunc> implements RandomnessAwareVecOptimization<Loss> {
  }

}
