package com.expleague.ml.methods;

import com.expleague.ml.TargetFunc;
import com.expleague.ml.data.set.VecDataSet;
import com.expleague.ml.randomnessAware.RandomnessAwareTrans;

public interface RandomnessAwareVecOptimization<Loss extends TargetFunc> extends VecOptimization<Loss> {

  @Override
  RandomnessAwareTrans fit(VecDataSet learn, Loss loss);

  abstract class Stub<Loss extends TargetFunc> implements RandomnessAwareVecOptimization<Loss> {
  }

}
