package com.expleague.ml.methods;

import com.expleague.commons.func.impl.WeakListenerHolderImpl;
import com.expleague.commons.math.Trans;
import com.expleague.commons.math.vectors.Vec;
import com.expleague.commons.random.FastRandom;
import com.expleague.ml.data.set.VecDataSet;
import com.expleague.ml.data.tools.DataTools;
import com.expleague.ml.loss.AdditiveLoss;
import com.expleague.ml.loss.WeightedLoss;

/**
* User: solar
* Date: 21.12.2010
* Time: 22:13:54
*/
public class WeightedBootstrapOptimization<Loss extends AdditiveLoss> extends WeakListenerHolderImpl<Trans> implements VecOptimization<Loss> {
  private final Vec weights;
  protected final FastRandom rnd;
  private final VecOptimization<WeightedLoss<? extends Loss>> weak;

  public WeightedBootstrapOptimization(VecOptimization<WeightedLoss<? extends Loss>> weak, Vec weights, FastRandom rnd) {
    this.weak = weak;
    this.weights = weights;
    this.rnd = rnd;
  }

  @Override
  public Trans fit(final VecDataSet learn, final Loss globalLoss) {
    final float[] poissonWeights = new float[globalLoss.xdim()];
    for (int i = 0; i < globalLoss.xdim(); i++) {
      poissonWeights[i] = rnd.nextPoisson(1.) * (float)weights.get(i);
    }
    final WeightedLoss<Loss> localLoss = new WeightedLoss<>(globalLoss, poissonWeights);
    return weak.fit(learn, localLoss);
  }
}
