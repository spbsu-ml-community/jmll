package com.expleague.ml.methods;

import com.expleague.commons.func.impl.WeakListenerHolderImpl;
import com.expleague.commons.math.Trans;
import com.expleague.commons.math.vectors.VecTools;
import com.expleague.commons.math.vectors.impl.vectors.ArrayVec;
import com.expleague.commons.random.FastRandom;
import com.expleague.ml.data.set.VecDataSet;
import com.expleague.ml.data.tools.DataTools;
import com.expleague.ml.loss.AdditiveLoss;
import com.expleague.ml.loss.WeightedLoss;
import com.expleague.ml.func.Ensemble;

public class RandomForest<Loss extends AdditiveLoss> extends WeakListenerHolderImpl<Trans> implements VecOptimization<Loss> {
  protected final FastRandom rnd;
  private final VecOptimization<WeightedLoss<? extends Loss>> weak;
  private final int treesCount;

  public RandomForest(final VecOptimization<WeightedLoss<? extends Loss>> weak, final FastRandom rnd, final int treesCount) {
    this.weak = weak;
    this.treesCount = treesCount;
    this.rnd = rnd;
  }

  @Override
  public Trans fit(final VecDataSet learn, final Loss globalLoss) {
    final Trans[] weakModels = new Trans[treesCount];
    for (int i = 0; i < treesCount; ++i)
      weakModels[i] = weak.fit(learn, DataTools.bootstrap(globalLoss, rnd));
    return new Ensemble(weakModels, VecTools.fill(new ArrayVec(weakModels.length), 1.0 / treesCount));
  }
}
