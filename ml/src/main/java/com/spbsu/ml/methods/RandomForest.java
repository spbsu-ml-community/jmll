package com.spbsu.ml.methods;

import com.spbsu.commons.func.impl.WeakListenerHolderImpl;
import com.spbsu.commons.math.vectors.VecTools;
import com.spbsu.commons.math.vectors.impl.vectors.ArrayVec;
import com.spbsu.commons.random.FastRandom;
import com.spbsu.commons.math.Trans;
import com.spbsu.ml.data.set.VecDataSet;
import com.spbsu.ml.data.tools.DataTools;
import com.spbsu.ml.func.Ensemble;
import com.spbsu.ml.loss.StatBasedLoss;
import com.spbsu.ml.loss.WeightedLoss;

public class RandomForest<Loss extends StatBasedLoss> extends WeakListenerHolderImpl<Trans> implements VecOptimization<Loss> {
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
