package com.spbsu.ml.methods;

import com.spbsu.commons.func.impl.WeakListenerHolderImpl;
import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.commons.random.FastRandom;
import com.spbsu.ml.Trans;
import com.spbsu.ml.data.set.VecDataSet;
import com.spbsu.ml.data.set.VecDataSet;
import com.spbsu.ml.data.tools.DataTools;
import com.spbsu.ml.loss.StatBasedLoss;
import com.spbsu.ml.loss.WeightedLoss;

/**
* User: solar
* Date: 21.12.2010
* Time: 22:13:54
*/
public class BootstrapOptimization<Loss extends StatBasedLoss> extends WeakListenerHolderImpl<Trans> implements VecOptimization<Loss> {
  protected final FastRandom rnd;
  private final VecOptimization<WeightedLoss<? extends Loss>> weak;

  public BootstrapOptimization(VecOptimization<WeightedLoss<? extends Loss>> weak, FastRandom rnd) {
    this.weak = weak;
    this.rnd = rnd;
  }

  @Override
  public Trans fit(VecDataSet learn, Loss globalLoss) {
    return weak.fit(learn, DataTools.bootstrap(globalLoss, rnd));
  }

  @Override
  public Class<Vec> itemClass() {
    return Vec.class;
  }
}
