package com.spbsu.ml.methods;

import com.spbsu.commons.func.impl.WeakListenerHolderImpl;
import com.spbsu.commons.random.FastRandom;
import com.spbsu.commons.math.Trans;
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

  public BootstrapOptimization(final VecOptimization<WeightedLoss<? extends Loss>> weak, final FastRandom rnd) {
    this.weak = weak;
    this.rnd = rnd;
  }

  @Override
  public Trans fit(final VecDataSet learn, final Loss globalLoss) {
    return weak.fit(learn, DataTools.bootstrap(globalLoss, rnd));
  }
}
