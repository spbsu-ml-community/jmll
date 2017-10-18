package com.expleague.ml.methods;

import com.expleague.commons.func.impl.WeakListenerHolderImpl;
import com.expleague.commons.random.FastRandom;
import com.expleague.commons.math.Trans;
import com.expleague.ml.data.set.VecDataSet;
import com.expleague.ml.data.tools.DataTools;
import com.expleague.ml.loss.StatBasedLoss;
import com.expleague.ml.loss.WeightedLoss;

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
