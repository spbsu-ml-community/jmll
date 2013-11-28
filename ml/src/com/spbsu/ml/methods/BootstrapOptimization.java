package com.spbsu.ml.methods;

import com.spbsu.commons.func.impl.WeakListenerHolderImpl;
import com.spbsu.commons.random.RandomExt;
import com.spbsu.ml.Trans;
import com.spbsu.ml.data.DataSet;
import com.spbsu.ml.data.DataTools;
import com.spbsu.ml.loss.StatBasedLoss;

import java.util.Random;

/**
* User: solar
* Date: 21.12.2010
* Time: 22:13:54
*/
public class BootstrapOptimization<Loss extends StatBasedLoss> extends WeakListenerHolderImpl<Trans> implements Optimization<Loss> {
  protected final RandomExt rnd;
  private final Optimization weak;

  public BootstrapOptimization(Optimization weak, Random rnd) {
    this.weak = weak;
    this.rnd = new RandomExt(rnd);
  }

  @Override
  public Trans fit(DataSet learn, Loss globalLoss) {
    return weak.fit(learn, DataTools.bootstrap(globalLoss, rnd));
  }
}
