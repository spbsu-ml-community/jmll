package com.expleague.ml.methods.seq;

import com.expleague.commons.func.Computable;
import com.expleague.commons.math.vectors.impl.vectors.ArrayVec;
import com.expleague.commons.func.impl.WeakListenerHolderImpl;
import com.expleague.commons.math.Trans;
import com.expleague.commons.math.vectors.Vec;
import com.expleague.commons.random.FastRandom;
import com.expleague.commons.seq.Seq;
import com.expleague.ml.data.set.DataSet;
import com.expleague.ml.loss.L2;
import com.expleague.ml.loss.WeightedL2;
import com.expleague.ml.methods.SeqOptimization;

public class BootstrapSeqOptimization<T, Loss extends L2> extends WeakListenerHolderImpl<Trans> implements SeqOptimization<T, Loss> {
  protected final FastRandom rnd;
  private final SeqOptimization<T, ? super WeightedL2> weak;

  public BootstrapSeqOptimization(final SeqOptimization<T, ? super WeightedL2> weak, final FastRandom rnd) {
    this.weak = weak;
    this.rnd = rnd;
  }

  @Override
  public Computable<Seq<T>, Vec> fit(final DataSet<Seq<T>> learn, final Loss globalLoss) {
    return weak.fit(learn, bootstrap(globalLoss, rnd));
  }

  private WeightedL2 bootstrap(Loss loss, FastRandom rnd) {
    final double[] poissonWeights = new double[loss.xdim()];
    for (int i = 0; i < loss.xdim(); i++) {
      poissonWeights[i] = rnd.nextPoisson(1.);
    }
    return new WeightedL2(loss.target, loss.owner(), new ArrayVec(poissonWeights));
  }
}

