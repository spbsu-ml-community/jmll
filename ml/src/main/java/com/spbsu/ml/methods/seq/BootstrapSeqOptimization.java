package com.spbsu.ml.methods.seq;

import com.spbsu.commons.func.Computable;
import com.spbsu.commons.func.impl.WeakListenerHolderImpl;
import com.spbsu.commons.math.Trans;
import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.commons.math.vectors.impl.vectors.ArrayVec;
import com.spbsu.commons.random.FastRandom;
import com.spbsu.commons.seq.Seq;
import com.spbsu.ml.data.set.DataSet;
import com.spbsu.ml.data.tools.DataTools;
import com.spbsu.ml.loss.L2;
import com.spbsu.ml.loss.StatBasedLoss;
import com.spbsu.ml.loss.WeightedL2;
import com.spbsu.ml.loss.WeightedLoss;
import com.spbsu.ml.methods.SeqOptimization;

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

