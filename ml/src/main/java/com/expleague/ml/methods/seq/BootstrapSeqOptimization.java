package com.expleague.ml.methods.seq;

import com.expleague.commons.func.impl.WeakListenerHolderImpl;
import com.expleague.commons.math.Trans;
import com.expleague.commons.math.vectors.Vec;
import com.expleague.commons.math.vectors.impl.vectors.ArrayVec;
import com.expleague.commons.random.FastRandom;
import com.expleague.commons.seq.Seq;
import com.expleague.ml.data.set.DataSet;
import com.expleague.ml.loss.L2;
import com.expleague.ml.loss.WeightedL2;
import com.expleague.ml.methods.SeqOptimization;

import java.util.function.Function;

public class BootstrapSeqOptimization<T, Loss extends L2> extends WeakListenerHolderImpl<Trans> implements SeqOptimization<T, Loss> {
  protected final FastRandom rnd;
  private final SeqOptimization<T, ? super WeightedL2> weak;
  private int ydim;

  public BootstrapSeqOptimization(final SeqOptimization<T, ? super WeightedL2> weak, final FastRandom rnd) {
    this.weak = weak;
    this.rnd = rnd;
    this.ydim = 1;
  }

  public BootstrapSeqOptimization(final SeqOptimization<T, ? super WeightedL2> weak, final FastRandom rnd, int ydim) {
    this.weak = weak;
    this.rnd = rnd;
    this.ydim = ydim;
  }

  @Override
  public Function<Seq<T>, Vec> fit(final DataSet<Seq<T>> learn, final Loss globalLoss) {
    return weak.fit(learn, bootstrap(globalLoss, rnd));
  }

  private WeightedL2 bootstrap(Loss loss, FastRandom rnd) {
    final double[] poissonWeights = new double[loss.xdim()];
    for (int i = 0; i < loss.dim() / ydim; i++) {
      int w = rnd.nextPoisson(1.);
      for (int j = 0; j < ydim; j++) {
        poissonWeights[i * ydim + j] = w;
      }
    }
    return new WeightedL2(loss.target, loss.owner(), new ArrayVec(poissonWeights));
  }
}

