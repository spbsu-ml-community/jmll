package com.spbsu.ml.methods.seq;

import com.spbsu.commons.func.Computable;
import com.spbsu.commons.func.impl.WeakListenerHolderImpl;
import com.spbsu.commons.math.Trans;
import com.spbsu.commons.math.vectors.SingleValueVec;
import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.commons.math.vectors.VecTools;
import com.spbsu.commons.math.vectors.impl.vectors.ArrayVec;
import com.spbsu.commons.seq.Seq;
import com.spbsu.ml.TargetFunc;
import com.spbsu.ml.data.set.DataSet;
import com.spbsu.ml.data.tools.DataTools;
import com.spbsu.ml.loss.L2;
import com.spbsu.ml.methods.SeqOptimization;

import java.util.ArrayList;
import java.util.List;


public class GradientSeqBoosting<T, GlobalLoss extends TargetFunc> extends WeakListenerHolderImpl<Computable<Seq<T>, Vec>> implements SeqOptimization<T, GlobalLoss> {
  protected final SeqOptimization<T, L2> weak;
  private final Class<? extends L2> factory;
  int iterationsCount;

  double step;

  public GradientSeqBoosting(final SeqOptimization<T, L2> weak, final int iterationsCount, final double step) {
    this(weak, L2.class, iterationsCount, step);
  }

  public GradientSeqBoosting(final SeqOptimization<T, L2> weak, final Class<? extends L2> factory, final int iterationsCount, final double step) {
    this.weak = weak;
    this.factory = factory;
    this.iterationsCount = iterationsCount;
    this.step = step;
  }

  @Override
  public Computable<Seq<T>, Vec> fit(final DataSet<Seq<T>> learn, final GlobalLoss globalLoss) {
    final Vec cursor = new ArrayVec(globalLoss.xdim());
    final List<Computable<Seq<T>, Vec>> weakModels = new ArrayList<>(iterationsCount);
    final Trans gradient = globalLoss.gradient();
    for (int t = 0; t < iterationsCount; t++) {
      final Vec gradientValueAtCursor = gradient.trans(cursor);
      final L2 localLoss = DataTools.newTarget(factory, gradientValueAtCursor, learn);
      System.out.println("Iteration " + t + ". Gradient norm: " + VecTools.norm(localLoss.target));
      final Computable<Seq<T>, Vec> weakModel = weak.fit(learn, localLoss);
      weakModels.add(weakModel);
      final Computable<Seq<T>, Vec> curRes = getResult(new ArrayList<>(weakModels));
      invoke(curRes);
      for (int i = 0; i < learn.length(); i++) {
        cursor.adjust(i, weakModel.compute(learn.at(i)).get(0) * -step);
      }
    }
    return getResult(weakModels);
  }

  private Computable<Seq<T>, Vec> getResult(final List<Computable<Seq<T>, Vec>> weakModels) {
    return argument -> {
      double result = 0;
      for (Computable<Seq<T>, Vec> model: weakModels) {
        result += model.compute(argument).get(0) * -step;
      }
      return new SingleValueVec(result);
    };
  }
}
