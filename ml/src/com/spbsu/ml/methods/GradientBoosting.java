package com.spbsu.ml.methods;

import com.spbsu.commons.func.Computable;
import com.spbsu.commons.func.impl.WeakListenerHolderImpl;
import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.commons.math.vectors.VecTools;
import com.spbsu.commons.math.vectors.impl.ArrayVec;
import com.spbsu.ml.Func;
import com.spbsu.ml.Trans;
import com.spbsu.ml.data.DataSet;
import com.spbsu.ml.loss.L2;
import com.spbsu.ml.loss.SatL2;
import com.spbsu.ml.func.Ensemble;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

/**
* User: solar
* Date: 21.12.2010
* Time: 22:13:54
*/
public class GradientBoosting<GlobalLoss extends Func> extends WeakListenerHolderImpl<Trans> implements Optimization<GlobalLoss> {
  protected final Optimization<L2> weak;
  private final Computable<Vec, L2> factory;
  int iterationsCount;

  double step;

  public GradientBoosting(Optimization<L2> weak, int iterationsCount, double step) {
    this(weak, new Computable<Vec, L2>() {
      @Override
      public L2 compute(Vec argument) {
        return new SatL2(argument);
      }
    }, iterationsCount, step);
  }

  public GradientBoosting(Optimization<L2> weak, Computable<Vec, L2> factory, int iterationsCount, double step) {
    this.weak = weak;
    this.factory = factory;
    this.iterationsCount = iterationsCount;
    this.step = step;
  }

  @Override
  public Ensemble fit(DataSet learn, GlobalLoss globalLoss) {
    final Vec cursor = new ArrayVec(globalLoss.xdim());
    List<Trans> weakModels = new ArrayList<Trans>(iterationsCount);
    final Trans gradient = globalLoss.gradient();

    for (int t = 0; t < iterationsCount; t++) {
      final Vec gradientValueAtCursor = gradient.trans(cursor);
      final L2 localLoss = factory.compute(gradientValueAtCursor);
      final Trans weakModel = weak.fit(learn, localLoss);
      weakModels.add(weakModel);
      invoke(new Ensemble(weakModels, -step));
      VecTools.append(cursor, VecTools.scale(weakModel.transAll(learn.data()), -step));
    }
    return new Ensemble(weakModels, -step);
  }
}
