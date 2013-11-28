package com.spbsu.ml.methods;

import com.spbsu.commons.func.Computable;
import com.spbsu.commons.func.impl.WeakListenerHolderImpl;
import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.commons.math.vectors.VecTools;
import com.spbsu.commons.math.vectors.impl.ArrayVec;
import com.spbsu.commons.random.RandomExt;
import com.spbsu.ml.Func;
import com.spbsu.ml.VecFunc;
import com.spbsu.ml.data.DataSet;
import com.spbsu.ml.loss.L2;
import com.spbsu.ml.loss.SatL2;
import com.spbsu.ml.models.Ensemble;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

/**
* User: solar
* Date: 21.12.2010
* Time: 22:13:54
*/
public class GradientBoosting<GlobalLoss extends Func> extends WeakListenerHolderImpl<Func> implements Optimization<GlobalLoss> {
  protected final Optimization<L2> weak;
  private final Computable<Vec, L2> factory;
  int iterationsCount;

  double step;

  public GradientBoosting(Optimization<L2> weak, int iterationsCount, double step, Random rnd) {
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
    List<Func> weakModels = new ArrayList<Func>(iterationsCount);
    final VecFunc gradient = globalLoss.gradient();

    for (int t = 0; t < iterationsCount; t++) {
      final Vec gradientValueAtCursor = gradient.vvalue(cursor);
      final L2 localLoss = factory.compute(gradientValueAtCursor);
      final Func weakModel = weak.fit(learn, localLoss);
      weakModels.add(weakModel);
      invoke(new Ensemble(weakModels, -step));
      VecTools.append(cursor, VecTools.scale(weakModel.value(learn.data()), -step));
    }
    return new Ensemble(weakModels, -step);
  }
}
