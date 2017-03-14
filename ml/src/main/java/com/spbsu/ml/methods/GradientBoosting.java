package com.spbsu.ml.methods;

import com.spbsu.commons.func.impl.WeakListenerHolderImpl;
import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.commons.math.vectors.VecTools;
import com.spbsu.commons.math.vectors.impl.vectors.ArrayVec;
import com.spbsu.ml.TargetFunc;
import com.spbsu.commons.math.Trans;
import com.spbsu.ml.data.set.VecDataSet;
import com.spbsu.ml.data.tools.DataTools;
import com.spbsu.ml.func.Ensemble;
import com.spbsu.ml.loss.L2;
import com.spbsu.ml.loss.SatL2;

import java.util.ArrayList;
import java.util.List;

/**
 * User: solar
 * Date: 21.12.2010
 * Time: 22:13:54
 */
public class GradientBoosting<GlobalLoss extends TargetFunc> extends WeakListenerHolderImpl<Trans> implements VecOptimization<GlobalLoss> {
  protected final VecOptimization<L2> weak;
  private final Class<? extends L2> factory;
  int iterationsCount;

  double step;

  public GradientBoosting(final VecOptimization<L2> weak, final int iterationsCount, final double step) {
    this(weak, SatL2.class, iterationsCount, step);
  }

  public GradientBoosting(final VecOptimization<L2> weak, final Class<? extends L2> factory, final int iterationsCount, final double step) {
    this.weak = weak;
    this.factory = factory;
    this.iterationsCount = iterationsCount;
    this.step = step;
  }

  @Override
  public Ensemble fit(final VecDataSet learn, final GlobalLoss globalLoss) {
    final Vec cursor = new ArrayVec(globalLoss.xdim());
    final List<Trans> weakModels = new ArrayList<>(iterationsCount);
    final Trans gradient = globalLoss.gradient();

    for (int t = 0; t < iterationsCount; t++) {
      final Vec gradientValueAtCursor = gradient.trans(cursor);
      final L2 localLoss = DataTools.newTarget(factory, gradientValueAtCursor, learn);
      final Trans weakModel = weak.fit(learn, localLoss);
      weakModels.add(weakModel);
      invoke(new Ensemble(weakModels, -step));
      VecTools.append(cursor, VecTools.scale(weakModel.transAll(learn.data()), -step));
    }
    return new Ensemble(weakModels, -step);
  }
}
