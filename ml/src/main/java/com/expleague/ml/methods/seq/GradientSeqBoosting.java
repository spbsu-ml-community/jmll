package com.expleague.ml.methods.seq;

import com.expleague.commons.func.impl.WeakListenerHolderImpl;
import com.expleague.commons.math.Trans;
import com.expleague.commons.math.vectors.Vec;
import com.expleague.commons.math.vectors.VecTools;
import com.expleague.commons.math.vectors.impl.vectors.ArrayVec;
import com.expleague.commons.seq.Seq;
import com.expleague.ml.TargetFunc;
import com.expleague.ml.data.set.DataSet;
import com.expleague.ml.data.tools.DataTools;
import com.expleague.ml.loss.L2;
import com.expleague.ml.methods.SeqOptimization;

import java.util.ArrayList;
import java.util.List;
import java.util.function.Function;


public class GradientSeqBoosting<T, GlobalLoss extends TargetFunc> extends WeakListenerHolderImpl<Function<Seq<T>,Vec>> implements SeqOptimization<T, GlobalLoss> {
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
  public Function<Seq<T>, Vec> fit(final DataSet<Seq<T>> learn, final GlobalLoss globalLoss) {
    final Vec cursor = new ArrayVec(globalLoss.xdim());
    final List<Function<Seq<T>,Vec>> weakModels = new ArrayList<>(iterationsCount);
    final Trans gradient = globalLoss.gradient();
    for (int t = 0; t < iterationsCount; t++) {
      assert gradient != null;
      final Vec gradientValueAtCursor = gradient.trans(cursor);
      final L2 localLoss = DataTools.newTarget(factory, gradientValueAtCursor, learn);
      System.out.println("Iteration " + t + ". Gradient norm: " + VecTools.norm(localLoss.target));
      final Function<Seq<T>, Vec> weakModel = weak.fit(learn, localLoss);
      weakModels.add(weakModel);
      final Function<Seq<T>,Vec> curRes = getResult(new ArrayList<>(weakModels));
      invoke(curRes);
      for (int i = 0; i < learn.length(); i++) {
        cursor.adjust(i, weakModel.apply(learn.at(i)).get(0) * -step);
      }
    }
    return getResult(weakModels);
  }

  private Function<Seq<T>, Vec> getResult(final List<Function<Seq<T>,Vec>> weakModels) {
    return argument -> {
      Vec result = null;
      for (Function<Seq<T>, Vec> model: weakModels) {
        if (result == null) {
          result = model.apply(argument);
        } else {
          VecTools.append(result, model.apply(argument));
        }
      }
      VecTools.scale(result, -step);
      return result;
    };
  }
}
