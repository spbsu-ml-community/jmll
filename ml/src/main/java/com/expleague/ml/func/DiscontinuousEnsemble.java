package com.expleague.ml.func;

import com.expleague.commons.math.DiscontinuousTrans;
import com.expleague.commons.math.vectors.Vec;
import com.expleague.commons.math.vectors.VecTools;
import com.expleague.commons.math.vectors.impl.vectors.ArrayVec;
import org.jetbrains.annotations.NotNull;

import java.util.List;

public class DiscontinuousEnsemble<X extends DiscontinuousTrans> extends Ensemble<X> implements DiscontinuousTrans {
  public DiscontinuousEnsemble(X[] models, Vec weights) {
    super(models, weights);
  }

  public DiscontinuousEnsemble(List<X> weakModels, double step) {
    super(weakModels, step);
  }

  @NotNull
  @Override
  public Vec left(Vec x) {
    final Vec result = new ArrayVec(ydim());
    return leftTo(x, result);
  }

  @NotNull
  @Override
  public Vec right(Vec x) {
    final Vec result = new ArrayVec(ydim());
    return rightTo(x, result);
  }

  @NotNull
  @Override
  public Vec leftTo(Vec x, Vec to) {
    for (int i = 0; i < models.length; i++) {
      VecTools.append(to, VecTools.scale(models[i].left(x), weights.get(i)));
    }
    return to;
  }

  @NotNull
  @Override
  public Vec rightTo(Vec x, Vec to) {
    for (int i = 0; i < models.length; i++) {
      VecTools.append(to, VecTools.scale(models[i].right(x), weights.get(i)));
    }
    return to;
  }
}
