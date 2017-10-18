package com.expleague.ml.loss;

import com.expleague.commons.math.vectors.Vec;
import com.expleague.commons.math.vectors.VecTools;
import com.expleague.commons.math.vectors.impl.vectors.ArrayVec;
import org.jetbrains.annotations.NotNull;


import com.expleague.ml.data.set.DataSet;

/**
 * Created by irlab on 22.02.2015.
 */
public class WeightedL2 extends L2 {
  private Vec weights;
  private double sumWeights;

  public WeightedL2(final Vec target, final DataSet<?> owner) {
    this(target, owner, VecTools.fill(new ArrayVec(target.dim()), 1));
  }

  public WeightedL2(final Vec target, final DataSet<?> owner, final Vec weights) {
    super(target, owner);
    this.weights = weights;
    this.sumWeights = VecTools.sum(weights);
  }

  public void setWeights(final Vec weights) {
    this.weights = weights;
    this.sumWeights = VecTools.sum(weights);
  }

  public Vec getWeights() {
    return weights;
  }

    @NotNull
  @Override
  public Vec gradient(final Vec x) {
    // 2 * (x[i] - target[i]) * weight[i]
    final Vec result = VecTools.copy(x);
    VecTools.scale(result, -1);
    VecTools.append(result, target);
    VecTools.scale(result, -2);
    VecTools.scale(result, weights);
    return result;
  }

  @Override
  public double value(final Vec point) {
    // \sqrt{ ( \sum_i (target[i] - point[i])^2 * weight[i] ) / \sum_i weight[i] }
    final Vec x = VecTools.copy(point);
    VecTools.scale(x, -1);
    VecTools.append(x, target);
    VecTools.scale(x, x);
    VecTools.scale(x, weights);
    final double sumSquared = VecTools.sum(x);
    return Math.sqrt(sumSquared / sumWeights);
  }
}
