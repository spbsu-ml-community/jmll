package com.spbsu.ml.loss;

import org.jetbrains.annotations.NotNull;


import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.commons.math.vectors.VecTools;
import com.spbsu.commons.math.vectors.impl.vectors.ArrayVec;
import com.spbsu.ml.data.set.DataSet;

import static com.spbsu.commons.math.vectors.VecTools.*;

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

  @NotNull
  @Override
  public Vec gradient(final Vec x) {
    // 2 * (x[i] - target[i]) * weight[i]
    final Vec result = copy(x);
    scale(result, -1);
    append(result, target);
    scale(result, -2);
    scale(result, weights);
    return result;
  }

  @Override
  public double value(final Vec point) {
    // \sqrt{ ( \sum_i (target[i] - point[i])^2 * weight[i] ) / \sum_i weight[i] }
    final Vec x = copy(point);
    scale(x, -1);
    append(x, target);
    scale(x, x);
    scale(x, weights);
    final double sumSquared = sum(x);
    return Math.sqrt(sumSquared / sumWeights);
  }
}
