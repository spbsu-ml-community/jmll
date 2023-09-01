package com.expleague.ml.loss;

import com.expleague.commons.func.AdditiveStatistics;
import com.expleague.commons.math.FuncC1;
import com.expleague.commons.math.MathTools;
import com.expleague.commons.math.vectors.Vec;
import com.expleague.commons.math.vectors.VecTools;
import com.expleague.commons.math.vectors.impl.vectors.ArrayVec;
import com.expleague.ml.TargetFunc;


import com.expleague.ml.data.set.DataSet;

import java.util.function.IntFunction;

/**
 * Created by irlab on 22.02.2015.
 */
public class WeightedL2 extends FuncC1.Stub implements AdditiveLoss<WeightedL2.Stat>, TargetFunc {
  private final Vec targets;
  private final DataSet<?> owner;
  private Vec weights;
  private double sumWeights;

  public WeightedL2(final Vec targets, final DataSet<?> owner) {
    this(owner, targets, VecTools.fill(new ArrayVec(targets.dim()), 1));
  }

  public WeightedL2(final DataSet<?> owner, final Vec targets, final Vec weights) {
    this.owner = owner;
    this.targets = targets;
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

  @Override
  public int dim() {
    return targets.dim();
  }

  @Override
  public int components() {
    return targets.dim();
  }

  @Override
  public DataSet<?> owner() {
    return owner;
  }

  public Vec target() {
    return targets;
  }

  @Override
  public double value(int component, double x) {
    return MathTools.sqr(targets.get(component) - x) * weights.get(component);
  }

  @Override
  public double value(Stat comb) {
    return comb.sum2;
  }

  @Override
  public Vec gradient(final Vec x) {
    // 2 * (x[i] - target[i]) * weight[i]
    final Vec result = VecTools.copy(x);
    VecTools.scale(result, -1);
    VecTools.append(result, targets);
    VecTools.scale(result, -2);
    VecTools.scale(result, weights);
    return result;
  }

  @Override
  public double value(final Vec point) {
    // \sqrt{ ( \sum_i (target[i] - point[i])^2 * weight[i] ) / \sum_i weight[i] }
    final Vec x = VecTools.copy(point);
    VecTools.scale(x, -1);
    VecTools.append(x, targets);
    VecTools.scale(x, x);
    VecTools.scale(x, weights);
    final double sumSquared = VecTools.sum(x);
    return Math.sqrt(sumSquared / sumWeights);
  }

  @Override
  public IntFunction<Stat> statsFactory() {
    return findex -> new Stat();
  }

  @Override
  public double score(final Stat stats) {
    return stats.weight > MathTools.EPSILON ? (-stats.sum * stats.sum / stats.weight)/* + 5 * stats.weight2*/ : 0/*+ 5 * stats.weight2*/;
  }

  @Override
  public double bestIncrement(final Stat stats) {
    return stats.weight > MathTools.EPSILON ? stats.sum / stats.weight : 0;
  }

  public class Stat implements AdditiveStatistics {
    public double sum;
    public double sum2;
    public double weight;
    public double weight2;

    @Override
    public Stat remove(final int index, final int times) {
      final double v = targets.get(index);
      final double w = weights.get(index) * times;
      sum -= w * v;
      sum2 -= w * v * v;
      weight -= w;
      weight2 -= w * w;
      return this;
    }

    @Override
    public Stat remove(final AdditiveStatistics otheras) {
      final Stat other = (Stat) otheras;
      sum -= other.sum;
      sum2 -= other.sum2;
      weight -= other.weight;
      weight2 -= other.weight2;
      return this;
    }

    @Override
    public Stat append(final int index, final int times) {
      final double v = targets.get(index);
      final double w = weights.get(index) * times;
      sum += w * v;
      sum2 += w * v * v;
      weight += w;
      weight2 += w * w;
      return this;
    }

    @Override
    public Stat append(final AdditiveStatistics otheras) {
      final Stat other = (Stat) otheras;
      sum += other.sum;
      sum2 += other.sum2;
      weight += other.weight;
      weight2 += other.weight2;
      return this;
    }

    @Override
    public Stat append(int index, double weight) {
      final double v = targets.get(index);
      final double w = weights.get(index) * weight;
      sum += w * v;
      sum2 += w * v * v;
      this.weight += w;
      weight2 += w * w;

      return this;
    }

    @Override
    public Stat remove(int index, double weight) {
      final double v = targets.get(index);
      final double w = weights.get(index) * weight;
      sum -= w * v;
      sum2 -= w * v * v;
      this.weight -= w;
      weight2 -= w * w;

      return this;
    }
  }
}
