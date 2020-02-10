package com.expleague.ml.loss;

import com.expleague.commons.func.AdditiveStatistics;
import com.expleague.commons.func.Factory;
import com.expleague.commons.math.FuncC1;
import com.expleague.commons.math.MathTools;
import com.expleague.commons.math.vectors.Vec;
import com.expleague.commons.math.vectors.VecTools;
import com.expleague.ml.data.set.DataSet;
import com.expleague.ml.TargetFunc;
import org.jetbrains.annotations.NotNull;

import java.util.function.IntFunction;

/**
 * User: solar
 * Date: 21.12.2010
 * Time: 22:37:55
 */
public class L2 extends FuncC1.Stub implements AdditiveLoss<L2.Stat>, TargetFunc {
  public final Vec target;
  private final DataSet<?> owner;

  public L2(final Vec target, final DataSet<?> owner) {
    this.target = target;
    this.owner = owner;
  }

  public static double weight(final AdditiveStatistics stats) {
    if (stats instanceof WeightedLoss.Stat)
      return ((Stat) ((WeightedLoss.Stat) stats).inside).weight;
    if (stats instanceof Stat)
      return ((Stat) stats).weight;
    throw new IllegalArgumentException();
  }

  public static double sum(final AdditiveStatistics stats) {
    if (stats instanceof WeightedLoss.Stat)
      return ((Stat) ((WeightedLoss.Stat) stats).inside).sum;
    if (stats instanceof Stat)
      return ((Stat) stats).sum;
    throw new IllegalArgumentException();
  }

  public static double sum2(final AdditiveStatistics stats) {
    if (stats instanceof WeightedLoss.Stat)
      return ((Stat) ((WeightedLoss.Stat) stats).inside).sum2;
    if (stats instanceof Stat)
      return ((Stat) stats).sum2;
    throw new IllegalArgumentException();
  }

  @NotNull
  @Override
  public Vec gradient(final Vec x) {
    final Vec result = VecTools.copy(x);
    VecTools.scale(result, -1);
    VecTools.append(result, target);
    VecTools.scale(result, -2);
    return result;
  }

  @Override
  public int dim() {
    return target.dim();
  }

  @Override
  public double value(final Vec point) {
    final Vec temp = VecTools.copy(point);
    VecTools.scale(temp, -1);
    VecTools.append(temp, target);
    return Math.sqrt(VecTools.sum2(temp) / temp.dim());
  }

  @Override
  public IntFunction<Stat> statsFactory() {
    return (findex) -> new Stat(target);
  }

  @Override
  public int components() {
    return target.dim();
  }

  @Override
  public double value(int component, double x) {
    return MathTools.sqr(target.get(component) - x);
  }

  public Vec target() {
    return target;
  }

  @Override
  public double value(final Stat stats) {
    return stats.sum2;
  }

  @Override
  public double score(final Stat stats) {
    return stats.weight > MathTools.EPSILON ? (-stats.sum * stats.sum / stats.weight)/* + 5 * stats.weight2*/ : 0/*+ 5 * stats.weight2*/;
  }

  @Override
  public double bestIncrement(final Stat stats) {
    return stats.weight > MathTools.EPSILON ? stats.sum / stats.weight : 0;
  }

  public double get(final int i) {
    return target.get(i);
  }

  @Override
  public DataSet<?> owner() {
    return owner;
  }

  public static class Stat implements AdditiveStatistics {
    public double sum;
    public double sum2;
    public double weight;
    public double weight2;

    private final Vec targets;

    public Stat(final Vec target) {
      this.targets = target;
    }

    @Override
    public Stat remove(final int index, final int times) {
      final double v = targets.get(index);
      sum -= times * v;
      sum2 -= times * v * v;
      weight -= times;
      weight2 -= times * times;
      return this;
    }


    public Stat remove(final int index, final int times, final double p) {
      final double v = targets.get(index);
      sum -= p * times * v;
      sum2 -= p * times * v * v;
      weight -= p * times;
      weight2 -= p * times * p * times;
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
      sum += times * v;
      sum2 += times * v * v;
      weight += times;
      weight2 += times * times;
      return this;
    }

    public Stat append(final int index, final int times, final double p) {
      final double v = targets.get(index);
      sum += p * times * v;
      sum2 += p * times * v * v;
      weight += p * times;
      weight2 += p * times * times;
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
    public Stat append(int index, double w) {
      final double v = targets.get(index);
      sum += w * v;
      sum2 += w * v * v;
      weight += w;
      weight2 += w * w;

      return this;
    }

    @Override
    public Stat remove(int index, double w) {
      final double v = targets.get(index);
      sum -= w * v;
      sum2 -= w * v * v;
      weight -= w;
      weight2 -= w * w;

      return this;
    }
  }
}
