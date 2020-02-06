package com.expleague.ml.loss;

import com.expleague.commons.func.AdditiveStatistics;
import com.expleague.commons.math.Func;
import com.expleague.commons.math.Trans;
import com.expleague.ml.data.set.DataSet;
import gnu.trove.list.array.TIntArrayList;
import org.jetbrains.annotations.Nullable;

import java.util.function.IntFunction;
import java.util.stream.DoubleStream;
import java.util.stream.IntStream;

/**
 * User: solar
 * Date: 26.11.13
 * Time: 9:54
 */
public class WeightedLoss<BasedOn extends AdditiveLoss> extends Func.Stub implements AdditiveLoss<WeightedLoss.Stat> {
  private final BasedOn baseLoss;
  private final int[] weights;

  public WeightedLoss(final BasedOn baseLoss, final int[] weights) {
    this.baseLoss = baseLoss;
    this.weights = weights;
  }

  @Override
  public IntFunction<Stat> statsFactory() {
    return (findex) -> new Stat(weights, (AdditiveStatistics) baseLoss.statsFactory().apply(findex));
  }

  @Override
  public int components() {
    return weights.length;
  }

  @Override
  public IntStream nzComponents() {
    return IntStream.range(0, components()).filter(i -> weights[i] != 0);
  }

  @Override
  public double value(int component, double x) {
    return weights[component] != 0 ? baseLoss.value(component, x) * weights[component] : 0;
  }

  @Override
  public double bestIncrement(final Stat comb) {
    return baseLoss.bestIncrement(comb.inside);
  }

  @Override
  public double score(final Stat comb) {
    return baseLoss.score(comb.inside);
  }

  @Override
  public double value(final Stat comb) {
    return baseLoss.value(comb.inside);
  }

  @Override
  public int dim() {
    return baseLoss.xdim();
  }

  @Nullable
  @Override
  public Trans gradient() {
    return baseLoss.gradient();
  }

  public double weight(final int index) {
    return weights[index];
  }

  public double[] weights() {
    return IntStream.of(weights).mapToDouble(w -> (double)w).toArray();
  }

  public BasedOn base() {
    return baseLoss;
  }

  @Override
  public DataSet<?> owner() {
    return baseLoss.owner();
  }

  public int[] points() {
    final TIntArrayList result = new TIntArrayList(weights.length + 1000); // Julian ????
    for(int i = 0; i < weights.length; i++) {
      if (weights[i] > 0)
        result.add(i);
    }
    return result.toArray();
  }

  public int[] zeroPoints() {
    final TIntArrayList result = new TIntArrayList(weights.length);
    for(int i = 0; i < weights.length; i++) {
      if (weights[i] == 0)
        result.add(i);
    }
    return result.toArray();
  }

  public static class Stat implements AdditiveStatistics {
    public AdditiveStatistics inside;
    private final int[] weights;

    public Stat(final int[] weights, final AdditiveStatistics inside) {
      this.weights = weights;
      this.inside = inside;
    }

    @Override
    public Stat append(final int index, final int times) {
      final int count = weights[index];
      inside.append(index, count * times);
      return this;
    }

    @Override
    public Stat append(final AdditiveStatistics other) {
      inside.append(((Stat) other).inside);
      return this;
    }

    @Override
    public Stat remove(final int index, final int times) {
      final int count = weights[index];
      inside.remove(index, count * times);
      return this;
    }

    @Override
    public Stat remove(final AdditiveStatistics other) {
      inside.remove(((Stat) other).inside);
      return this;
    }

    @Override
    public Stat append(int index, double weight) {
      final int count = weights[index];
      inside.append(index,weight*count);
      return this;
    }

    @Override
    public Stat remove(int index, double weight) {
      final int count = weights[index];
      inside.remove(index,weight*count);
      return this;
    }
  }
}
