package com.spbsu.ml.loss;

import com.spbsu.commons.func.AdditiveStatistics;
import com.spbsu.commons.func.Factory;
import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.commons.math.Func;
import com.spbsu.commons.math.Trans;
import com.spbsu.ml.data.set.DataSet;
import gnu.trove.list.array.TIntArrayList;
import org.jetbrains.annotations.Nullable;

/**
 * User: solar
 * Date: 26.11.13
 * Time: 9:54
 */
public class WeightedLoss<BasedOn extends StatBasedLoss> extends Func.Stub implements StatBasedLoss<WeightedLoss.Stat> {
  private final BasedOn metric;
  private final int[] weights;

  public WeightedLoss(final BasedOn metric, final int[] weights) {
    this.metric = metric;
    this.weights = weights;
  }

  @Override
  public Factory<Stat> statsFactory() {
    return new Factory<Stat>() {
      @Override
      public Stat create() {
        return new Stat(weights, (AdditiveStatistics) metric.statsFactory().create());
      }
    };
  }

  @Override
  public Vec target() {
    return metric.target();
  }

  @Override
  public double bestIncrement(final Stat comb) {
    return metric.bestIncrement(comb.inside);
  }

  @Override
  public double score(final Stat comb) {
    return metric.score(comb.inside);
  }

  @Override
  public double value(final Stat comb) {
    return metric.value(comb.inside);
  }

  @Override
  public int dim() {
    return metric.xdim();
  }

  @Nullable
  @Override
  public Trans gradient() {
    return metric.gradient();
  }

  @Override
  public double value(final Vec x) {
    return metric.trans(x).get(0);
  }

  public double weight(final int index) {
    return weights[index];
  }

  public BasedOn base() {
    return metric;
  }

  @Override
  public DataSet<?> owner() {
    return metric.owner();
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
