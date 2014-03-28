package com.spbsu.ml.loss;

import com.spbsu.commons.func.AdditiveStatistics;
import com.spbsu.commons.func.Factory;
import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.ml.Func;
import com.spbsu.ml.Trans;
import org.jetbrains.annotations.Nullable;

/**
 * User: solar
 * Date: 26.11.13
 * Time: 9:54
 */
public class WeightedLoss<BasedOn extends StatBasedLoss> extends Func.Stub implements StatBasedLoss<WeightedLoss.Stat> {
  private final BasedOn metric;
  private final int[] weights;

  public WeightedLoss(BasedOn metric, int[] weights) {
    this.metric = metric;
    this.weights = weights;
  }

  @Override
  public Factory<Stat> statsFactory() {
    return new Factory<Stat>() {
      @Override
      public Stat create() {
        return new Stat(weights, (AdditiveStatistics)metric.statsFactory().create());
      }
    };
  }

  @Override
  public double bestIncrement(Stat comb) {
    return metric.bestIncrement(comb.inside);
  }

  @Override
  public double score(Stat comb) {
    return metric.score(comb.inside);
  }

  @Override
  public double value(Stat comb) {
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
  public double value(Vec x) {
    return metric.trans(x).get(0);
  }

  public static class Stat implements AdditiveStatistics {
    public AdditiveStatistics inside;
    private final int[] weights;

    public Stat(int[] weights, AdditiveStatistics inside) {
      this.weights = weights;
      this.inside = inside;
    }

    @Override
    public Stat append(int index, int times) {
      int count = weights[index];
      inside.append(index, count * times);
      return this;
    }

    @Override
    public Stat append(AdditiveStatistics other) {
      inside.append(((Stat) other).inside);
      return this;
    }

    @Override
    public Stat remove(int index, int times) {
      int count = weights[index];
      inside.remove(index, count * times);
      return this;
    }

    @Override
    public Stat remove(AdditiveStatistics other) {
      inside.remove(((Stat)other).inside);
      return this;
    }
  }
}
