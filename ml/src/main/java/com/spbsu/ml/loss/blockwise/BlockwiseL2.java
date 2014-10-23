package com.spbsu.ml.loss.blockwise;

import com.spbsu.commons.func.AdditiveStatistics;
import com.spbsu.commons.func.Factory;
import com.spbsu.commons.math.MathTools;
import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.ml.BlockwiseFuncC1;
import com.spbsu.ml.TargetFunc;
import com.spbsu.ml.data.set.DataSet;

/**
 * User: solar
 * Date: 21.12.2010
 * Time: 22:37:55
 */
public class BlockwiseL2 extends BlockwiseFuncC1.Stub implements BlockwiseStatBasedLoss<BlockwiseL2.MSEStats>, TargetFunc {
  public final Vec target;
  private final DataSet<?> owner;

  public BlockwiseL2(Vec target, DataSet<?> owner) {
    this.target = target;
    this.owner = owner;
  }

  public int dim() {
    return target.dim();
  }

  @Override
  public void gradient(final Vec pointBlock, final Vec result, final int index) {
    final int blockSize = blockSize();
    for (int i = 0; i < blockSize; i++) {
      result.set(i, 2 * (pointBlock.get(i) - target.get(index * blockSize + i)));
    }
  }

  @Override
  public double value(final Vec pointBlock, final int index) {
    double result = 0.0;
    final int blockSize = blockSize();
    for (int i = 0; i < blockSize; i++) {
      double val = pointBlock.get(i) - target.get(index * blockSize + i);
      result += val * val;
    }
    return result;
  }

  @Override
  public double transformResultValue(final double value) {
    return Math.sqrt(value / dim());
  }

  @Override
  public int blockSize() {
    return 1;
  }

  @Override
  public Factory<MSEStats> statsFactory() {
    return new Factory<MSEStats>() {
      @Override
      public MSEStats create() {
        return new MSEStats(target);
      }
    };
  }

  public double value(MSEStats stats) {
    return stats.sum2;
  }

  @Override
  public double score(MSEStats stats) {
    return stats.weight > MathTools.EPSILON ? (stats.sum2 - stats.sum * stats.sum / stats.weight) : stats.sum2;
  }

  public double bestIncrement(MSEStats stats) {
    return stats.weight > MathTools.EPSILON ? stats.sum / stats.weight : 0;
  }

  public double get(final int i) {
    return target.get(i);
  }

  @Override
  public DataSet<?> owner() {
    return owner;
  }

  public static class MSEStats implements AdditiveStatistics {
    public volatile double sum;
    public volatile double sum2;
    public volatile int weight;

    private final Vec targets;

    public MSEStats(Vec target) {
      this.targets = target;
    }

    @Override
    public MSEStats remove(int index, int times) {
      final double v = targets.get(index);
      sum -= times * v;
      sum2 -= times * v * v;
      weight -= times;
      return this;
    }

    @Override
    public MSEStats remove(AdditiveStatistics otheras) {
      MSEStats other = (MSEStats) otheras;
      sum -= other.sum;
      sum2 -= other.sum2;
      weight -= other.weight;
      return this;
    }

    @Override
    public AdditiveStatistics remove(int index, double p, int times) {
      double v = targets.get(index);
      sum -= times * p * v;
      sum2 -= times * p * v * v;
      weight -= times * p;
      return this;
    }

    @Override
    public AdditiveStatistics remove(int index, double p) {
      return remove(index, p, 1);
    }

    @Override
    public MSEStats append(int index, int times) {
      final double v = targets.get(index);
      sum += times * v;
      sum2 += times * v * v;
      weight += times;
      return this;
    }

    @Override
    public AdditiveStatistics append(int index, double prob, int times) {
      final double v = targets.get(index);
      sum += times * prob * v;
      sum2 += times * prob * v * v;
      weight += times * prob;
      return this;
    }

    @Override
    public AdditiveStatistics append(int index, double prob) {
      return append(index, prob, 1);
    }

    @Override
    public MSEStats append(AdditiveStatistics otheras) {
      MSEStats other = (MSEStats) otheras;
      sum += other.sum;
      sum2 += other.sum2;
      weight += other.weight;
      return this;
    }

    public Vec getTargets() {
      return targets;
    }
  }
}
