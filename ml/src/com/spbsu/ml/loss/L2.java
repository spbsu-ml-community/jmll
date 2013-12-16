package com.spbsu.ml.loss;

import com.spbsu.commons.func.AdditiveStatistics;
import com.spbsu.commons.func.Factory;
import com.spbsu.commons.math.MathTools;
import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.ml.FuncC1;
import com.sun.javafx.beans.annotations.NonNull;

import static com.spbsu.commons.math.vectors.VecTools.*;

/**
 * User: solar
 * Date: 21.12.2010
 * Time: 22:37:55
 */
public class L2 extends FuncC1.Stub implements StatBasedLoss<L2.MSEStats> {
  public final Vec target;

  public L2(Vec target) {
    this.target = target;
  }

  @NonNull
  @Override
  public Vec gradient(Vec x) {
    Vec result = copy(x);
    scale(result, -1);
    append(result, target);
    scale(result, -2);
    return result;
  }

  public int dim() {
    return target.dim();
  }

  public double value(Vec point) {
    Vec temp = copy(point);
    scale(temp, -1);
    append(temp, target);
    return Math.sqrt(sum2(temp) / temp.dim());
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
    return stats.weight > MathTools.EPSILON ? stats.sum/stats.weight : 0;
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
      MSEStats other = (MSEStats)otheras;
      sum -= other.sum;
      sum2 -= other.sum2;
      weight -= other.weight;
      return this;
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
    public MSEStats append(AdditiveStatistics otheras) {
      MSEStats other = (MSEStats)otheras;
      sum += other.sum;
      sum2 += other.sum2;
      weight += other.weight;
      return this;
    }
  }
}
