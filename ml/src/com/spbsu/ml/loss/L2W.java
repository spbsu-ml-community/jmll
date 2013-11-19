package com.spbsu.ml.loss;

import com.spbsu.commons.func.AdditiveGator;
import com.spbsu.commons.func.Factory;
import com.spbsu.commons.math.vectors.Vec;

import static com.spbsu.commons.math.vectors.VecTools.*;

/**
 * User: solar
 * Date: 21.12.2010
 * Time: 22:37:55
 */
public class L2W implements StatBasedOracle<L2W.MSEStats> {
  protected final Vec target;
  protected final Vec weight;

  public L2W(Vec target, Vec weight) {
    this.target = target;
    this.weight = weight;
  }

  @Override
  public Vec gradient(Vec point) {
    Vec result = copy(point);
    scale(result, -1);
    append(result, target);
    scale(result, weight);
    return result;
  }

  @Override
  public int dim() {
    return target.dim();
  }

  public double value(Vec point) {
    Vec temp = copy(point);
    scale(temp, -1);
    append(temp, target);
    scale(temp, weight);
    return Math.sqrt(sum2(temp) / sum2(weight));
  }

  @Override
  public Factory<MSEStats> statsFactory() {
    return new Factory<MSEStats>() {
      @Override
      public MSEStats create() {
        return new MSEStats(target, weight);
      }
    };
  }


  public double value(MSEStats stats) {
    return Math.sqrt(stats.sum2 / stats.weight);
  }

  @Override
  public double score(MSEStats stats) {
    return Math.sqrt((stats.sum2 - stats.sum * stats.sum / stats.weight) / stats.weight);
  }

  public double gradient(MSEStats stats) {
    return stats.sum/stats.weight;
  }

  public static class MSEStats implements AdditiveGator<MSEStats> {
    public double sum;
    public double sum2;
    public double weight;

    private final Vec weights;
    private final Vec targets;

    public MSEStats(Vec target, Vec weight) {
      this.weights = weight;
      this.targets = target;
    }

    @Override
    public synchronized MSEStats remove(int index) {
      final double v = targets.get(index);
      sum -= v;
      sum2 -= v * v;
      weight -= weights.get(index);
      return this;
    }

    @Override
    public synchronized MSEStats remove(MSEStats other) {
      sum -= other.sum;
      sum2 -= other.sum2;
      weight -= other.weight;
      return this;
    }

    @Override
    public synchronized MSEStats append(int index) {
      final double v = targets.get(index);
      sum += v;
      sum2 += v * v;
      weight += weights.get(index);
      return this;
    }

    @Override
    public synchronized MSEStats append(MSEStats other) {
      sum += other.sum;
      sum2 += other.sum2;
      weight += other.weight;
      return this;
    }
  }
}
