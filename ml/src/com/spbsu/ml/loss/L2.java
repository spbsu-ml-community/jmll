package com.spbsu.ml.loss;

import com.spbsu.commons.func.AdditiveGator;
import com.spbsu.commons.func.Computable;
import com.spbsu.commons.func.Factory;
import com.spbsu.commons.math.MathTools;
import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.ml.Oracle1;

import static com.spbsu.commons.math.vectors.VecTools.*;

/**
 * User: solar
 * Date: 21.12.2010
 * Time: 22:37:55
 */
public class L2 implements StatBasedOracle<L2.MSEStats>, Oracle1 {
  public static final Computable<Vec, L2> FACTORY = new Computable<Vec, L2>() {
    @Override
    public L2 compute(Vec argument) {
      return new L2(argument);
    }
  };
  protected final Vec target;

  public L2(Vec target) {
    this.target = target;
  }

  @Override
  public Vec gradient(Vec point) {
    Vec result = copy(point);
    scale(result, -1);
    append(result, target);
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

  public double gradient(MSEStats stats) {
    return stats.weight > MathTools.EPSILON ? stats.sum/stats.weight : 0;
  }

  public static class MSEStats implements AdditiveGator<MSEStats> {
    public double sum;
    public double sum2;
    public int weight;

    private final Vec targets;

    public MSEStats(Vec target) {
      this.targets = target;
    }

    @Override
    public synchronized MSEStats remove(int index) {
      final double v = targets.get(index);
      sum -= v;
      sum2 -= v * v;
      weight--;
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
      weight++;
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
