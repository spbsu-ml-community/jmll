package com.expleague.ml.bayesianEstimation.impl;

import com.expleague.commons.func.AdditiveStatistics;
import com.expleague.commons.math.MathTools;
import com.expleague.commons.math.Trans;
import com.expleague.commons.math.vectors.Vec;
import com.expleague.commons.math.vectors.VecTools;

import java.util.List;


/**
 * Created by noxoomo on 06/11/2017.
 */
public class SufficientStat implements AdditiveStatistics {

  public interface Transform {
    double value(double value);
  }

  public interface Combine {
    double value(double left, double right, double times);
  }

  public static class SumCombine implements Combine {

    @Override
    public double value(final double left, final double right, double times) {
      return left + right * times;
    }
  }

  public static class MulCombine implements Combine {

    @Override
    public double value(final double left, final double right, double times) {
      return left * Math.pow(right, times);
    }
  }

  public static class MaxCombine implements Combine {

    @Override
    public double value(final double left, final double right, double times) {
      return Math.max(left, right);
    }
  }

  public static class MinCombine implements Combine {

    @Override
    public double value(final double left, final double right, double times) {
      return Math.min(left, right);
    }
  }

  public static class IdentityTransform implements Transform {

    @Override
    public double value(final double value) {
      return value;
    }
  }

  public static class ToOneTransform implements Transform {

    @Override
    public double value(final double value) {
      return 1.0;
    }
  }


  public static class LogTransform implements Transform {

    @Override
    public double value(final double value) {
      return Math.log(value);
    }
  }

  public static class ExpTransform implements Transform {
    @Override
    public double value(final double value) {
      return Math.exp(value);
    }
  }

  public static class SqrTransform implements Transform {

    @Override
    public double value(final double value) {
      return MathTools.sqr(value);
    }
  }


  private Vec target;
  private Vec statistic;
  private List<Transform> map;
  private List<Combine> combine;


  @Override
  public AdditiveStatistics append(final int index, final int times) {
    return append(index, (double)times);
  }

  @Override
  public AdditiveStatistics append(final AdditiveStatistics other) {
    VecTools.append(statistic, ((SufficientStat)other).statistic);
    return this;
  }

  @Override
  public AdditiveStatistics remove(final int index, final int times) {

    return null;
  }

  @Override
  public AdditiveStatistics remove(final AdditiveStatistics other) {
    VecTools.scale(statistic, -1.0);
    VecTools.append(statistic, ((SufficientStat)other).statistic);
    VecTools.scale(statistic, -1.0);
    return this;
  }

  @Override
  public AdditiveStatistics append(final int index, final double weight) {
    for (int i = 0; i < map.size(); ++i) {
      final double current = statistic.get(i);
      statistic.set(i, combine.get(i).value(current, target.get(index),  weight));
    }
    return this;
  }

  @Override
  public AdditiveStatistics remove(final int index, final double weight) {
    for (int i = 0; i < map.size(); ++i) {
      final double current = statistic.get(i);
      statistic.set(i, combine.get(i).value(current, target.get(index),  weight));
    }
    return this;
  }
}



