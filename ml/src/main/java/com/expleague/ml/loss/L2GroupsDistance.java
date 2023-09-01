//package com.expleague.ml.loss;
//
//import com.expleague.commons.func.AdditiveStatistics;
//import com.expleague.commons.math.FuncC1;
//import com.expleague.commons.math.vectors.Vec;
//import com.expleague.ml.TargetFunc;
//import com.expleague.ml.data.set.DataSet;
//
///**
// * Created by irlab on 22.02.2015.
// */
//public class L2GroupsDistance extends FuncC1.Stub implements AdditiveLoss<L2GroupsDistance.Stat>, TargetFunc {
//  public L2GroupsDistance(final DataSet<?> owner, final Vec targets, final Vec weights) {
//    super(owner, targets, weights);
//  }
//
//  @Override
//  public double bestIncrement(final Stat stats) {
//    return stats.weight > 2 ? stats.sum/stats.weight : 0;
//  }
//
//  @Override
//  public double score(final Stat stats) {
//    final double n = stats.weight;
//    return n > 2 ? n*(n-2)/(n * n - 3 * n + 1) * (stats.sum2 - stats.sum * stats.sum / n) : stats.sum2;
//  }
//
//  public class Stat implements AdditiveStatistics {
//    public double sum;
//    public double sum2;
//    public double weight;
//    public double weight2;
//
//    @Override
//    public Stat remove(final int index, final int times) {
//      final double v = targets.get(index);
//      final double w = weights.get(index) * times;
//      sum -= w * v;
//      sum2 -= w * v * v;
//      weight -= w;
//      weight2 -= w * w;
//      return this;
//    }
//
//    @Override
//    public Stat remove(final AdditiveStatistics otheras) {
//      final Stat other = (Stat) otheras;
//      sum -= other.sum;
//      sum2 -= other.sum2;
//      weight -= other.weight;
//      weight2 -= other.weight2;
//      return this;
//    }
//
//    @Override
//    public Stat append(final int index, final int times) {
//      final double v = targets.get(index);
//      final double w = weights.get(index) * times;
//      sum += w * v;
//      sum2 += w * v * v;
//      weight += w;
//      weight2 += w * w;
//      return this;
//    }
//
//    @Override
//    public Stat append(final AdditiveStatistics otheras) {
//      final Stat other = (Stat) otheras;
//      sum += other.sum;
//      sum2 += other.sum2;
//      weight += other.weight;
//      weight2 += other.weight2;
//      return this;
//    }
//
//    @Override
//    public Stat append(int index, double weight) {
//      final double v = targets.get(index);
//      final double w = weights.get(index) * weight;
//      sum += w * v;
//      sum2 += w * v * v;
//      this.weight += w;
//      weight2 += w * w;
//
//      return this;
//    }
//
//    @Override
//    public Stat remove(int index, double weight) {
//      final double v = targets.get(index);
//      final double w = weights.get(index) * weight;
//      sum -= w * v;
//      sum2 -= w * v * v;
//      this.weight -= w;
//      weight2 -= w * w;
//
//      return this;
//    }
//  }
//}
