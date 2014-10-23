package com.spbsu.ml.methods.trees;

import com.spbsu.commons.func.AdditiveStatistics;
import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.commons.random.FastRandom;
import com.spbsu.ml.BFGrid;
import com.spbsu.ml.data.Aggregate;
import com.spbsu.ml.data.impl.BinarizedDataSet;
import com.spbsu.ml.loss.StatBasedLoss;
import gnu.trove.list.array.TIntArrayList;

import java.util.Arrays;


/**
 * User: solar
 * Date: 10.09.13
 * Time: 12:16
 */
@SuppressWarnings("unchecked")
public class BFOptimizationSubset {
  private final BinarizedDataSet bds;
  private int[] points;
  private final StatBasedLoss<AdditiveStatistics> oracle;
  private final Aggregate aggregate;

  public BFOptimizationSubset(BinarizedDataSet bds, StatBasedLoss oracle, int[] points) {
    this.bds = bds;
    this.points = points;
    this.oracle = oracle;
    this.aggregate = new Aggregate(bds, oracle.statsFactory(), points);
  }

  public BFOptimizationSubset split(BFGrid.BinaryFeature feature) {
    TIntArrayList left = new TIntArrayList(points.length);
    TIntArrayList right = new TIntArrayList(points.length);
    final byte[] bins = bds.bins(feature.findex);
    for (int i : points) {
      if (bins[i] <= feature.binNo) {
        left.add(i);
      } else {
        right.add(i);
      }
    }
    final BFOptimizationSubset rightBro = new BFOptimizationSubset(bds, oracle, right.toArray());
    aggregate.remove(rightBro.aggregate);
    points = left.toArray();
    return rightBro;
  }


  public void reweight(BFGrid.BinaryFeature feature, boolean mask, double[] weights) {
    final byte[] bins = bds.bins(feature.findex);
    double totalWeight = 0;
    for (int i = 0; i < points.length; ++i) {
      final int index = points[i];
      if (bins[index] > feature.binNo != mask) {
//        failedCount[i]++;
        weights[i] *= mask ? 1 - feature.prob : feature.prob;
        totalWeight += weights[i];
      }
    }
    for (int i = 0; i < weights.length; ++i) {
      weights[i] /= totalWeight;
    }
  }


  public int size() {
    return points.length;
  }

  public void visitAllSplits(Aggregate.SplitVisitor<? extends AdditiveStatistics> visitor) {
    aggregate.visit(visitor);
  }

  public <T extends AdditiveStatistics> void visitSplit(BFGrid.BinaryFeature bf, Aggregate.SplitVisitor<T> visitor) {
    final T left = (T) aggregate.combinatorForFeature(bf.bfIndex);
    final T right = (T) oracle.statsFactory().create().append(aggregate.total()).remove(left);
    visitor.accept(bf, left, right);
  }

  public AdditiveStatistics total() {
    return aggregate.total();
  }

  FastRandom rand = new FastRandom();

  public double regularizedIncrement(Vec target) {

    double[] sample = new double[points.length];
    for (int i = 0; i < points.length; ++i) {
      sample[i] = target.get(points[i]);
    }
    double sum = 0;
    double weight = 0;
    if (points.length < 4) {
//      return sample[rand.nextInt(sample.length)];
      for (double elem : sample) {
        sum += elem;
      }
      return sum / points.length;
    }
//    for (int i = 0; i < points.length; ++i) {
//      sum += sample[i];
//    }
//    sum /= points.length;
//    return sum;
    Arrays.sort(sample);
    int left = (int) (Math.ceil(sample.length * 0.01));
    int right = (int) (Math.ceil(sample.length * 0.99));
    while (left > 0 && sample[left] == sample[left - 1]) --left;
    while (right < sample.length && sample[right] == sample[right - 1]) ++right;

    for (int i = left; i < right; ++i) {
//    for (int i = 0; i < sample.length; ++i) {
      double entryWeight = 1.0;//rand.nextPoisson(200.0);
//      double entryWeight = rand.nextPoisson(1.0);
      sum += entryWeight * sample[i];
      weight += entryWeight;
    }
    if (weight == 0) {
      return 0;
    }
    return sum / weight;
  }
}
