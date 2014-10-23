package com.spbsu.ml.methods.greedyRegion;

import com.spbsu.commons.func.AdditiveStatistics;
import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.commons.random.FastRandom;
import com.spbsu.commons.util.ArrayTools;
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
public class BFStochasticOptimizationSubset {
  private final BinarizedDataSet bds;
  public int[] points;
  public int[] failedCount;
  private final StatBasedLoss<AdditiveStatistics> oracle;
  private final Aggregate aggregate;
  private final FastRandom random = new FastRandom();
  private int depth;

  public BFStochasticOptimizationSubset(BinarizedDataSet bds, StatBasedLoss oracle, int[] points) {
    this.bds = bds;
    this.points = points;
    this.failedCount = new int[points.length];
    this.oracle = oracle;
    this.aggregate = new Aggregate(bds, oracle.statsFactory(), points);
  }


  public BFStochasticOptimizationSubset split(BFGrid.BinaryFeature feature, boolean mask) {
    TIntArrayList out = new TIntArrayList(points.length);
    TIntArrayList in = new TIntArrayList(points.length);
    TIntArrayList inFailed = new TIntArrayList(points.length);
    final byte[] bins = bds.bins(feature.findex);

    for (int i = 0; i < points.length; ++i) {
      final int index = points[i];
      if (bins[index] > feature.binNo != mask) {
        failedCount[i]++;
      }
      double diff = mask ? bins[index] - feature.binNo - 1 : feature.binNo - bins[index];
      if (random.nextDouble() <= Math.pow(0.5, -diff / 2) || (depth > 0 && failedCount[i] <= 1)) {
//      if (random.nextPoisson(1.0) >= failedCount[i] || (!first && failedCount[i] <= 1)) {
//      if (bins[index] > feature.binNo == mask) {
//      if (random.nextDouble() < Math.pow(0.5, -diff)) {// (mask ? 1 - feature.prob : feature.prob)) {//* Math.pow(0.5, -diff)) {// Math.pow(0.5, -diff)) {
        in.add(index);
        inFailed.add(failedCount[i]);
      } else {
        out.add(index);
      }
    }
    ++depth;
    if (in.size() == 0) {
      return null;
    }
    final BFStochasticOptimizationSubset outRegion = new BFStochasticOptimizationSubset(bds, oracle, out.toArray());
    aggregate.remove(outRegion.aggregate);
    points = in.toArray();
    failedCount = inFailed.toArray();
    return outRegion;
  }


  public int size() {
    return points.length;
  }

  public void visitAllSplits(Aggregate.SplitVisitor<? extends AdditiveStatistics> visitor) {
    aggregate.visit(visitor);
  }

  public <Stat extends AdditiveStatistics> void visitSplit(BFGrid.BinaryFeature bf, Aggregate.SplitVisitor<Stat> visitor) {
    final Stat left = (Stat) aggregate.combinatorForFeature(bf.bfIndex);
    final Stat right = (Stat) oracle.statsFactory().create().append(aggregate.total()).remove(left);
    visitor.accept(bf, left, right);
  }


  public void clear(Vec target) {
    if (points.length < 5)
      return;
    double[] sample = new double[points.length];
    for (int i = 0; i < points.length; ++i) {
      sample[i] = target.get(points[i]);
    }
    ArrayTools.parallelSort(sample, points);
    int left = (int) (sample.length * 0.01);
    int right = (int) (sample.length * 0.99);
    while (left > 0 && sample[left] == sample[left - 1]) --left;
    while (right < sample.length && sample[right] == sample[right - 1]) ++right;


    TIntArrayList out = new TIntArrayList(points.length);
    TIntArrayList in = new TIntArrayList(points.length);
    for (int i = 0; i < left; ++i) {
      out.add(points[i]);
    }
    for (int i = right; i < points.length; ++i) {
      out.add(points[i]);
    }

    for (int i = left; i < right; ++i) {
      final int index = points[i];
      in.add(index);
    }
    final BFStochasticOptimizationSubset outRegion = new BFStochasticOptimizationSubset(bds, oracle, out.toArray());
    aggregate.remove(outRegion.aggregate);
    points = in.toArray();
  }

  public double regularizedTotal(Vec target) {

    double[] sample = new double[points.length];
    for (int i = 0; i < points.length; ++i) {
      sample[i] = target.get(points[i]);
    }
    double sum = 0;
    double weight = 0;
    if (points.length < 4) {
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
//    int sum = 0;
    int left = (int) (sample.length * 0.01);
    int right = (int) (sample.length * 0.99);
    while (left > 0 && sample[left] == sample[left - 1]) --left;
    while (right < sample.length && sample[right] == sample[right - 1]) ++right;

    for (int i = left; i < right; ++i) {
//    for (int i = 0; i < sample.length; ++i) {
//      double entryWeight = 1.0;//rand.nextPoisson(200.0);
      double entryWeight = 1.0;//rand.nextPoisson(1.0);
      sum += entryWeight * sample[i];
      weight += entryWeight;
    }
    if (weight == 0) {
      return 0;
    }
    return sum / weight;
  }


  public AdditiveStatistics total() {
    return aggregate.total();
  }
}
