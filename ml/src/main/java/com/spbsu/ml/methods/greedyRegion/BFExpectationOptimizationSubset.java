package com.spbsu.ml.methods.greedyRegion;

import com.spbsu.commons.func.AdditiveStatistics;
import com.spbsu.commons.random.FastRandom;
import com.spbsu.ml.BFGrid;
import com.spbsu.ml.data.impl.BinarizedDataSet;
import com.spbsu.ml.loss.StatBasedLoss;


/**
 * User: solar
 * Date: 10.09.13
 * Time: 12:16
 */
@SuppressWarnings("unchecked")
public class BFExpectationOptimizationSubset {
  private final BinarizedDataSet bds;
  private int[] points;
  private int[] failedCount;
  private double[] probs;
  private final StatBasedLoss<AdditiveStatistics> oracle;
  private final ExpectationAggregate aggregate;
  private final FastRandom random = new FastRandom();

  public BFExpectationOptimizationSubset(BinarizedDataSet bds, StatBasedLoss oracle, int[] points, double[] probs) {
    this.bds = bds;
    this.points = points;
    this.probs = probs;
    this.failedCount = new int[points.length];
    this.oracle = oracle;
    this.aggregate = new ExpectationAggregate(bds, oracle.statsFactory(), points, probs);
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

  public void split(BFGrid.BinaryFeature feature, boolean mask) {
    final byte[] bins = bds.bins(feature.findex);

    for (int i = 0; i < points.length; ++i) {
      final int index = points[i];
      if (bins[index] > feature.binNo != mask) {
//        failedCount[i]++;
        double diff = mask ? bins[index] - feature.binNo - 1 : feature.binNo - bins[index];
        probs[i] *= Math.pow(0.5, -diff);
        probs[i] *= mask ? 1 - feature.prob : feature.prob;
      }
//      double diff = mask ? bins[index] - feature.binNo - 1 : feature.binNo - bins[index];
//      probs[i] *= Math.pow(0.25,-diff);
    }
//    double totalWeight = 0;
//    for (int i = 0; i < probs.length; ++i)
//      totalWeight += probs[i];
//    for (int i = 0; i < probs.length; ++i) {
//      probs[i] /= totalWeight;
//    }
    aggregate.rebuild();
  }


  public int size() {
    return points.length;
  }

  public void visitAllSplits(ExpectationAggregate.SplitVisitor<? extends AdditiveStatistics> visitor) {
    aggregate.visit(visitor);
  }

  public <Stat extends AdditiveStatistics> void visitSplit(BFGrid.BinaryFeature bf, ExpectationAggregate.SplitVisitor<Stat> visitor) {
    final Stat left = (Stat) aggregate.combinatorForFeature(bf.bfIndex);
    final Stat right = (Stat) oracle.statsFactory().create().append(aggregate.total()).remove(left);
    visitor.accept(bf, left, right);
  }


  public AdditiveStatistics total() {
    return aggregate.total();
  }
}
