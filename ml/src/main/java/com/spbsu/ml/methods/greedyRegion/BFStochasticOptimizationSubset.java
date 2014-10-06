package com.spbsu.ml.methods.greedyRegion;

import com.spbsu.commons.func.AdditiveStatistics;
import com.spbsu.commons.random.FastRandom;
import com.spbsu.ml.BFGrid;
import com.spbsu.ml.data.Aggregate;
import com.spbsu.ml.data.impl.BinarizedDataSet;
import com.spbsu.ml.loss.StatBasedLoss;
import gnu.trove.list.array.TIntArrayList;


/**
 * User: solar
 * Date: 10.09.13
 * Time: 12:16
 */
@SuppressWarnings("unchecked")
public class BFStochasticOptimizationSubset {
  private final BinarizedDataSet bds;
  public int[] points;
  private final StatBasedLoss<AdditiveStatistics> oracle;
  private final Aggregate aggregate;
  private final FastRandom random = new FastRandom();

  public BFStochasticOptimizationSubset(BinarizedDataSet bds, StatBasedLoss oracle, int[] points) {
    this.bds = bds;
    this.points = points;
    this.oracle = oracle;
    this.aggregate = new Aggregate(bds, oracle.statsFactory(), points);
  }


  public BFStochasticOptimizationSubset split(BFGrid.BinaryFeature feature, boolean mask) {
    TIntArrayList out = new TIntArrayList(points.length);
    TIntArrayList in = new TIntArrayList(points.length);
    final byte[] bins = bds.bins(feature.findex);

    for (int i = 0; i < points.length; ++i) {
      final int index = points[i];
      double diff = mask ? bins[index] - feature.binNo - 1 : feature.binNo - bins[index];
      if (random.nextDouble() < Math.pow(0.5, -diff)) {
        in.add(index);
      } else {
        out.add(index);
      }
    }
    if (in.size() == 0 || in.size() == points.length) {
      return null;
    }
    final BFStochasticOptimizationSubset inRegion = new BFStochasticOptimizationSubset(bds, oracle, in.toArray());
    aggregate.remove(inRegion.aggregate);
    points = out.toArray();
    return inRegion;
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


  public AdditiveStatistics total() {
    return aggregate.total();
  }
}
