package com.spbsu.ml.methods.trees;

import com.spbsu.commons.func.AdditiveStatistics;
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

  public int size() {
    return points.length;
  }

  public void visitAllSplits(Aggregate.SplitVisitor<? extends AdditiveStatistics> visitor) {
    aggregate.visit(visitor);
  }

  public <T extends AdditiveStatistics> void visitSplit(BFGrid.BinaryFeature bf, Aggregate.SplitVisitor<T> visitor) {
    final T left = (T)aggregate.combinatorForFeature(bf.bfIndex);
    final T right = (T)oracle.statsFactory().create().append(aggregate.total()).remove(left);
    visitor.accept(bf, left, right);
  }

  public AdditiveStatistics total() {
    return aggregate.total();
  }
}
