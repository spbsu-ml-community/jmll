package com.expleague.ml.methods.trees;

import com.expleague.commons.func.AdditiveStatistics;
import com.expleague.ml.BFGrid;
import com.expleague.ml.data.Aggregate;
import com.expleague.ml.data.impl.BinarizedDataSet;
import com.expleague.ml.loss.StatBasedLoss;
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
  public final Aggregate aggregate;

  public BFOptimizationSubset(final BinarizedDataSet bds, final StatBasedLoss oracle, final int[] points) {
    this.bds = bds;
    this.points = points;
    this.oracle = oracle;
    this.aggregate = new Aggregate(bds, oracle.statsFactory(), points);
  }

  public BFOptimizationSubset split(final BFGrid.Feature feature) {
    final TIntArrayList left = new TIntArrayList(points.length);
    final TIntArrayList right = new TIntArrayList(points.length);
    final byte[] bins = bds.bins(feature.findex());
    for (final int i : points) {
      if (feature.value(bins[i])) {
        right.add(i);
      } else {
        left.add(i);
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

  public void visitAllSplits(final Aggregate.SplitVisitor<? extends AdditiveStatistics> visitor) {
    aggregate.visit(visitor);
  }

  public <T extends AdditiveStatistics> void visitSplit(final BFGrid.Feature bf, final Aggregate.SplitVisitor<T> visitor) {
    final T left = (T) aggregate.combinatorForFeature(bf.index());
    final T right = (T) oracle.statsFactory().create().append(aggregate.total()).remove(left);
    visitor.accept(bf, left, right);
  }

  public AdditiveStatistics total() {
    return aggregate.total();
  }

  public int[] getPoints() {
    return points.clone();
  }
}
