package com.expleague.ml.methods.trees;

import com.expleague.ml.data.Aggregate;
import com.expleague.ml.data.impl.BinarizedDataSet;
import com.expleague.ml.loss.StatBasedLoss;
import com.expleague.commons.func.AdditiveStatistics;
import com.expleague.ml.BFGrid;
import gnu.trove.list.array.TIntArrayList;


/**
 * User: solar
 * Date: 10.09.13
 * Time: 12:16
 */
@SuppressWarnings("unchecked")
public class BFOptimizationSubset {
  private final BinarizedDataSet bds;
  private final StatBasedLoss<AdditiveStatistics> oracle;
  private Subset subset;


  private BFOptimizationSubset(final BinarizedDataSet bds,
                              final StatBasedLoss oracle,
                              final Subset subset) {
    this.bds = bds;
    this.oracle = oracle;
    this.subset = subset;

  }

  public BFOptimizationSubset(final BinarizedDataSet bds,
                              final StatBasedLoss oracle,
                              final int[] points) {
    this.bds = bds;
    this.oracle = oracle;
    this.subset =  new Subset(points);
  }

  public BFOptimizationSubset split(final BFGrid.BinaryFeature feature) {
    final TIntArrayList left = new TIntArrayList(subset.points.length);
    final TIntArrayList right = new TIntArrayList(subset.points.length);
    final byte[] bins = bds.bins(feature.findex);
    for (final int i : subset.points) {
      if (feature.value(bins[i])) {
        right.add(i);
      } else {
        left.add(i);
      }
    }
    final BFOptimizationSubset rightBro;

    if (left.size() < right.size()) {
      final Subset leftSubset = new Subset(left.toArray());
      final Aggregate rightAggregate = subset.aggregate;
      rightAggregate.remove(leftSubset.aggregate);
      final Subset rightSubset = new Subset(right.toArray(), rightAggregate);
      subset = leftSubset;
      rightBro = new BFOptimizationSubset(bds, oracle, rightSubset);
    } else {
      rightBro = new BFOptimizationSubset(bds, oracle, right.toArray());
      final Aggregate leftAggregate = subset.aggregate;
      leftAggregate.remove(rightBro.subset.aggregate);
      subset = new Subset(left.toArray(), leftAggregate);
    }
    return rightBro;
  }

  public Aggregate aggregate() {
    return subset.aggregate;
  }

  public int size() {
    return subset.points.length;
  }

  public void visitAllSplits(final Aggregate.SplitVisitor<? extends AdditiveStatistics> visitor) {
    subset.aggregate.visit(visitor);
  }

  public <T extends AdditiveStatistics> void visitSplit(final BFGrid.BinaryFeature bf, final Aggregate.SplitVisitor<T> visitor) {
    final T left = (T) subset.aggregate.combinatorForFeature(bf.bfIndex);
    final T right = (T) oracle.statsFactory().create().append(subset.aggregate.total()).remove(left);
    visitor.accept(bf, left, right);
  }

  public AdditiveStatistics total() {
    return subset.aggregate.total();
  }

  public int[] getPoints() {
    return subset.points.clone();
  }

  class Subset {
    public final int[] points;
    public final Aggregate aggregate;

    Subset(final int[] points) {
      this.points = points;
      this.aggregate = new Aggregate(bds, oracle.statsFactory(), points);
    }

    Subset(final int[] points, final Aggregate aggregate) {
      this.points = points;
      this.aggregate = aggregate;
    }
  }
}
