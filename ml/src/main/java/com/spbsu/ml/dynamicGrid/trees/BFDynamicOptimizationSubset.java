package com.spbsu.ml.dynamicGrid.trees;

import com.spbsu.commons.func.AdditiveStatistics;
import com.spbsu.ml.dynamicGrid.AggregateDynamic;
import com.spbsu.ml.dynamicGrid.impl.BinarizedDynamicDataSet;
import com.spbsu.ml.dynamicGrid.interfaces.BinaryFeature;
import com.spbsu.ml.loss.StatBasedLoss;
import gnu.trove.list.array.TIntArrayList;


@SuppressWarnings("unchecked")
public class BFDynamicOptimizationSubset {
  private final BinarizedDynamicDataSet bds;
  public int[] points;
  private final StatBasedLoss<AdditiveStatistics> oracle;
  private final AggregateDynamic aggregate;


  public BFDynamicOptimizationSubset(final BinarizedDynamicDataSet bds, final StatBasedLoss oracle, final int[] points) {
    this.bds = bds;
    this.points = points;
    this.oracle = oracle;
    this.aggregate = new AggregateDynamic(bds, oracle.statsFactory(), points);
  }

  public BFDynamicOptimizationSubset split(final BinaryFeature feature) {
    final TIntArrayList left = new TIntArrayList(points.length);
    final TIntArrayList right = new TIntArrayList(points.length);
    final short[] bins = bds.bins(feature.fIndex());
    for (final int i : points) {
      if (bins[i] <= feature.binNo()) {
        left.add(i);
      } else {
        right.add(i);
      }
    }
    final BFDynamicOptimizationSubset rightBro = new BFDynamicOptimizationSubset(bds, oracle, right.toArray());
    aggregate.remove(rightBro.aggregate);
    points = left.toArray();
    aggregate.updatePoints(points);
    return rightBro;
  }


  public int size() {
    return points.length;
  }

  public void visitAllSplits(final AggregateDynamic.SplitVisitor<? extends AdditiveStatistics> visitor) {
    aggregate.visit(visitor);
  }

  public <T extends AdditiveStatistics> void visitSplit(final BinaryFeature bf, final AggregateDynamic.SplitVisitor<T> visitor) {
    final T left = (T) aggregate.combinatorForFeature(bf);
    final T right = (T) oracle.statsFactory().create().append(aggregate.total()).remove(left);
    visitor.accept(bf, left, right);
  }

  public AdditiveStatistics total() {
    return aggregate.total();
  }


  public void rebuild(final int... features) {
    this.aggregate.rebuild(features);

  }
}
