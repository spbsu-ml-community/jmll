package com.expleague.ml.dynamicGrid.trees;

import com.expleague.commons.func.AdditiveStatistics;
import com.expleague.ml.dynamicGrid.impl.BinarizedDynamicDataSet;
import com.expleague.ml.dynamicGrid.interfaces.BinaryFeature;
import com.expleague.ml.loss.AdditiveLoss;
import com.expleague.ml.dynamicGrid.AggregateDynamic;
import gnu.trove.list.array.TIntArrayList;


@SuppressWarnings("unchecked")
public class BFDynamicOptimizationSubset {
  private final BinarizedDynamicDataSet bds;
  public int[] points;
  private final AdditiveLoss<AdditiveStatistics> oracle;
  private final AggregateDynamic aggregate;


  public BFDynamicOptimizationSubset(final BinarizedDynamicDataSet bds, final AdditiveLoss oracle, final int[] points) {
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
    final T right = (T) oracle.statsFactory().apply(bf.fIndex()).append(aggregate.total(bf.fIndex())).remove(left);
    visitor.accept(bf, left, right);
  }

  public AdditiveStatistics total(int findex) {
    return aggregate.total(findex);
  }

  public void rebuild(final int... features) {
    this.aggregate.rebuild(features);
  }
}
