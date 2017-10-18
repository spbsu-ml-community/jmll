package com.expleague.ml.methods.greedyRegion;

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
public class BFOptimizationRegion {
  protected final BinarizedDataSet bds;
  protected int[] pointsInside;
  protected final StatBasedLoss<AdditiveStatistics> oracle;
  protected Aggregate aggregate;

  public BFOptimizationRegion(final BinarizedDataSet bds,
                              final StatBasedLoss oracle,
                              final int[] points) {
    this.bds = bds;
    this.pointsInside = points;
    this.oracle = oracle;
    this.aggregate = new Aggregate(bds, oracle.statsFactory(), points);
  }


  public void split(final BFGrid.BinaryFeature feature, final boolean mask) {
    final byte[] bins = bds.bins(feature.findex);

    final TIntArrayList newInside = new TIntArrayList();
    final TIntArrayList newOutside = new TIntArrayList();


    for (int index : pointsInside) {
      if ((bins[index] > feature.binNo) != mask) {
        newOutside.add(index);
      } else {
        newInside.add(index);
      }
    }
    pointsInside = newInside.toArray();
    if (newInside.size() < newOutside.size()) {
      aggregate = new Aggregate(bds, oracle.statsFactory(), pointsInside);
    } else {
      aggregate.remove(new BFOptimizationRegion(bds, oracle, newOutside.toArray()).aggregate);
    }
  }

  int size() {
    return pointsInside.length;
  }

  public void visitAllSplits(final Aggregate.SplitVisitor<? extends AdditiveStatistics> visitor) {
    aggregate.visit(visitor);
  }

  public <T extends AdditiveStatistics> void visitSplit(final BFGrid.BinaryFeature bf, final Aggregate.SplitVisitor<T> visitor) {
    final T left = (T) aggregate.combinatorForFeature(bf.bfIndex);
    final T right = (T) oracle.statsFactory().create().append(aggregate.total()).remove(left);
    visitor.accept(bf, left, right);
  }

  public AdditiveStatistics total() {
    return aggregate.total();
  }
}
