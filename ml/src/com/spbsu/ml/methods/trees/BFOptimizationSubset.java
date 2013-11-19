package com.spbsu.ml.methods.trees;

import com.spbsu.commons.func.AdditiveGator;
import com.spbsu.ml.BFGrid;
import com.spbsu.ml.data.Aggregate;
import com.spbsu.ml.data.impl.BinarizedDataSet;
import com.spbsu.ml.loss.StatBasedOracle;
import gnu.trove.TIntArrayList;

/**
 * User: solar
 * Date: 10.09.13
 * Time: 12:16
 */
@SuppressWarnings("unchecked")
public class BFOptimizationSubset {
  private final BinarizedDataSet bds;
  private int[] points;
  private final StatBasedOracle<AdditiveGator> oracle;
  private final Aggregate aggregate;

  public BFOptimizationSubset(BinarizedDataSet bds, StatBasedOracle oracle, int[] points) {
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
    final BFOptimizationSubset rightBro = new BFOptimizationSubset(bds, oracle, right.toNativeArray());
    aggregate.remove(rightBro.aggregate);
    points = left.toNativeArray();
    return rightBro;
  }

  public int size() {
    return points.length;
  }

  public void visitAllSplits(Aggregate.SplitVisitor<? extends AdditiveGator> visitor) {
    aggregate.visit(visitor);
  }

  public <T extends AdditiveGator> void visitSplit(BFGrid.BinaryFeature bf, Aggregate.SplitVisitor<T> visitor) {
    final T left = (T)aggregate.combinatorForFeature(bf.bfIndex);
    final T right = (T)oracle.statsFactory().create().append(aggregate.total()).remove(left);
    visitor.accept(bf, left, right);
  }

  public AdditiveGator total() {
    return aggregate.total();
  }
}
