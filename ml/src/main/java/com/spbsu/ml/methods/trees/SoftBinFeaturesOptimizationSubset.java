package com.spbsu.ml.methods.trees;

import com.spbsu.commons.func.AdditiveStatistics;
import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.commons.random.FastRandom;
import com.spbsu.ml.BFGrid;
import com.spbsu.ml.data.Aggregate;
import com.spbsu.ml.data.SoftAggregate;
import com.spbsu.ml.data.impl.BinarizedDataSet;
import com.spbsu.ml.data.softBorders.dataSet.SoftDataSet;
import com.spbsu.ml.data.softBorders.dataSet.SoftGrid;
import com.spbsu.ml.loss.StatBasedLoss;
import gnu.trove.list.array.TIntArrayList;


/**
 * User: solar
 * Date: 10.09.13
 * Time: 12:16
 */
@SuppressWarnings("unchecked")
public class SoftBinFeaturesOptimizationSubset {
  private final SoftDataSet ds;
  private int[] points;
  private final StatBasedLoss<AdditiveStatistics> oracle;
  public final SoftAggregate aggregate;
  private FastRandom random = new FastRandom();

  final int randomBin(final Vec weights) {
    int bin = -1;
    double gain = random.nextDouble();
    while (gain > 0 && (bin < weights.length() - 1)) {
      gain -= weights.get(++bin);
    }
    return bin;
  }

  public SoftBinFeaturesOptimizationSubset(final SoftDataSet ds,
                                           final StatBasedLoss oracle,
                                           final int[] points) {
    this.ds = ds;
    this.points = points;
    this.oracle = oracle;
    this.aggregate = new SoftAggregate(ds, oracle.statsFactory(), points);
  }

  public SoftBinFeaturesOptimizationSubset split(final SoftGrid.SoftRow.BinFeature feature) {
    final TIntArrayList left = new TIntArrayList(points.length);
    final TIntArrayList right = new TIntArrayList(points.length);
    for (final int i : points) {
      final Vec binDistr  = ds.binDistribution(feature.row().featureIdx(), i);
      final int bin = randomBin(binDistr);
//      final byte bin = ds.bin(feature.row().featureIdx(), i);
      if (bin <= feature.binIdx) {
        left.add(i);
      } else {
        right.add(i);
      }
    }
    final SoftBinFeaturesOptimizationSubset rightBro = new SoftBinFeaturesOptimizationSubset(ds, oracle, right.toArray());
    aggregate.remove(rightBro.aggregate);
    points = left.toArray();
    return rightBro;
  }


  public int size() {
    return points.length;
  }

  public void visitAllSplits(final SoftAggregate.SplitVisitor<? extends AdditiveStatistics> visitor) {
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

  public int[] getPoints() {
    return points.clone();
  }
}
