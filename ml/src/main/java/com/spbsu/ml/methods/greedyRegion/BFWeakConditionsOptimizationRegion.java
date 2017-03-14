package com.spbsu.ml.methods.greedyRegion;

import com.spbsu.commons.func.AdditiveStatistics;
import com.spbsu.commons.util.ArrayTools;
import com.spbsu.ml.BFGrid;
import com.spbsu.ml.data.Aggregate;
import com.spbsu.ml.data.impl.BinarizedDataSet;
import com.spbsu.ml.loss.StatBasedLoss;
import com.spbsu.ml.methods.trees.BFOptimizationSubset;
import gnu.trove.list.array.TIntArrayList;


/**
 * User: solar
 * Date: 10.09.13
 * Time: 12:16
 */
@SuppressWarnings("unchecked")

public class BFWeakConditionsOptimizationRegion {
  protected final BinarizedDataSet bds;
  protected int[] points;
  protected int[] failedCount;
  protected final StatBasedLoss<AdditiveStatistics> oracle;
  protected final Aggregate aggregate;
  protected final int maxFailed;
  public AdditiveStatistics nonCriticalTotal;
  public final AdditiveStatistics excluded;
  protected final int[] failedBorders;

  public BFWeakConditionsOptimizationRegion(final BinarizedDataSet bds, final StatBasedLoss oracle, final int[] points, final BFGrid.BinaryFeature[] features, final boolean[] masks, final int maxFailed) {
    this.bds = bds;
    this.excluded = (AdditiveStatistics) oracle.statsFactory().create();
    this.points = points;
    this.failedCount = new int[points.length];
    final byte[][] bins = new byte[features.length][];
    for (int f = 0; f < features.length; ++f)
      bins[f] = bds.bins(features[f].findex);

    this.nonCriticalTotal = (AdditiveStatistics) oracle.statsFactory().create();
    final TIntArrayList maxFailedPoints = new TIntArrayList();
    for (int i = 0; i < points.length; ++i) {
      final int index = points[i];
      int failed = 0;
      for (int f = 0; f < features.length; ++f) {
        if (bins[f][index] > features[f].binNo != masks[f]) {
          ++failed;
        }
      }
      failedCount[i] = failed;
      if (failed < maxFailed) {
        this.nonCriticalTotal.append(index, 1);
      } else if (failed == maxFailed) {
        maxFailedPoints.add(index);
      } else {
        excluded.append(index, 1);
      }
    }

    this.oracle = oracle;
    this.maxFailed = maxFailed;
    this.aggregate = new Aggregate(bds, oracle.statsFactory(), maxFailedPoints.toArray());
    this.failedBorders = new int[maxFailed + 1];
    ArrayTools.parallelSort(failedCount, points);
    failedBorders[maxFailed] = points.length;
    updateFailedBorders(failedCount, failedBorders);
  }

  protected void updateFailedBorders(final int[] failedCount, final int[] failedBorders) {
    final int rightLimit = failedBorders[maxFailed];
    failedBorders[maxFailed] = upperBound(failedCount, maxFailed, 0, rightLimit);
    for (int i = maxFailed - 1; i >= 0; --i) {
      failedBorders[i] = upperBound(failedCount, i, 0, failedBorders[i + 1]);
    }
  }

  //java version doesn't guarantee, that we'll find last or first entry
  private int upperBound(final int[] arr, final int key, final int fromIndex, final int toIndex) {
    int left = fromIndex;
    int right = toIndex;
    while (right - left > 1) {
      final int mid = (left + right) >>> 1;
      final int midVal = arr[mid];
      if (midVal <= key)
        left = mid;
      else
        right = mid;
    }
    if (right > 0 && arr[right - 1] < key) {
      return right - 1;
    } else
      return right;
  }


  public BFOptimizationSubset split(final BFGrid.BinaryFeature feature, final boolean mask) {
    final TIntArrayList out = new TIntArrayList(points.length);
    final byte[] bins = bds.bins(feature.findex);
    final TIntArrayList newCriticalPoints = new TIntArrayList();
    final AdditiveStatistics newCritical = oracle.statsFactory().create();
    final AdditiveStatistics test = oracle.statsFactory().create();

    for (int i = 0; i < failedBorders[maxFailed]; ++i) {
      final int index = points[i];
      if ((bins[index] > feature.binNo) != mask) {
        failedCount[i]++;
        if (failedCount[i] == maxFailed) {
          newCriticalPoints.add(index);
          newCritical.append(index, 1);
        } else if (failedCount[i] == (maxFailed + 1)) {
          out.add(index);
          excluded.append(index, 1);
          test.append(index, 1);
        }
      }
    }
    final BFOptimizationSubset outRegion = new BFOptimizationSubset(bds, oracle, out.toArray());
    aggregate.remove(outRegion.aggregate);
    aggregate.append(newCriticalPoints.toArray());
    nonCriticalTotal.remove(newCritical);
    ArrayTools.parallelSort(failedCount, points, 0, failedBorders[maxFailed] - 1);
    updateFailedBorders(failedCount, failedBorders);
    return outRegion;
  }


  public int size() {
    return failedBorders[maxFailed];
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
    return aggregate.total().append(nonCriticalTotal);
  }
}
