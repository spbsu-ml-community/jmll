package com.expleague.ml.methods.greedyRegion;

import com.expleague.commons.func.AdditiveStatistics;
import com.expleague.commons.util.ArrayTools;
import com.expleague.ml.BFGrid;
import com.expleague.ml.data.Aggregate;
import com.expleague.ml.data.impl.BinarizedDataSet;
import com.expleague.ml.loss.AdditiveLoss;
import com.expleague.ml.methods.trees.BFOptimizationSubset;
import com.expleague.ml.models.Region;
import gnu.trove.list.array.TIntArrayList;

import java.util.Arrays;


/**
 * User: solar
 * Date: 10.09.13
 * Time: 12:16
 */
@SuppressWarnings({"unchecked", "unused"})

public class BFWeakConditionsOptimizationRegion extends Region {
  protected final BinarizedDataSet bds;
  protected int[] points;
  protected int[] failedCount;
  protected final AdditiveLoss<AdditiveStatistics> oracle;
  protected final Aggregate aggregate;
  protected final int maxFailed;
  public AdditiveStatistics nonCriticalTotal;
  public final AdditiveStatistics excluded;
  protected final int[] failedBorders;

  public BFWeakConditionsOptimizationRegion(final BinarizedDataSet bds, final AdditiveLoss oracle, final int[] points, final BFGrid.Feature[] features, final boolean[] masks, final int maxFailed) {
    super(Arrays.asList(features), masks, Double.NaN, Double.NaN,0, Double.NaN, maxFailed);
    this.bds = bds;
    this.excluded = (AdditiveStatistics) oracle.statsFactory().apply(0);
    this.points = points;
    this.failedCount = new int[points.length];
    final byte[][] bins = new byte[features.length][];
    for (int f = 0; f < features.length; ++f)
      bins[f] = bds.bins(features[f].findex());

    this.nonCriticalTotal = (AdditiveStatistics) oracle.statsFactory().apply(0);
    final TIntArrayList maxFailedPoints = new TIntArrayList();
    for (int i = 0; i < points.length; ++i) {
      final int index = points[i];
      int failed = 0;
      for (int f = 0; f < features.length; ++f) {
        if (bins[f][index] > features[f].bin() != masks[f]) {
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
    this.aggregate = new Aggregate(bds, oracle.statsFactory());
    aggregate.append(maxFailedPoints.toArray());
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

  public BFOptimizationSubset split(final BFGrid.Feature feature, final boolean mask) {
    final TIntArrayList out = new TIntArrayList(points.length);
    final byte[] bins = bds.bins(feature.findex());
    final TIntArrayList newCriticalPoints = new TIntArrayList();
    final AdditiveStatistics newCritical = oracle.statsFactory().apply(feature.findex());
    final AdditiveStatistics test = oracle.statsFactory().apply(feature.findex());

    for (int i = 0; i < failedBorders[maxFailed]; ++i) {
      final int index = points[i];
      if ((bins[index] > feature.bin()) != mask) {
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

  public <T extends AdditiveStatistics> void visitSplit(final BFGrid.Feature bf, final Aggregate.SplitVisitor<T> visitor) {
    final T left = (T) aggregate.combinatorForFeature(bf.index());
    final T right = (T) oracle.statsFactory().apply(bf.findex()).append(aggregate.total(-1)).remove(left);
    visitor.accept(bf, left, right);
  }

  public AdditiveStatistics total() {
    return aggregate.total(0).append(nonCriticalTotal);
  }
}
