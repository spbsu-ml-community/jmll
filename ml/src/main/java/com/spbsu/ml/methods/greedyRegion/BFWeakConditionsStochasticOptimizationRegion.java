package com.spbsu.ml.methods.greedyRegion;

import com.spbsu.commons.func.AdditiveStatistics;
import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.commons.random.FastRandom;
import com.spbsu.commons.util.ArrayTools;
import com.spbsu.ml.BFGrid;
import com.spbsu.ml.data.impl.BinarizedDataSet;
import com.spbsu.ml.data.set.VecDataSet;
import com.spbsu.ml.loss.StatBasedLoss;
import com.spbsu.ml.methods.trees.BFOptimizationSubset;
import gnu.trove.list.array.TIntArrayList;


/**
 * User: solar
 * Date: 10.09.13
 * Time: 12:16
 */
@SuppressWarnings("unchecked")
public class BFWeakConditionsStochasticOptimizationRegion extends BFWeakConditionsOptimizationRegion {
  private final FastRandom random = new FastRandom();
  double alpha = 0.02;
  double beta = 0.5;

  public BFWeakConditionsStochasticOptimizationRegion(BinarizedDataSet bds, StatBasedLoss oracle, int[] points, BFGrid.BinaryFeature[] features, boolean[] masks, int maxFailed) {
    super(bds, oracle, points, features, masks, maxFailed);
  }

  public BFOptimizationSubset split(BFGrid.BinaryFeature feature, boolean mask) {
    TIntArrayList out = new TIntArrayList(points.length);
    final byte[] bins = bds.bins(feature.findex);
    TIntArrayList newCriticalPoints = new TIntArrayList();
    AdditiveStatistics newCritical = oracle.statsFactory().create();

    int nonCriticalEnd = maxFailed > 0 ? failedBorders[maxFailed - 1] : 0;
    for (int i = 0; i < nonCriticalEnd; ++i) {
      final int index = points[i];
      if ((bins[index] > feature.binNo) != mask) {
        failedCount[i]++;
        if (failedCount[i] == maxFailed) {
          newCriticalPoints.add(index);
          newCritical.append(index, 1);
        }
      }
    }
    boolean[] failed = splitCritical(points, nonCriticalEnd, failedBorders[maxFailed], feature, mask, bins);
    for (int i = nonCriticalEnd; i < failedBorders[maxFailed]; ++i) {
      final int index = points[i];
      if (failed[i - nonCriticalEnd]) {
        excluded.append(index, 1);
        failedCount[i]++;
        out.add(index);
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

  private boolean[] splitCritical(int[] points, int left, int right, BFGrid.BinaryFeature feature, boolean mask, byte[] bins) {
    boolean[] result = new boolean[right - left];
//    for (int i = left; i < right;++i) {
//      final int index = points[i];
//      final double diff = mask ? bins[index] - feature.binNo - 1 : feature.binNo - bins[index];
//      result[i-left] = random.nextDouble() >= Math.pow(0.5, -diff / 1.3);
//    }
//    return result;
    double[] values = new double[right - left];
    Vec featureValues = ((VecDataSet) bds.original()).data().col(feature.findex);
    int[] order = ArrayTools.sequence(0, values.length);
    for (int i = 0; i < values.length; ++i) {
      final int index = points[i + left];
      values[i] = featureValues.get(index);
    }
    ArrayTools.parallelSort(values, order);
    double[] ranks = rank(values);
    int split = upperBound(values, feature.condition);
    for (int i = 0; i < values.length; ++i) {
      if (values[i] > feature.condition != mask) {
        //if !mask, than diff = #points <= point - #points in left
        //if mask, than diff = #points in left - #points < point
        //points in left = split
        double diff = mask ? split - ranks[i] + 1 : ranks[i] - split;
        result[order[i]] = random.nextDouble() >= Math.pow(0.5, alpha * diff);
      } else {
        //if mask, than diff =  #points <= point - #points in left
        //if !mask, than diff = #points in left - #points < point
        //points in left = split
        double diff = mask ? ranks[i] - split : split - ranks[i] + 1;
        result[order[i]] = random.nextDouble() <= Math.pow(0.5, beta * diff);
      }
    }
    return result;
  }

  private double[] rank(double[] sortedSample) {
    double[] ranks = new double[sortedSample.length];
    for (int i = 0; i < sortedSample.length; ++i) {
      int j = i + 1;
      while (j < sortedSample.length && Math.abs(sortedSample[j] - sortedSample[j - 1]) < 1e-9) ++j;
      final double rk = i + 0.5 * (j - i + 1);
      for (; i < j; ++i) {
        ranks[i] = rk;
      }
      --i;
    }
    return ranks;
  }


  //java version doesn't guarantee, that we'll find last entry
  //should return first index, that greater than key
  private int upperBound(double[] arr, double key) {
    int left = 0;
    int right = arr.length;
    while (right - left > 1) {
      final int mid = (left + right) >>> 1;
      final double midVal = arr[mid];
      if (midVal <= key)
        left = mid;
      else
        right = mid;
    }
    return right;
  }

  //should return last index, that less than key +1
  private int lowerBound(double[] arr, double key) {
    int left = 0;
    int right = arr.length;
    while ((right - left) > 1) {
      final int mid = (left + right) >>> 1;
      final double midVal = arr[mid];
      if (midVal < key)
        left = mid;
      else
        right = mid;
    }
    return right;
  }

}
