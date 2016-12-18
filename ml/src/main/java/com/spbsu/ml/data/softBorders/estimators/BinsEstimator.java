package com.spbsu.ml.data.softBorders.estimators;

import com.spbsu.ml.data.softBorders.Estimator;
import com.spbsu.ml.data.softBorders.dataSet.WeightedFeature;

/**
 * Created by noxoomo on 06/11/2016.
 */
public class BinsEstimator implements Estimator<int[]> {
  final int[] borders;
  final WeightedFeature feature;
  final double[] bins;
  final int binsCount;
  int count = 0;

  public BinsEstimator(final WeightedFeature feature,
                       final int[] borders) {
    this.borders = borders;
    this.feature = feature;
    this.binsCount = borders.length + 1;
    this.bins = new double[binsCount *  feature.size()];
  }

  public double[] softBins() {
    final double[] result = new double[bins.length];
    System.arraycopy(bins, 0, result, 0, result.length);
    for (int i = 0; i < result.length; ++i) {
      result[i] /= count;
    }
    return result;
  }

  @Override
  public BinsEstimator add(final int[] permutation) {
    int bin = 0;
    for (int i = 0; i < permutation.length; ++i) {
      final int rk = permutation[i];
      bins[rk * binsCount + bin]++;
      if (bin < borders.length && i == borders[bin]) {
        ++bin;
      }
    }
    ++count;
    return this;
  }
}
