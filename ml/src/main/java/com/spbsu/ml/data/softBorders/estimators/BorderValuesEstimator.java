package com.spbsu.ml.data.softBorders.estimators;

import com.spbsu.ml.data.softBorders.Estimator;
import com.spbsu.ml.data.softBorders.dataSet.WeightedFeature;

/**
 * Created by noxoomo on 06/11/2016.
 */
public class BorderValuesEstimator implements Estimator<int[]> {
  private final WeightedFeature feature;
  private final int[] borderIndices;
  private final double[] values;
  private int count;

  public BorderValuesEstimator(final WeightedFeature feature,
                               final int[] borderIndices) {
    this.feature = feature;
    this.borderIndices = borderIndices;
    this.values = new double[borderIndices.length];
  }

  @Override
  public BorderValuesEstimator add(final int[] permutation) {
    for (int i = 0; i < borderIndices.length; ++i) {
      final int borderIdx = borderIndices[i];
      final int rk = permutation[borderIdx];
      values[i] += feature.value(rk);
    }
    ++count;
    return this;
  }

  public double[] borders() {
    double[] result = new double[values.length];
    for (int i = 0; i < result.length; ++i) {
      result[i] = values[i] / count;
    }
    return result;
  }
}


