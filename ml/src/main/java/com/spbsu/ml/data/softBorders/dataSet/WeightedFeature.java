package com.spbsu.ml.data.softBorders.dataSet;

import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.commons.math.vectors.impl.vectors.ArrayVec;
import com.spbsu.commons.util.ArrayTools;
import com.spbsu.ml.data.set.VecDataSet;
import gnu.trove.list.array.TDoubleArrayList;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * Created by noxoomo on 06/11/2016.
 */
public class WeightedFeature {
  private double[] values;
  private double[] weight;

  public WeightedFeature(final double[] featureValues,
                         final double[] featureWeights) {
    this.values = featureValues;
    this.weight = featureWeights;
  }

  public double value(int rank) {
    return values[rank];
  }

  public double weight(int rank) {
    return weight[rank];
  }

  public double[] weights() {
    return weight;
  }

  public int size() {
    return values.length;
  }

  public static WeightedFeature build(final Vec sortedValues) {
    TDoubleArrayList vals = new TDoubleArrayList(sortedValues.dim());
    TDoubleArrayList weights = new TDoubleArrayList(sortedValues.dim());

    double val = sortedValues.at(0);
    double weight = 0;
    for (int i = 0; i < sortedValues.dim(); ++i) {
      weight++;
      final double nextVal = (i + 1) == sortedValues.dim()
              ? Double.POSITIVE_INFINITY
              : sortedValues.at(i + 1);
      if (val != nextVal) {
        vals.add(val);
        weights.add(weight);
        weight = 0;
        val = nextVal;
      }
    }
    return new WeightedFeature(vals.toArray(), weights.toArray());
  }

  public static List<WeightedFeature> build(final VecDataSet dataSet) {
    List<WeightedFeature> weightedFeatures = new ArrayList<>();

    for (int f = 0; f < dataSet.xdim(); ++f) {
      final int[] order = dataSet.order(f);
      final Vec feature = new ArrayVec(dataSet.length());
      for (int i = 0; i < feature.dim(); i++) {
        feature.set(i, dataSet.at(order[i]).get(f));
      }
      weightedFeatures.add(WeightedFeature.build(feature));
    }
    return weightedFeatures;
  }

  public int rankByValue(final double value) {
    final int idx = Arrays.binarySearch(values, value);
    if (idx < 0) {
      throw new RuntimeException("Error: unknown value");
    }
    return idx;
  }
}

