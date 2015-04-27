package com.spbsu.ml.models;


import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.ml.BFGrid;

import java.util.List;

/**
 * Created with IntelliJ IDEA.
 * User: towelenee
 * Date: 30.11.13
 * Time: 18:03
 * Idea please stop making my code yellow
 */
public class ExponentialObliviousTree extends ObliviousTree {
  private final double DistCoef;
  private final List<BFGrid.BinaryFeature> features;
  private final double[] values;

  public ExponentialObliviousTree(
      final List<BFGrid.BinaryFeature> features,
      final double[] values,
      final double[] basedOn,
      final double DistCoef) {
    super(features, values, basedOn);
    this.DistCoef = DistCoef;
    this.features = features;
    this.values = values;
  }

  private static double getDistanceFromPointToRegion(int region, final Vec point, final List<BFGrid.BinaryFeature> features) {
    double sum2 = 0;
    for (int i = 0; i < features.size(); i++) {
      if (features.get(i).value(point) != ((region >> i) == 1)) {
        sum2 += Math.pow(point.get(features.get(i).findex) - features.get(i).condition, 2);
      }
    }
    return sum2;
  }

  public static double getWeightOfPointInRegion(int region, final Vec point, final List<BFGrid.BinaryFeature> features, double DistCoef) {
    final double distance = getDistanceFromPointToRegion(region, point, features);
    return Math.exp(-DistCoef * distance);
  }

  @Override
  public double value(final Vec point) {
    double sumWeights = 0;
    double sumTarget = 0;
    for (int region = 0; region < values.length; region++) {
      double weight = getWeightOfPointInRegion(region, point, features, DistCoef);
      sumWeights += weight;
      sumTarget += weight * values[region];
    }


    return sumTarget / sumWeights;
  }
}
