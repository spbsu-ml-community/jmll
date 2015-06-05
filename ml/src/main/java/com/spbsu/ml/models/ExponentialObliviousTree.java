package com.spbsu.ml.models;


import com.spbsu.commons.math.MathTools;
import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.ml.BFGrid;
import com.spbsu.ml.methods.trees.GreedyExponentialObliviousTree;

import java.util.List;

/**
 * Created with IntelliJ IDEA.
 * User: towelenee
 * Date: 30.11.13
 * Time: 18:03
 * Idea please stop making my code yellow
 */
public class ExponentialObliviousTree extends ObliviousTree {
  private final GreedyExponentialObliviousTree parent;
  private final List<BFGrid.BinaryFeature> features;

  public ExponentialObliviousTree(
      final List<BFGrid.BinaryFeature> features,
      final double[] values,
      final double[] basedOn,
      final GreedyExponentialObliviousTree parent) {
    super(features, values, basedOn);
    this.parent = parent;
    this.features = features;
  }

  @Override
  public double value(final Vec point) {
    double sumTarget = 0;
    double sumWeights = 0;
    double[] weights = parent.getProbabilitiesBeingInRegion(features, point);
    for (int region = 0; region < values.length; region++) {
      if (weights[region] > MathTools.EPSILON) {
        sumTarget += weights[region] * values[region];
        sumWeights += weights[region];
      }
    }
    if (sumWeights < 0.999 || sumWeights > 1.01)
      throw new RuntimeException("");
    return sumTarget / sumWeights; //another dirty hack
  }
}
