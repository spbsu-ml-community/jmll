package com.spbsu.ml.models;


import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.ml.BFGrid;
import com.spbsu.ml.methods.trees.GreedyObliviousLinearTree;

import java.util.List;

/**
 * Created with IntelliJ IDEA.
 * User: towelenee
 * Date: 30.11.13
 * Time: 18:03
 * Idea please stop making my code yellow
 */
public class ExponentialObliviousTree extends PolynomialObliviousTree {
  private final double DistCoef;

  public ExponentialObliviousTree(final BFGrid.BinaryFeature[] features, final double[][] values, final double DistCoef) {
    super(features, values);
    this.DistCoef = DistCoef;
  }

  double sqr(final double x) {
    return x * x;
  }

  double calcDistanseToRegion(final int index, final Vec point) {
    double ans = 0;
    for (int i = 0; i < features.length; i++) {
      if (features[i].value(point) != ((index >> i) == 1)) {
        ans += sqr(point.get(features[i].findex) - features[i].condition);//L2
      }
    }
    return DistCoef * ans;
  }

  @Override
  public double value(final Vec point) {
    double sum = 0;

    final double[] factors = GreedyObliviousLinearTree.getSignificantFactors(point, features);
    final double sumWeights = 0;
    //for (int index = 0; index < 1 << lines.length; index++) {
    //double weight = Math.exp(-calcDistanseToRegion(index, _x));
    //sumWeights += weight;
    int index = 0;

    for(int i = 0; i < factors.length;i++)
      for(int j = 0; j <= i; j++)
        sum += values[index][i * (i + 1) / 2 + j] * factors[i] * factors[j];


    return sum ;// / sumWeights;
  }
}
