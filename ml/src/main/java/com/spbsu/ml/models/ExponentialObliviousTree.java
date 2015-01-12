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
public class ExponentialObliviousTree extends ContinousObliviousTree {
  private final double DistCoef;

  public ExponentialObliviousTree(final List<BFGrid.BinaryFeature> features, final double[][] values, final double _distCoef) {
    super(features, values);
    DistCoef = _distCoef;
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
  public double value(final Vec _x) {
    double sum = 0;

    final double[] x = new double[features.length + 1];
    for (int i = 0; i < features.length; i++)
      x[i + 1] = _x.get(features[i].findex);
    x[0] = 1;
    final double sumWeights = 0;
    //for (int index = 0; index < 1 << lines.length; index++) {
    //double weight = Math.exp(-calcDistanseToRegion(index, _x));
    //sumWeights += weight;
    int index = 0;
    for (int j = 0; j < features.length; j++) {
      index <<= 1;
      if (features[j].value(_x))
        index++;
    }
    for(int i = 0; i < x.length;i++)
      for(int j = 0; j <= i; j++)
        sum += values[index][i * (i + 1) / 2 + j] * x[i] * x[j];


    return sum ;// / sumWeights;
  }
}
