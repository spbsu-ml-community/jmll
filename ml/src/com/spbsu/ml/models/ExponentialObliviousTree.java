package com.spbsu.ml.models;


import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.ml.BFGrid;
import com.spbsu.ml.Func;
import com.spbsu.ml.methods.trees.GreedyExponentialObliviousTree;

import java.util.List;

/**
 * Created with IntelliJ IDEA.
 * User: towelenee
 * Date: 30.11.13
 * Time: 18:03
 * Idea please stop making my code yellow
 */
public class ExponentialObliviousTree extends Func.Stub {
  private List<BFGrid.BinaryFeature> features;
  private double[][] values;
  private final double DistCoef;
  private BFGrid grid;

  public ExponentialObliviousTree(final List<BFGrid.BinaryFeature> features, double[][] values, double distcoef, BFGrid grid) {
    this.features = features;
    this.values = values;
    DistCoef = distcoef;
    this.grid = grid;
  }

  @Override
  public double value(Vec _x) {
    double sum = 0;

    double x[] = new double[features.size() + 1];
    for (int i = 0; i < features.size(); i++)
      x[i + 1] = _x.get(features.get(i).findex);
    x[0] = 1;
    double sumWeights = 0;
    for (int index = 0; index < 1 << features.size(); index++) {
      double weight = GreedyExponentialObliviousTree.calcDistanseToRegion(grid, DistCoef, index, _x, features);
      if (weight > 1e-9)
        for (int i = 0; i < x.length; i++)
          for (int j = 0; j <= i; j++)
            sum += values[index][i * (i + 1) / 2 + j] * x[i] * x[j] * weight;
      sumWeights += weight;
    }
    if (sumWeights == 0) {
      int index = 0;
      for (int j = 0; j < features.size(); j++) {
        if (features.get(j).value(_x))
          index += 1 << j;
      }

      System.out.println(GreedyExponentialObliviousTree.calcDistanseToRegion(grid, DistCoef, index, _x, features));
      System.exit(-1);
    }
    return sum / sumWeights;
  }

  @Override
  public int dim() {
    return grid.size();
  }
}
