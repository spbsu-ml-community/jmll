package com.spbsu.ml.models;

import com.spbsu.commons.math.MathTools;
import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.commons.math.vectors.VectorOfMultiplications;
import com.spbsu.ml.BFGrid;
import com.spbsu.ml.Func;
import com.spbsu.ml.methods.trees.GreedyObliviousPolynomialTree;

/**
 * Created with IntelliJ IDEA.
 * User: towelenee
 * Date: 14.05.13
 * Time: 20:50
 * To change this template use File | Settings | File Templates.
 */
public class PolynomialObliviousTree extends ObliviousTree {
  private final int dimensions;
  private final int numberOfVariablesInRegion;

  public PolynomialObliviousTree(final ObliviousTree based, final double[] values, int dimensions, int depth) {
    super(based.features(), values);
    this.dimensions = dimensions;
    numberOfVariablesInRegion = MathTools.combinationsWithRepetition(depth + 1, dimensions);
  }

  @Override
  public double value(final Vec point) {
    final int region = bin(point);
    double sum = 0;
    Vec factors = new VectorOfMultiplications(getSignificantFactors(point), dimensions);

    for (int i = 0; i < numberOfVariablesInRegion; i++) {
      sum += factors.get(i) * values[GreedyObliviousPolynomialTree.convertMultiIndex(region, i, numberOfVariablesInRegion)];
    }
    return sum;
  }
}
