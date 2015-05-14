package com.spbsu.ml.models;

import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.commons.math.vectors.VectorOfMultiplicationsFactory;
import com.spbsu.ml.methods.trees.GreedyObliviousPolynomialTree;

/**
 * Created with IntelliJ IDEA.
 * User: towelenee
 * Date: 14.05.13
 * Time: 20:50
 * To change this template use File | Settings | File Templates.
 */
public class PolynomialObliviousTree extends ObliviousTree {
  private final VectorOfMultiplicationsFactory multiplicationsFactory;
  private final int numberOfVariablesInRegion;

  public PolynomialObliviousTree(
      final ObliviousTree based,
      final double[] values,
      final VectorOfMultiplicationsFactory multiplicationsFactory
  ) {
    super(based.features(), values);
    this.multiplicationsFactory = multiplicationsFactory;
    numberOfVariablesInRegion = multiplicationsFactory.getDim();
  }

  @Override
  public double value(final Vec point) {
    final int region = bin(point);
    double sum = 0;
    final Vec factors = getSignificantFactors(point);
    for (int i = 0; i < numberOfVariablesInRegion; i++) {
      sum +=
          multiplicationsFactory.get(factors, i) *
          values[GreedyObliviousPolynomialTree.convertMultiIndex(region, i, numberOfVariablesInRegion)];
    }
    return sum;
  }
}
