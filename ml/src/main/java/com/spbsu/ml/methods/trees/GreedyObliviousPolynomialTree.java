package com.spbsu.ml.methods.trees;

import com.spbsu.commons.math.vectors.Mx;
import com.spbsu.commons.math.vectors.MxTools;
import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.commons.math.vectors.VectorOfMultiplicationsFactory;
import com.spbsu.commons.math.vectors.impl.mx.VecBasedMx;
import com.spbsu.commons.math.vectors.impl.vectors.ArrayVec;
import com.spbsu.ml.BFGrid;
import com.spbsu.ml.data.set.VecDataSet;
import com.spbsu.ml.loss.L2;
import com.spbsu.ml.loss.WeightedLoss;
import com.spbsu.ml.models.ObliviousTree;
import com.spbsu.ml.models.PolynomialObliviousTree;

/**
 * Created with IntelliJ IDEA.
 * User: towelenee
 * Date: 14.05.13
 * Time: 21:09
 */
public class GreedyObliviousPolynomialTree extends GreedyObliviousTree<WeightedLoss<? extends L2>> {
  private final int numberOfVariables;
  private final double regulationCoefficient;
  private final VectorOfMultiplicationsFactory multiplicationsFactory;
  private final int numberOfVariablesInRegion;

  public GreedyObliviousPolynomialTree(final BFGrid grid, int depth, int dimensions, double regulationCoefficient) {
    super(grid, depth);
    this.regulationCoefficient = regulationCoefficient;
    multiplicationsFactory = new VectorOfMultiplicationsFactory(depth + 1, dimensions);
    numberOfVariablesInRegion = multiplicationsFactory.getDim();
    numberOfVariables = numberOfVariablesInRegion << depth;
  }

  public static int convertMultiIndex(int region, int index, int numberOfVariablesInRegion) {
    return region * numberOfVariablesInRegion + index;
  }

  private int convertMultiIndex(int region, int index)
  {
    return convertMultiIndex(region, index, multiplicationsFactory.getDim());
  }

  private Vec calculateDerivativeVec(VecDataSet dataSet, WeightedLoss<? extends L2> loss, final ObliviousTree based) {
    Vec derivativeVec = new ArrayVec(numberOfVariables);
    for (int i = 0; i < loss.dim(); i++) {
      final double weight = loss.weight(i);
      final double target = loss.target().get(i);
      final Vec point = dataSet.data().row(i);
      int region = based.bin(point);

      final Vec factors = based.getSignificantFactors(point);

      for (int j = 0; j < numberOfVariablesInRegion; j++) {
        derivativeVec.adjust(convertMultiIndex(region, j), 2 * weight * target * multiplicationsFactory.get(factors, j));
      }
    }
    return derivativeVec;
  }

  private Mx calculateLossDerivativeMatrix(
      final VecDataSet dataSet,
      final WeightedLoss<? extends L2> loss,
      final ObliviousTree based
  ) {
    Mx derivativeMx = new VecBasedMx(numberOfVariables, numberOfVariables);
    for (int i = 0; i < dataSet.xdim(); i++) {
      final double weight = loss.weight(i);
      final Vec point = dataSet.data().row(i);
      final int region = based.bin(point);
      final Vec factors = based.getSignificantFactors(point);
      for (int j = 0; j < numberOfVariablesInRegion; j++) {
        for (int g = 0; g < numberOfVariablesInRegion; g++) {
          derivativeMx.adjust(
              convertMultiIndex(region, j),
              convertMultiIndex(region, g),
              weight * multiplicationsFactory.get(factors, j) * multiplicationsFactory.get(factors, g)
          );
        }
      }
    }
    return derivativeMx;
  }

  @Override
  public PolynomialObliviousTree fit(VecDataSet ds, WeightedLoss<? extends L2> loss) {
    final ObliviousTree based = super.fit(ds, loss);
    final Mx derivativeMatrix = calculateLossDerivativeMatrix(ds, loss, based);
    final Vec derivativeVec = calculateDerivativeVec(ds, loss, based);

    { // Adding regulation
      for (int i = 0; i < derivativeMatrix.rows(); ++i) {
        derivativeMatrix.adjust(i, i, regulationCoefficient);
      }
    }

    final Vec regressionCoefficients = MxTools.solveSystemLq(derivativeMatrix, derivativeVec);

    return new PolynomialObliviousTree(based, regressionCoefficients.toArray(), multiplicationsFactory);
  }

}
