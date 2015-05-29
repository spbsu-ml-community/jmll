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

import java.util.ArrayList;

/**
 * Created with IntelliJ IDEA.
 * User: towelenee
 * Date: 14.05.13
 * Time: 21:09
 */
public abstract class GreedyObliviousPolynomialTree extends GreedyObliviousTree<WeightedLoss<? extends L2>> {
  protected final int numberOfVariables;
  protected final VectorOfMultiplicationsFactory multiplicationsFactory;
  protected final int numberOfVariablesInRegion;

  public GreedyObliviousPolynomialTree(final BFGrid grid, int depth, int dimensions) {
    super(grid, depth);
    multiplicationsFactory = new VectorOfMultiplicationsFactory(depth + 1, dimensions);
    numberOfVariablesInRegion = multiplicationsFactory.getDim();
    numberOfVariables = numberOfVariablesInRegion << depth;
  }

  public static int convertMultiIndex(int region, int index, int numberOfVariablesInRegion) {
    return region * numberOfVariablesInRegion + index;
  }

  protected int convertMultiIndex(int region, int index)
  {
    if (convertMultiIndex(region, index, numberOfVariablesInRegion) >= numberOfVariables)
      throw new RuntimeException("");
    return convertMultiIndex(region, index, numberOfVariablesInRegion);
  }

  protected Vec calculateDerivativeVec(VecDataSet dataSet, WeightedLoss<? extends L2> loss, final ObliviousTree based) {
    Vec derivativeVec = new ArrayVec(numberOfVariables);
    for (int i = 0; i < dataSet.data().rows(); i++) {
      final double weight = loss.weight(i);
      final double target = loss.target().get(i);
      final Vec point = dataSet.data().row(i);
      int region = based.bin(point);

      final Vec factors = based.getSignificantFactors(point);

      for (int j = 0; j < numberOfVariablesInRegion; j++) {
        derivativeVec.adjust(convertMultiIndex(region, j), weight * target * multiplicationsFactory.get(factors, j));
      }
    }
    return derivativeVec;
  }

  protected ArrayList<Mx> calculateLossDerivativeMatrices(
      final VecDataSet dataSet,
      final WeightedLoss<? extends L2> loss,
      final ObliviousTree based
  ) {
    final ArrayList<Mx> matrices = new ArrayList<>(1 << depth);
    for (int i = 0; i < 1 << depth; i++) {
      matrices.add(new VecBasedMx(numberOfVariablesInRegion, numberOfVariablesInRegion));
    }
    for (int i = 0; i < dataSet.data().rows(); i++) {
      final double weight = loss.weight(i);
      final Vec point = dataSet.data().row(i);
      final int region = based.bin(point);
      final Vec factors = based.getSignificantFactors(point);
      for (int j = 0; j < numberOfVariablesInRegion; j++) {
        for (int g = 0; g < numberOfVariablesInRegion; g++) {
          matrices.get(region).adjust(
              j,
              g,
              weight * multiplicationsFactory.get(factors, j) * multiplicationsFactory.get(factors, g)
          );
        }
      }
    }
    return matrices;
  }
}
