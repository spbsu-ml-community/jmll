package com.spbsu.ml.methods.trees;

import com.spbsu.commons.math.vectors.Mx;
import com.spbsu.commons.math.vectors.MxTools;
import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.commons.math.vectors.impl.mx.SparseMx;
import com.spbsu.ml.BFGrid;
import com.spbsu.ml.data.set.VecDataSet;
import com.spbsu.ml.loss.L2;
import com.spbsu.ml.loss.WeightedLoss;
import com.spbsu.ml.models.ObliviousTree;
import com.spbsu.ml.models.PolynomialObliviousTree;

import java.util.List;

/**
 * Created by towelenee on 5/23/15.
 */
public class GreedyObliviousPolynomialTreeImpl extends GreedyObliviousPolynomialTree {
  private final double regulationCoefficient;

  public GreedyObliviousPolynomialTreeImpl(BFGrid grid, int depth, int dimensions, double regulationCoefficient) {
    super(grid, depth, dimensions);
    this.regulationCoefficient = regulationCoefficient;
  }

  @Override
  public PolynomialObliviousTree fit(VecDataSet ds, WeightedLoss<? extends L2> loss) {
    final ObliviousTree based = super.fit(ds, loss);
    final Vec derivativeVec = calculateDerivativeVec(ds, loss, based);
    SparseMx derivativeMatrix = new SparseMx(numberOfVariables, numberOfVariables);
    {
      final List<Mx> matrixes = calculateLossDerivativeMatrices(ds, loss, based);

      for (int region = 0; region < matrixes.size(); region++) {
        for (int i = 0; i < numberOfVariablesInRegion; i++) {
          for (int j = 0; j < numberOfVariablesInRegion; j++) {
            derivativeMatrix.set(
                convertMultiIndex(region, i),
                convertMultiIndex(region, j),
                matrixes.get(region).get(i, j)
            );
          }
        }
      }

    }

    { // Adding regulation
      for (int region = 0; region < (1 << depth); region++) {
        for (int feature = 0; feature < depth; feature++) {
          final int neighbourRegion = region ^ (1 << feature);
          for (int i = 0; i < numberOfVariablesInRegion; i++) {
            derivativeMatrix.adjust(convertMultiIndex(region, i), convertMultiIndex(region, i), regulationCoefficient);
            derivativeMatrix.adjust(convertMultiIndex(region, i), convertMultiIndex(neighbourRegion, i), -regulationCoefficient);
          }
        }
      }

      /*for (int i = 0; i < derivativeMatrix.rows(); ++i) {
        derivativeMatrix.adjust(i, i, regulationCoefficient);
      }*/
    }

    final Vec regressionCoefficients = MxTools.solveGaussZeildel(derivativeMatrix, derivativeVec);
//    final Vec regressionCoefficients = MxTools.solveCholesky(derivativeMatrix, derivativeVec);
    MxTools.multiply(derivativeMatrix, regressionCoefficients);
    return new PolynomialObliviousTree(based, regressionCoefficients.toArray(), multiplicationsFactory);
  }

}
