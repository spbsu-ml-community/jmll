package com.spbsu.ml.methods.trees;

import com.spbsu.commons.math.vectors.Mx;
import com.spbsu.commons.math.vectors.MxTools;
import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.commons.math.vectors.VecTools;
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
 * Created by towelenee on 5/23/15.
 * Polynomial Trees with regions trying to be closer to each other
 */
public class PacGreedyPolynomialObliviousTree extends GreedyObliviousPolynomialTree {
  private final double regulationCoefficient;

  public PacGreedyPolynomialObliviousTree(BFGrid grid, int depth, int dimensions, double regulationCoefficient) {
    super(grid, depth, dimensions);
    this.regulationCoefficient = regulationCoefficient;
  }

  @Override
  public PolynomialObliviousTree fit(VecDataSet ds, WeightedLoss<? extends L2> loss) {
    final ObliviousTree based = super.fit(ds, loss);
    final Vec derivativeVec = calculateDerivativeVec(ds, loss, based);
    final ArrayList<Mx> matrices = calculateLossDerivativeMatrices(ds, loss, based);
    Vec oldCoefficients = new ArrayVec(numberOfVariables);
    Vec newCoefficients = new ArrayVec(numberOfVariables);
    for (int iter = 0; iter < 2; iter++) {
      for (int region = 0; region < 1 << depth; region++) {
        final double numberOfPointsInRegion = based.based()[region];
        if (numberOfPointsInRegion == 0) {
          continue;
        }
        Mx matrix = new VecBasedMx(numberOfVariablesInRegion, numberOfVariablesInRegion);
        VecTools.assign(matrix, matrices.get(region));
        Vec vector = new ArrayVec(numberOfVariablesInRegion);
        for (int i = 0; i < numberOfVariablesInRegion; i++) {
          double value = derivativeVec.get(convertMultiIndex(region, i));

          //adding regulation of near
          for (int feature = 0; feature < depth; feature++) {
            final int neighbourRegion = region ^ (1 << feature);
            if (based.based()[neighbourRegion] != 0) {
              value += regulationCoefficient * oldCoefficients.get(convertMultiIndex(neighbourRegion, i)) / numberOfPointsInRegion;
              matrix.adjust(i, i, regulationCoefficient / numberOfPointsInRegion);
            }

          }
          vector.set(i, value);
        }
        final Vec solution = MxTools.solveGaussZeildel(matrix, vector, 1e-3);
        for (int i = 0; i < numberOfVariablesInRegion; i++) {
          newCoefficients.set(convertMultiIndex(region, i), solution.get(i));
        }
      }
      VecTools.assign(oldCoefficients, newCoefficients);
    }
    return new PolynomialObliviousTree(based, oldCoefficients.toArray(), multiplicationsFactory);

  }
}
