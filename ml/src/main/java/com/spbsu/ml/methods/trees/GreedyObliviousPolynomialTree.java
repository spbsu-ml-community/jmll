package com.spbsu.ml.methods.trees;

import com.spbsu.commons.math.vectors.Mx;
import com.spbsu.commons.math.vectors.MxTools;
import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.commons.math.vectors.impl.mx.VecBasedMx;
import com.spbsu.commons.math.vectors.impl.vectors.ArrayVec;
import com.spbsu.ml.BFGrid;
import com.spbsu.ml.Func;
import com.spbsu.ml.Trans;
import com.spbsu.ml.data.set.VecDataSet;
import com.spbsu.ml.loss.L2;
import com.spbsu.ml.loss.WeightedLoss;
import com.spbsu.ml.methods.VecOptimization;
import com.spbsu.ml.methods.greedyRegion.GreedyTDRegion;
import com.spbsu.ml.models.ObliviousTree;
import com.spbsu.ml.models.PolynomialObliviousTree;

import java.util.List;

/**
 * Created with IntelliJ IDEA.
 * User: towelenee
 * Date: 14.05.13
 * Time: 21:09
 */
public class GreedyObliviousPolynomialTree implements VecOptimization<WeightedLoss<? extends L2>> {
  private final int depth;
  private final int dimensions;
  private final int numberOfVariables;
  private final int numberOfRegions;
  private final GreedyObliviousTree<WeightedLoss<? extends L2>> got;
  private final int numberOfVariablesInRegion;
  private final double regulationCoefficient;
  private final double continuousFine;

  public GreedyObliviousPolynomialTree(final BFGrid grid, int depth, int dimensions, double regulationCoefficient) {
    this.dimensions = dimensions;
    this.regulationCoefficient = regulationCoefficient;
    got = new GreedyObliviousTree<>(grid, depth);
    numberOfRegions = 1 << depth;
    numberOfVariablesInRegion = count(depth, dimensions);
    numberOfVariables = numberOfRegions * numberOfVariablesInRegion;
    this.depth = depth;
    continuousFine = 0;
  }

  static int[][] answer = new int[100][100];

  public static int count(int maxElement, int dim) {
    if (dim == 0) {
      return 1;
    }
    if (answer[maxElement][dim] != 0) {
      return answer[maxElement][dim];
    }
    int sum = 0;
    for (int i = 0; i <= maxElement; i++) {
      sum += count(i, dim - 1);
    }
    answer[maxElement][dim] = sum;
    return sum;
  }

  public static double get(int number, int maxElement, int dim, final double[] feature) {
    if (dim == 0) {
      return 1;
    }
    for (int i = 0; i <= maxElement; i++) {
      if (count(i, dim - 1) <= number) {
        number -= count(i, dim - 1);
      } else {
        return feature[i] * get(number, i, dim - 1, feature);
      }
    }
    throw new IllegalArgumentException("");
  }

  static public void addL2Regulation(final Mx mx, double regulationCoefficient) {
    for (int i = 0; i < mx.rows(); ++i) {
      mx.adjust(i, i, regulationCoefficient);
    }
  }

  static public double[] getSignificantFactors(final Vec x, final BFGrid.BinaryFeature[] features) {
    double factors[] = new double[features.length + 1];
    factors[0] = 1;
    for (int j = 0; j < features.length; j++) {
      factors[j + 1] = x.get(features[j].findex);
    }
    return factors;

  }

  static public double[][] parseByRegions(Vec output, int numberOfVariablesInRegion, int numberOfRegions) {
    if (output.dim() != numberOfRegions * numberOfVariablesInRegion) {
      throw new IllegalArgumentException("output don't fit");
    }
    double[][] parsed = new double[numberOfRegions][numberOfVariablesInRegion];
    for (int i = 0; i < numberOfRegions; i++) {
      for (int j = 0; j < numberOfVariablesInRegion; j++) {
        parsed[i][j] = output.get(i * numberOfVariablesInRegion + j);
      }
    }
    return parsed;
  }

  private void addConditionToMatrix(final Mx mx, final int[] conditionIndexes, double[] conditionCoefficients) {
    double normalization = 0;
    for (double coefficient : conditionCoefficients) {
      normalization += coefficient * coefficient;
    }
    for (int i = 0; i < conditionCoefficients.length; i++) {
      for (int j = 0; j < conditionCoefficients.length; j++) {
        mx.adjust(conditionIndexes[i], conditionIndexes[j], conditionCoefficients[i] * conditionCoefficients[j] / normalization);
      }
    }
  }

  private int convertMultiIndex(int region, int index) {
    return region * numberOfVariablesInRegion + index;
  }

  private Vec calculateDiverativeVec(VecDataSet dataSet, WeightedLoss<? extends L2> loss, BFGrid.BinaryFeature[] features) {
    Vec diverativeVec = new ArrayVec(numberOfVariables);
    for (int i = 0; i < loss.dim(); i++) {
      final double weight = loss.weight(i);
      final double target = loss.target().get(i);
      final Vec point = dataSet.data().row(i);
      int region = ObliviousTree.bin(features, point);
      double[] factors = getSignificantFactors(point, features);
      for (int index = 0; index < numberOfVariablesInRegion; index++) {
        diverativeVec.adjust(convertMultiIndex(region, index), 2 * weight * target * get(index, factors.length - 1, dimensions, factors));
      }
    }
    return diverativeVec;
  }

  private Mx calculateLossDiverativeMatrix(
      final VecDataSet dataSet,
      final WeightedLoss<? extends L2> loss,
      final BFGrid.BinaryFeature[] features
  ) {
    Mx diverativeMx = new VecBasedMx(numberOfVariables, numberOfVariables);
    for (int i = 0; i < dataSet.xdim(); i++) {
      final double weight = loss.weight(i);
      final Vec point = dataSet.data().row(i);
      final int region = ObliviousTree.bin(features, point);
      double[] factors = getSignificantFactors(point, features);
      for (int index = 0; index < numberOfVariablesInRegion; index++) {
        for (int jindex = 0; jindex < numberOfVariablesInRegion; jindex++) {
          diverativeMx.adjust(
              convertMultiIndex(region, index),
              convertMultiIndex(region, jindex),
              weight * get(index, factors.length - 1, dimensions, factors) * get(jindex, factors.length - 1, dimensions, factors)
          );
        }
      }
    }
    return diverativeMx;
  }

  @Override
  public PolynomialObliviousTree fit(VecDataSet ds, WeightedLoss<? extends L2> loss) {
    final List<BFGrid.BinaryFeature> binaryFeatures = got.fit(ds, loss).features();
    BFGrid.BinaryFeature features[] =  binaryFeatures.toArray(new BFGrid.BinaryFeature[binaryFeatures.size()]);
    final Mx diverativeMatrix = calculateLossDiverativeMatrix(ds, loss, features);
    final Vec diverativeVec = calculateDiverativeVec(ds, loss, features);
    addL2Regulation(diverativeMatrix, regulationCoefficient);

    final Vec regressionCoefficients = MxTools.solveSystemLq(diverativeMatrix, diverativeVec);

    return new PolynomialObliviousTree(features, parseByRegions(regressionCoefficients, numberOfVariablesInRegion, numberOfRegions), dimensions, depth);
  }

}
