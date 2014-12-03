package com.spbsu.ml.methods.trees;

import com.spbsu.commons.math.vectors.Mx;
import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.commons.math.vectors.impl.ArrayVec;
import com.spbsu.commons.math.vectors.impl.VecBasedMx;
import com.spbsu.ml.BFGrid;
import com.spbsu.ml.data.DataSet;
import com.spbsu.ml.loss.L2;
import com.spbsu.ml.loss.WeightedLoss;
import com.spbsu.ml.methods.GreedyPolynomialExponentRegion;
import com.spbsu.ml.methods.Optimization;
import com.spbsu.ml.models.ContinousObliviousTree;

import java.util.List;

/**
 * Created with IntelliJ IDEA.
 * User: towelenee
 * Date: 14.05.13
 * Time: 21:09
 */
public class GreedyContinuesObliviousSoftBondariesRegressionTree implements Optimization<WeightedLoss<L2>> {
  private final int depth;
  private final int numberOfVariables;
  private final int numberOfRegions;
  private final GreedyObliviousTree<WeightedLoss<L2>> got;
  private final int numberOfVariablesInRegion;
  private final double regulationCoefficient;
  private final double continousFine;

  public GreedyContinuesObliviousSoftBondariesRegressionTree(BFGrid grid, int depth) {
    got = new GreedyObliviousTree<WeightedLoss<L2>>(grid, depth);
    numberOfRegions = 1 << depth;
    numberOfVariablesInRegion = (depth + 1) * (depth + 2) / 2;
    numberOfVariables = numberOfRegions * numberOfVariablesInRegion;
    this.depth = depth;
    regulationCoefficient = 1000;
    continousFine = 0;
  }

  //Make 2 dimension index 1
  public int getIndex(int mask, int i, int j) {
    if (i < j) {
      int temp = i;
      i = j;
      j = temp;
    }

    return mask * (depth + 1) * (depth + 2) / 2 + i * (i + 1) / 2 + j;
  }

  public void addCondtionToMatrix(final Mx mx, int[] conditionIndexs, double[] conditionCoefficients) {
    double normalization = 0;
    for (double coeficient : conditionCoefficients) {
      normalization += coeficient * coeficient;
    }
    for (int i = 0; i < conditionCoefficients.length; i++)
      for (int j = 0; j < conditionCoefficients.length; j++)
        mx.adjust(conditionIndexs[i], conditionIndexs[j], conditionCoefficients[i] * conditionCoefficients[j] / normalization);
  }

  public void addInPointEqualCondition(double[] point, int mask, int neighbourMask, Mx mx) {
    int cnt = 0;
    int index[] = new int[2 * numberOfVariablesInRegion];
    double coef[] = new double[2 * numberOfVariablesInRegion];
    for (int i = 0; i <= depth; i++) {
      for (int j = 0; j <= i; j++) {
        index[cnt] = getIndex(mask, i, j);
        coef[cnt++] = point[i] * point[j];
        index[cnt] = getIndex(neighbourMask, i, j);
        coef[cnt++] = -point[i] * point[j];
      }
    }
    addCondtionToMatrix(mx, index, coef);
  }

  public Mx continousConditions(final List<BFGrid.BinaryFeature> features) {
    Mx conditionMatrix = new VecBasedMx(numberOfVariables, numberOfVariables);
    for (int mask = 0; mask < 1 << depth; mask++) {
      for (int _featureNum = 0; _featureNum < depth; _featureNum++) {
        if (((mask >> _featureNum) & 1) == 0) {

          double C = features.get(_featureNum).condition;
          int neighbourMask = mask ^ (1 << _featureNum);

          int featureNum = _featureNum + 1;
          //Equals at 0 points
          {

            double[] point = new double[depth + 1];
            /*
            for (int i = 0; i < depth; i++) {
              if ((numberOfPointInLeaf[mask] + numberOfPointInLeaf[neighbourMask]) != 0) {
                point[i + 1] = (coordinateSum[mask][i] + coordinateSum[neighbourMask][i]) / (double) (numberOfPointInLeaf[mask] + numberOfPointInLeaf[neighbourMask]);
              }
            }
            */
            point[0] = 1;
            point[featureNum] = C;
            addInPointEqualCondition(point, mask, neighbourMask, conditionMatrix);
          }
          //Quadratic boundary
          //All monoms must have equal coefficient in both leafs
          for (int i = 1; i <= depth; i++) {
            for (int j = 1; j <= i; j++) {
              if ((i != featureNum) && (j != featureNum)) {
                addCondtionToMatrix(
                        conditionMatrix,
                        new int[]{getIndex(mask, i, j), getIndex(neighbourMask, i, j)},
                        new double[]{1, -1}
                );
              }
            }
          }
          //Linear boundary
          for (int i = 1; i <= depth; i++) {
            if (i != featureNum) {
              addCondtionToMatrix(
                      conditionMatrix,
                      new int[]{getIndex(mask, 0, i), getIndex(neighbourMask, 0, i), getIndex(mask, featureNum, i), getIndex(neighbourMask, featureNum, i)},
                      new double[]{1, -1, C, -C}
              );
            }
          }
        }
      }
    }
    return conditionMatrix;
  }

  int getRegionOfPoint(final Vec point, final List<BFGrid.BinaryFeature> features) {
    int index = 0;
    for (BFGrid.BinaryFeature feature : features) {
      index <<= 1;
      if (feature.value(point)) {
        index++;
      }
    }
    return index;
  }

  Vec calculateDiverativeVec(DataSet dataSet, WeightedLoss<L2> loss, List<BFGrid.BinaryFeature> features) {
    Vec diverativeVec = new ArrayVec(numberOfVariables);
    for (int i = 0; i < dataSet.xdim(); i++) {
      final int weight = loss.getWeights()[i];
      final double target = loss.getMetric().target.get(i);
      if (weight == 0) {
        continue;
      }
      final Vec point = dataSet.data().row(i);
      final int index = getRegionOfPoint(point, features);
      for (int x = 0; x <= depth; x++) {
        for (int y = 0; y <= x; y++) {
          diverativeVec.adjust(getIndex(index, x, y), -2 * weight * target * point.get(x) * point.get(y));
        }
      }
    }
    return diverativeVec;
  }

  Mx calculateDiverativeMatrix(
          final DataSet dataSet,
          final WeightedLoss<L2> loss,
          final List<BFGrid.BinaryFeature> features
  ) {
    double[][][] quadraticMissCoefficient = new double[1 << depth][numberOfVariablesInRegion][numberOfVariablesInRegion];
    for (int i = 0; i < dataSet.xdim(); i++) {
      final int weight = loss.getWeights()[i];
      if (weight == 0) {
        continue;
      }
      final Vec point = dataSet.data().row(i);
      final int index = getRegionOfPoint(point, features);
      for (int x = 0; x <= depth; x++) {
        for (int y = 0; y <= x; y++) {
          for (int x1 = 0; x1 <= depth; x1++) {
            for (int y1 = 0; y1 <= x1; y1++) {
              quadraticMissCoefficient[index][getIndex(0, x, y)][getIndex(0, x1, y1)] += weight * point.get(x) * point.get(y) * point.get(x1) * point.get(y1);
            }
          }
        }
      }
    }

    Mx diverativeMx = new VecBasedMx(numberOfVariables, numberOfVariables);
    for (int index = 0; index < numberOfRegions; index++) {
      for (int x = 0; x <= depth; x++) {
        for (int y = 0; y <= depth; y++) {
          for (int x1 = 0; x1 <= depth; x1++) {
            for (int y1 = 0; y1 <= x1; y1++) {
              diverativeMx.set(
                      getIndex(index, x, y),
                      getIndex(index, x1, y1),
                      quadraticMissCoefficient[index][getIndex(0, x, y)][getIndex(0, x1, y1)]
              );
            }
          }
        }
      }
    }
    return diverativeMx;
  }

  public ContinousObliviousTree fit(DataSet ds, WeightedLoss<L2> loss) {
    final List<BFGrid.BinaryFeature> features = got.fit(ds, loss).features();
    if (features.size() != depth) {
      System.out.println("Greedy oblivious tree bug");
      System.exit(-1);
    }
    final Mx diverativeMatrix = calculateDiverativeMatrix(ds, loss, features);
    final Vec diverativeVec = calculateDiverativeVec(ds, loss, features);

    for (int i = 0; i < numberOfVariables; ++i) {
      diverativeMatrix.adjust(i, i, regulationCoefficient);
    }
    final Mx continousConditions = continousConditions(features);
    for (int i = 0; i < numberOfVariables; ++i) {
      for (int j = 0; j < numberOfVariables; ++j) {
        diverativeMatrix.adjust(i, j, continousConditions.get(i, j) * continousFine);
      }
    }

    final Vec answer = GreedyPolynomialExponentRegion.solveLinearEquationUsingLQ(diverativeMatrix, diverativeVec);
    double out[][] = new double[1 << depth][(depth + 1) * (depth + 2) / 2];
    for (int i = 0; i < 1 << depth; i++) {
      for (int k = 0; k <= depth; k++) {
        for (int j = 0; j <= k; j++) {
          out[i][k * (k + 1) / 2 + j] = -answer.get(getIndex(i, k, j));
        }
      }
    }
    return new ContinousObliviousTree(features, out);
  }


}