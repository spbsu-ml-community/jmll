package com.spbsu.ml.methods.trees;

import com.spbsu.commons.math.vectors.Mx;
import com.spbsu.commons.math.vectors.MxTools;
import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.commons.math.vectors.impl.mx.VecBasedMx;
import com.spbsu.commons.math.vectors.impl.vectors.ArrayVec;
import com.spbsu.ml.BFGrid;
import com.spbsu.ml.data.set.VecDataSet;
import com.spbsu.ml.loss.L2;
import com.spbsu.ml.loss.WeightedLoss;
import com.spbsu.ml.methods.greedyRegion.GreedyTDRegion;
import com.spbsu.ml.models.LinearOblivousTree;
import com.spbsu.ml.models.ObliviousTree;
import com.spbsu.ml.models.PolynomialObliviousTree;

import java.util.List;

/**
 * Created with IntelliJ IDEA.
 * User: towelenee
 * Date: 14.05.13
 * Time: 21:09
 */
public class GreedyObliviousPolynomialTree extends GreedyTDRegion {
    private final int depth;
    private final int numberOfVariables;
    private final int numberOfRegions;
    private final GreedyObliviousTree<WeightedLoss<L2>> got;
    private final int numberOfVariablesInRegion;
    private final double regulationCoefficient;
    private final double continuousFine;

    public GreedyObliviousPolynomialTree(final BFGrid grid, int depth) {
        super(grid);
        got = new GreedyObliviousTree<>(grid, depth);
        numberOfRegions = 1 << depth;
        numberOfVariablesInRegion = (depth + 1) * (depth + 2) / 2;
        numberOfVariables = numberOfRegions * numberOfVariablesInRegion;
        this.depth = depth;
        regulationCoefficient = 1000;
        continuousFine = 0;
    }

    public int convertMultiIndex(int mask, int i, int j) {
        if (i < j)
          return convertMultiIndex(mask, j, i);

        return mask * numberOfVariablesInRegion + i * (i + 1) / 2 + j;
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

    private void addInPointEqualCondition(final double[] point, int mask, int neighbourMask, Mx mx) {
        int cnt = 0;
        int index[] = new int[2 * numberOfVariablesInRegion];
        double coef[] = new double[2 * numberOfVariablesInRegion];
        for (int i = 0; i <= depth; i++) {
            for (int j = 0; j <= i; j++) {
                index[cnt] = convertMultiIndex(mask, i, j);
                coef[cnt++] = point[i] * point[j];
                index[cnt] = convertMultiIndex(neighbourMask, i, j);
                coef[cnt++] = -point[i] * point[j];
            }
        }
        addConditionToMatrix(mx, index, coef);
    }

    private Mx continuousConditions(final List<BFGrid.BinaryFeature> features) {
        Mx conditionMatrix = new VecBasedMx(numberOfVariables, numberOfVariables);
        for (int mask = 0; mask < 1 << depth; mask++) {
            for (int _featureNum = 0; _featureNum < depth; _featureNum++) {
                if (((mask >> _featureNum) & 1) == 0) {

                    double C = features.get(_featureNum).condition;
                    int neighbourMask = mask ^ (1 << _featureNum);

                    int featureNum = _featureNum + 1;
                    {
                        double[] point = new double[depth + 1];
                        point[0] = 1;
                        point[featureNum] = C;
                        addInPointEqualCondition(point, mask, neighbourMask, conditionMatrix);
                    }
                    for (int i = 1; i <= depth; i++) {
                        for (int j = 1; j <= i; j++) {
                            if ((i != featureNum) && (j != featureNum)) {
                                addConditionToMatrix(
                                        conditionMatrix,
                                        new int[]{convertMultiIndex(mask, i, j), convertMultiIndex(neighbourMask, i, j)},
                                        new double[]{1, -1}
                                );
                            }
                        }
                    }
                    for (int i = 1; i <= depth; i++) {
                        if (i != featureNum) {
                            addConditionToMatrix(
                                    conditionMatrix,
                                    new int[]{convertMultiIndex(mask, 0, i), convertMultiIndex(neighbourMask, 0, i), convertMultiIndex(mask, featureNum, i), convertMultiIndex(neighbourMask, featureNum, i)},
                                    new double[]{1, -1, C, -C}
                            );
                        }
                    }
                }
            }
        }
        return conditionMatrix;
    }


    private Vec calculateDiverativeVec(VecDataSet dataSet, WeightedLoss<L2> loss, BFGrid.BinaryFeature[] features) {
        Vec diverativeVec = new ArrayVec(numberOfVariables);
        for (int i = 0; i < loss.dim(); i++) {
            final double weight = loss.weight(i);
            final double target = loss.target().get(i);
                final Vec point = dataSet.data().row(i);
          int region = ObliviousTree.bin(features, point);
          double[] factors = GreedyObliviousLinearTree.getSignificantFactors(point, features);
          for (int x = 0; x <= depth; x++) {
                    for (int y = 0; y <= x; y++) {
                        diverativeVec.adjust(convertMultiIndex(region, x, y), -2 * weight * target * factors[x] * factors[y]);
                    }
                }
        }
        return diverativeVec;
    }
   private Mx calculateLossDiverativeMatrix(
            final VecDataSet dataSet,
            final WeightedLoss<L2> loss,
            final BFGrid.BinaryFeature[] features
    ) {
        Mx diverativeMx = new VecBasedMx(numberOfVariables, numberOfVariables);
        for (int i = 0; i < dataSet.xdim(); i++) {
            final double weight = loss.weight(i);
            final Vec point = dataSet.data().row(i);
            final int region = ObliviousTree.bin(features, point);
          double[] factors = GreedyObliviousLinearTree.getSignificantFactors(point, features);
          for (int x = 0; x <= depth; x++) {
                for (int y = 0; y <= x; y++) {
                    for (int x1 = 0; x1 <= depth; x1++) {
                        for (int y1 = 0; y1 <= x1; y1++) {
                            diverativeMx.adjust(
                                convertMultiIndex(region, x, y),
                                convertMultiIndex(region, x1, y1),
                                weight * factors[x] * factors[y] * factors[x1] * factors[y1]
                            );
                        }
                    }
                }
            }
        }
     return diverativeMx;
    }

    public PolynomialObliviousTree fit(VecDataSet ds, WeightedLoss<L2> loss) {
        BFGrid.BinaryFeature features[] = (BFGrid.BinaryFeature[])got.fit(ds, loss).features().toArray();
        final Mx diverativeMatrix = calculateLossDiverativeMatrix(ds, loss, features);
        final Vec diverativeVec = calculateDiverativeVec(ds, loss, features);
        GreedyObliviousLinearTree.addL2Regulation(diverativeMatrix, regulationCoefficient);

        /*final Mx continuousConditions = continuousConditions(features);
        for (int i = 0; i < numberOfVariables; ++i) {
            for (int j = 0; j < numberOfVariables; ++j) {
                diverativeMatrix.adjust(i, j, continuousConditions.get(i, j) * continuousFine);
            }
        }*/

        final Vec answer = MxTools.solveSystemLq(diverativeMatrix, diverativeVec);

        return new PolynomialObliviousTree(features, GreedyObliviousLinearTree.parseByRegions(answer, numberOfVariablesInRegion, numberOfRegions));
    }
}
