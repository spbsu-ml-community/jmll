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

/**
 * Created by towelenee on 4/16/15.
 */
public class GreedyObliviousLinearTree extends GreedyTDRegion {
  private final int depth;
  private final int numberOfVariables;
  private final int numberOfRegions;
  private final GreedyObliviousTree<WeightedLoss<L2>> got;
  private final int numberOfVariablesInRegion;
  private final double regulationCoefficient;

  public GreedyObliviousLinearTree(final BFGrid grid, int depth, double regulationCoefficient) {
    super(grid);
    this.regulationCoefficient = regulationCoefficient;
    got = new GreedyObliviousTree<>(grid, depth);
    numberOfRegions = 1 << depth;
    numberOfVariablesInRegion = (depth + 1);
    numberOfVariables = numberOfRegions * numberOfVariablesInRegion;
    this.depth = depth;
  }

  public int convertMultiIndex(int mask, int i) {
    return mask * numberOfVariablesInRegion + i;
  }

  public Mx calculateDevirativeMatrix(final VecDataSet ds, final BFGrid.BinaryFeature[] features, WeightedLoss<L2> loss) {
    Mx devirative = new VecBasedMx(numberOfVariables, numberOfVariables);
    for (int i = 0; i < ds.xdim(); i++) {
      Vec x = ds.data().row(i);
      int region = ObliviousTree.bin(features, x);
      double[] factors = getSignificantFactors(x, features);
      double weight = loss.weight(i);

      for (int x1 = 0; x1 <= features.length; x1++) {
        for (int x2 = 0; x2 <= features.length; x2++) {
          devirative.adjust(convertMultiIndex(region, x1), convertMultiIndex(region, x2), factors[x1] * factors[x2] * weight);
        }
      }
    }
    return devirative;
  }

  private Vec calculateDevirativeVec(VecDataSet ds, BFGrid.BinaryFeature[] features, WeightedLoss<L2> loss) {
    Vec vec = new ArrayVec(numberOfVariables);
    for (int i = 0; i < ds.xdim(); i++) {
      Vec x = ds.data().row(i);
      int region = ObliviousTree.bin(features, x);
      double[] factors = getSignificantFactors(x, features);
      double weight = loss.weight(i);
      double target = loss.target().get(i);

      for (int x1 = 0; x1 <= features.length; x1++) {
        vec.adjust(convertMultiIndex(region, x1), target * factors[x1] * weight);
      }
    }
    return vec;
  }


  LinearOblivousTree fit(final VecDataSet ds, final WeightedLoss<L2> loss) {
    ObliviousTree tree = got.fit(ds, loss);
    BFGrid.BinaryFeature[] features = (BFGrid.BinaryFeature[]) tree.features().toArray();
    Mx mx = calculateDevirativeMatrix(ds, features, loss);
    addL2Regulation(mx, regulationCoefficient);
    Vec vec = calculateDevirativeVec(ds, features, loss);
    Vec output = MxTools.solveSystemLq(mx, vec);
    return new LinearOblivousTree(tree.features(), parseByRegions(output, numberOfVariablesInRegion, numberOfRegions));

  }

  static public void addL2Regulation(final Mx mx, double regulationCoefficient) {
    for (int i = 0; i < mx.dim(); ++i) {
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


}
