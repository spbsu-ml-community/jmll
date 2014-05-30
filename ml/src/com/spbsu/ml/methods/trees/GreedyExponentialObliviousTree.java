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
import com.spbsu.ml.models.ExponentialObliviousTree;
import com.spbsu.ml.models.ObliviousTree;

import java.util.List;

/*Created with IntelliJ IDEA.
    *User:towelenee
    *Date:30.11.13
    *Time:17:48
    *Idea please stop making my code yellow
*/

public class GreedyExponentialObliviousTree implements Optimization<WeightedLoss<L2>> {

  private final BFGrid grid;
  private int numberOfVariablesByLeaf;
  private double[][][] quadraticMissCoefficient;
  private double[][] linearMissCoefficient;
  private final double DistCoef;
  private int depth;
  private final GreedyObliviousTree<WeightedLoss<L2>> got;
  private List<BFGrid.BinaryFeature> features;

  public GreedyExponentialObliviousTree(BFGrid grid, int depth, double distCoef) {
    this.grid = grid;
    got = new GreedyObliviousTree<WeightedLoss<L2>>(grid, depth);
    DistCoef = distCoef;
  }

  public int getIndex(int mask, int i, int j) {
    if (i < j) {
      int temp = i;
      i = j;
      j = temp;
    }

    return mask * (depth + 1) * (depth + 2) / 2 + i * (i + 1) / 2 + j;
  }

  static double sqr(double x) {
    return x * x;
  }

  public static double calcDistanseToRegion(BFGrid grid, double distCoef, int index, Vec point, List<BFGrid.BinaryFeature> features) {
    double ans = 0;
    byte[] bin = new byte[point.dim()];
    grid.binarize(point, bin);

    for (int i = 0; i < features.size(); i++) {
      if (features.get(i).value(point) != (((index >> i) & 1) == 1)) {
        ans += sqr(bin[(features.get(i).findex)] - features.get(i).binNo);//L2
      }
    }
    return Math.exp(-distCoef * ans);
  }

  void precalculateMissCoefficients(DataSet ds, final WeightedLoss<L2> loss) {
    quadraticMissCoefficient = new double[1 << depth][numberOfVariablesByLeaf][numberOfVariablesByLeaf];
    linearMissCoefficient = new double[1 << depth][numberOfVariablesByLeaf];
    for (int i = 0; i < ds.power(); i++) {
      double data[] = new double[depth + 1];
      data[0] = 1;
      for (int s = 0; s < features.size(); s++) {
        data[s + 1] = ds.data().get(i, features.get(s).findex);
      }

      for (int index = 0; index < 1 << depth; index++) {
        double f = loss.getMetric().target.get(i);
        double weight = loss.getWeights()[i] * calcDistanseToRegion(grid, DistCoef, index, ds.data().row(i), features);
        if (weight > 1e-9) {
          for (int x = 0; x <= depth; x++)
            for (int y = 0; y <= x; y++) {
              linearMissCoefficient[index][getIndex(0, x, y)] -= 2 * f * data[x] * data[y] * weight;
            }
          for (int x = 0; x <= depth; x++)
            for (int y = 0; y <= x; y++)
              for (int x1 = 0; x1 <= depth; x1++)
                for (int y1 = 0; y1 <= x1; y1++)
                  quadraticMissCoefficient[index][getIndex(0, x, y)][getIndex(0, x1, y1)] += data[x] * data[y] * data[x1] * data[y1] * weight;
        }
      }
    }
  }


  @Override
  public ExponentialObliviousTree fit(DataSet ds, final WeightedLoss<L2> loss) {
    ObliviousTree base = got.fit(ds, loss);
    features = base.features();
    double baseMse = 0;
    for (int i = 0; i < ds.power(); i++)
      baseMse += sqr(base.value(ds.data().row(i)) - loss.getMetric().target.get(i));
    System.out.println("\nBase_MSE = " + baseMse);
    depth = features.size();
    numberOfVariablesByLeaf = (depth + 1) * (depth + 2) / 2;

    precalculateMissCoefficients(ds, loss);
    double out[][] = new double[1 << depth][(depth + 1) * (depth + 2) / 2];
    for (int index = 0; index < 1 << depth; index++) {
      Mx a = new VecBasedMx(numberOfVariablesByLeaf, numberOfVariablesByLeaf);
      Vec b = new ArrayVec(numberOfVariablesByLeaf);
      for (int i = 0; i < numberOfVariablesByLeaf; i++)
        b.set(i, -linearMissCoefficient[index][i]);
      for (int i = 0; i < numberOfVariablesByLeaf; i++)
        for (int j = 0; j < numberOfVariablesByLeaf; j++)
          a.set(i, j, quadraticMissCoefficient[index][i][j]);
      for (int i = 0; i < numberOfVariablesByLeaf; i++)
        a.adjust(i, i, 1e-1);
      Vec value = GreedyPolynomialExponentRegion.solveLinearEquationUsingLQ(a, b);
      for (int k = 0; k <= depth; k++)
        for (int j = 0; j <= k; j++)
          out[index][k * (k + 1) / 2 + j] = value.get(getIndex(0, k, j));
    }
    ExponentialObliviousTree ret = new ExponentialObliviousTree(features, out, DistCoef, grid);
    double mse = 0;
    for (int i = 0; i < ds.power(); i++)
      mse += sqr(ret.value(ds.data().row(i)) - loss.getMetric().target.get(i));

    System.out.println("MSE = " + mse);
    return ret;
  }


}
