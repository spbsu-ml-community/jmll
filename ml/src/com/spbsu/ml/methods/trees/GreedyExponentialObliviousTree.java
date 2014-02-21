package com.spbsu.ml.methods.trees;

import com.spbsu.commons.math.vectors.Mx;
import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.commons.math.vectors.VecTools;
import com.spbsu.commons.math.vectors.impl.ArrayVec;
import com.spbsu.commons.math.vectors.impl.VecBasedMx;
import com.spbsu.ml.BFGrid;
import com.spbsu.ml.data.DataSet;
import com.spbsu.ml.loss.L2;
import com.spbsu.ml.methods.GreedyPolynomialExponentRegion;
import com.spbsu.ml.methods.Optimization;
import com.spbsu.ml.models.ExponentialObliviousTree;

import java.util.List;
import java.util.Random;

/*Created with IntelliJ IDEA.
    *User:towelenee
    *Date:30.11.13
    *Time:17:48
    *Idea please stop making my code yellow
*/

public class GreedyExponentialObliviousTree implements Optimization<L2> {

  private final int numberOfVariablesByLeaf;
  private final int numberOfVariables;
  private double[][] quadraticMissCoefficient;
  private double[] linearMissCoefficient;
  private final double DistCoef;
  private final int depth;
  private final GreedyObliviousTree<L2> got;
  private List<BFGrid.BinaryFeature> features;

  public GreedyExponentialObliviousTree(Random rng, DataSet ds, BFGrid grid, int depth, double distCoef) {
    got = new GreedyObliviousTree(grid, depth);
    DistCoef = distCoef;
    this.depth = depth;
    numberOfVariablesByLeaf = (depth + 1) * (depth + 2) / 2;
    numberOfVariables = (1 << depth) * numberOfVariablesByLeaf;

  }

  public int getIndex(int mask, int i, int j) {
    if (i < j) {
      int temp = i;
      i = j;
      j = temp;
    }

    return mask * (depth + 1) * (depth + 2) / 2 + i * (i + 1) / 2 + j;
  }

  double sqr(double x) {
    return x * x;
  }

  double calcDistanseToRegion(int index, Vec point) {
    double ans = 0;
    for (int i = 0; i < features.size(); i++) {
      if (features.get(i).value(point) != ((index >> i) == 1)) {
        ans += sqr(point.get(features.get(i).findex) - features.get(i).condition);//L2
      }
    }

    return DistCoef * ans;
  }

  void precalculateMissCoefficients(DataSet ds, final L2 loss) {
    quadraticMissCoefficient = new double[numberOfVariables][numberOfVariables];
    linearMissCoefficient = new double[numberOfVariables];
    for (int i = 0; i < ds.power(); i++) {
      double data[] = new double[depth + 1];
      data[0] = 1;
      for (int s = 0; s < features.size(); s++) {
        data[s + 1] = ds.data().get(i, features.get(s).findex);
      }
      int index = 0;
      for (int j = 0; j < features.size(); j++)
        if (features.get(j).value(ds.data().row(i)))
          index |= 1 << j;
      double f = loss.target.get(i);
      double weight = 1; //Math.exp(-calcDistanseToRegion(index, ds.data().row(i)));
      //System.out.println(weight);
      for (int x = 0; x <= depth; x++)
        for (int y = 0; y <= x; y++) {
          linearMissCoefficient[getIndex(index, x, y)] -= f * data[x] * data[y] * weight;
        }
      for (int x = 0; x <= depth; x++)
        for (int y = 0; y <= x; y++)
          for (int x1 = 0; x1 <= depth; x1++)
            for (int y1 = 0; y1 <= x1; y1++)
              quadraticMissCoefficient[getIndex(index, x, y)][getIndex(index, x1, y1)] += data[x] * data[y] * data[x1] * data[y1] * weight;
    }
  }


  @Override
  public ExponentialObliviousTree fit(DataSet ds, final L2 loss) {
    features = got.fit(ds, loss).features();
    if (features.size() != depth) {
      System.out.println("Greedy oblivious tree bug");
      System.exit(-1);
    }

    precalculateMissCoefficients(ds, loss);
    System.out.println("Precalc is over");
    Mx a = new VecBasedMx(numberOfVariables, numberOfVariables);
    Vec b = new ArrayVec(numberOfVariables);
    for (int i = 0; i < numberOfVariables; i++)
      b.set(i, -linearMissCoefficient[i]);
    for (int i = 0; i < numberOfVariables; i++)
      for (int j = 0; j < numberOfVariables; j++)
        a.set(i, j, quadraticMissCoefficient[i][j]);
    //System.out.println(a);

    Vec value = GreedyPolynomialExponentRegion.solveLinearEquationUsingLQ(a,b);
    double out[][] = new double[1 << depth][(depth + 1) * (depth + 2) / 2];

    for (int i = 0; i < 1 << depth; i++)
      for (int k = 0; k <= depth; k++)
        for (int j = 0; j <= k; j++)
          out[i][k * (k + 1) / 2 + j] = value.get(getIndex(i, k, j));
    //for(int i =0 ; i < gradLambdas.size();i++)
    //    System.out.println(serializeCondtion(i));
    return new ExponentialObliviousTree(features, out, DistCoef);
  }


}
