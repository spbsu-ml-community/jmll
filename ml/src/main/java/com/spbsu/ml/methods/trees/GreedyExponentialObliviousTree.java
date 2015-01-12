package com.spbsu.ml.methods.trees;

import com.spbsu.commons.math.vectors.Mx;
import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.commons.math.vectors.impl.mx.VecBasedMx;
import com.spbsu.commons.math.vectors.impl.vectors.ArrayVec;
import com.spbsu.ml.BFGrid;
import com.spbsu.ml.data.set.DataSet;
import com.spbsu.ml.data.set.VecDataSet;
import com.spbsu.ml.loss.L2;
import com.spbsu.ml.methods.VecOptimization;
import com.spbsu.ml.methods.greedyRegion.GreedyPolynomialExponentRegion;
import com.spbsu.ml.models.ExponentialObliviousTree;
import com.spbsu.ml.models.ObliviousTree;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.PrintWriter;
import java.util.List;

/*Created with IntelliJ IDEA.
    *User:towelenee
    *Date:30.11.13
    *Time:17:48
    *Idea please stop making my code yellow
*/

public class GreedyExponentialObliviousTree extends VecOptimization.Stub<L2> {

  private final int numberOfVariablesByLeaf;
  private final int numberOfVariables;
  private double[][][] quadraticMissCoefficient;
  private double[][] linearMissCoefficient;
  private final double DistCoef;
  private final int depth;
  private final GreedyObliviousTree<L2> got;
  private List<BFGrid.BinaryFeature> features;

  public GreedyExponentialObliviousTree(final BFGrid grid, final int depth, final double distCoef) {
    got = new GreedyObliviousTree(grid, depth);
    DistCoef = distCoef;
    this.depth = depth;
    numberOfVariablesByLeaf = (depth + 1) * (depth + 2) / 2;
    numberOfVariables = (1 << depth) * numberOfVariablesByLeaf;
  }

  public int getIndex(final int mask, int i, int j) {
    if (i < j) {
      final int temp = i;
      i = j;
      j = temp;
    }

    return mask * (depth + 1) * (depth + 2) / 2 + i * (i + 1) / 2 + j;
  }

  double sqr(final double x) {
    return x * x;
  }

  double calcDistanseToRegion(final int index, final Vec point) {
    double ans = 0;
    for (int i = 0; i < features.size(); i++) {
      if (features.get(i).value(point) != ((index >> i) == 1)) {
        ans += sqr(point.get(features.get(i).findex) - features.get(i).condition);//L2
      }
    }

    return DistCoef * ans;
  }

  void precalculateMissCoefficients(final DataSet ds, final L2 loss) {
    quadraticMissCoefficient = new double[1 << depth][numberOfVariablesByLeaf][numberOfVariablesByLeaf];
    linearMissCoefficient = new double[1 << depth][numberOfVariablesByLeaf];
    for (int i = 0; i < ds.length(); i++) {
      final double[] data = new double[depth + 1];
      data[0] = 1;
      for (int s = 0; s < features.size(); s++) {
        data[s + 1] = ((VecDataSet) ds).data().get(i, features.get(s).findex);
      }
      int index = 0;
      for (int j = 0; j < features.size(); j++) {
        index <<= 1;
        if (features.get(j).value(((VecDataSet) ds).data().row(i)))
          index++;
      }
      //if(index == 1)
      //  System.out.println(lines.at(0).condition);
      final double f = loss.target.get(i);
      final double weight = 1; //Math.exp(-calcDistanseToRegion(index, ds.data().row(i)));
      //System.out.println(weight);
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


  @Override
  public ExponentialObliviousTree fit(final VecDataSet ds, final L2 loss) {
    final ObliviousTree base = got.fit(ds, loss);
    features = base.features();
    double baseMse = 0;
    for (int i = 0; i < ds.length(); i++)
      baseMse += sqr(base.value(ds.data().row(i)) - loss.target.get(i));
    System.out.println("\nBase_MSE = " + baseMse);

    if (features.size() != depth) {
      System.out.println("Oblivious Tree bug");
      try {
        final PrintWriter printWriter = new PrintWriter(new File("badloss.txt"));
        for (int i = 0; i < ds.length(); i++)
          printWriter.println(loss.target.get(i));
        printWriter.close();
      } catch (FileNotFoundException e) {
        e.printStackTrace();
      }
      System.exit(-1);
    }

    precalculateMissCoefficients(ds, loss);
    //System.out.println("Precalc is over");
    final double[][] out = new double[1 << depth][(depth + 1) * (depth + 2) / 2];
    for (int index = 0; index < 1 << depth; index++) {
      final Mx a = new VecBasedMx(numberOfVariablesByLeaf, numberOfVariablesByLeaf);
      final Vec b = new ArrayVec(numberOfVariablesByLeaf);
      for (int i = 0; i < numberOfVariablesByLeaf; i++)
        b.set(i, -linearMissCoefficient[index][i]);
      for (int i = 0; i < numberOfVariablesByLeaf; i++)
        for (int j = 0; j < numberOfVariablesByLeaf; j++)
          a.set(i, j, quadraticMissCoefficient[index][i][j]);
      for (int i = 0; i < numberOfVariablesByLeaf; i++)
        a.adjust(i, i, 1e-1);
      final Vec value = GreedyPolynomialExponentRegion.solveLinearEquationUsingLQ(a, b);
      //System.out.println(a);
      for (int k = 0; k <= depth; k++)
        for (int j = 0; j <= k; j++)
          out[index][k * (k + 1) / 2 + j] = value.get(getIndex(0, k, j));
      /*if(quadraticMissCoefficient[index][0][0] != 0)
        out[index][0] = linearMissCoefficient[index][0] / quadraticMissCoefficient[index][0][0];*/
      //out[index][0] = base.values()[index];
      //for (int i = 0; i < out[index].length; i++)
      //System.out.println(out[index][i]);
    }
    //for(int i =0 ; i < gradLambdas.size();i++)
    //    System.out.println(serializeCondtion(i));
    final ExponentialObliviousTree ret = new ExponentialObliviousTree(features, out, DistCoef);
    double mse = 0;
    for (int i = 0; i < ds.length(); i++)
      mse += sqr(ret.value(ds.data().row(i)) - loss.target.get(i));
    System.out.println("MSE = " + mse);
    /*if (mse > baseMse + 1e-5)
      try {
        throw new Exception("Bad model work mse of based model less than mse of extended model");
      } catch (Exception e) {
        e.printStackTrace();
        //System.exit(-1);
      }*/
    return ret;
  }


}
