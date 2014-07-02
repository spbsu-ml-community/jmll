package com.spbsu.ml.methods;

import com.spbsu.commons.math.vectors.Mx;
import com.spbsu.commons.math.vectors.MxTools;
import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.commons.math.vectors.VecTools;
import com.spbsu.commons.math.vectors.impl.vectors.ArrayVec;
import com.spbsu.commons.math.vectors.impl.mx.VecBasedMx;
import com.spbsu.ml.BFGrid;
import com.spbsu.ml.data.DataSet;
import com.spbsu.ml.loss.L2;
import com.spbsu.ml.models.PolynomialExponentRegion;
import com.spbsu.ml.models.Region;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.PrintWriter;

/**
 * Created by towelenee on 20.02.14.
 */
public class GreedyPolynomialExponentRegion implements Optimization<L2> {
  private final GreedyTDRegion greedyTDRegion;
  private BFGrid.BinaryFeature[] features;
  private boolean[] mask;
  private final double distCoeffiecent, regulationCoeffiecent;

  public GreedyPolynomialExponentRegion(BFGrid grid, double distCoeffiecent, double regulationCoeffiecent) {
    this.distCoeffiecent = distCoeffiecent;
    this.regulationCoeffiecent = regulationCoeffiecent;
    this.greedyTDRegion = new GreedyTDRegion<L2>(grid);
  }

  double getDistanseFromRegion(Vec x) {
    double distanse = 0;
    for (int i = 0; i < features.length; i++) {
      if (features[i].value(x) != mask[i])
        distanse += Math.pow(features[i].condition - x.get(features[i].findex), 2);
    }
    return distanse;
  }

  public static boolean validateSolution(Mx a, Vec right, Vec sol) {
    Vec val = MxTools.multiply(a, sol);
    double l2 = VecTools.distance(val, right);
    if (l2 > right.dim()) {
      /*for (int i = 0; i < right.dim(); i++)
        System.out.format("%f = %f\n", val.at(i), right.at(i));*/
    }
    //System.out.println(l2);
    return l2 < 0.1 * right.dim();
  }

  public static Vec solveLinearEquationUsingLQ(Mx mx, Vec right) {
    if (mx.rows() != mx.columns())
      throw new IllegalArgumentException("Matrix must be quadratic");
    if (right.dim() != mx.rows())
      throw new IllegalArgumentException("Vector must be the same size as Matrix");
    int n = mx.rows();
    Mx l = new VecBasedMx(n, n);
    Mx q = new VecBasedMx(n, n);
    MxTools.householderLQ(mx, l, q);
    //System.out.println(VecTools.inverseLTriangle(l));
    //System.out.println(VecTools.multiply(q,VecTools.transpose(q)));
    Vec first = new ArrayVec(n);
    for (int i = 0; i < n; i++) {
      if (Math.abs(l.get(i, i)) > 1e-5) {
        double val = right.get(i);
        for (int j = 0; j < i; j++)
          val -= first.get(j) * l.get(i, j);
        first.set(i, val / l.get(i, i));
      } else {
        first.set(i, 0);
      }
    }
    Vec ans = MxTools.multiply(q, first);
    if (!validateSolution(mx, right, ans)) {
      PrintWriter printWriter = null;
      try {
        printWriter = new PrintWriter(new File("badMx.txt"));
      } catch (FileNotFoundException e) {
        e.printStackTrace();
      }
      printWriter.println(mx.rows());
      for (int i = 0; i < mx.rows(); i++)
        for (int j = i + 1; j < mx.rows(); j++)
          if (l.get(i, j) > 1e-5)
            System.out.println("bad l" + l.get(i,j));
      printWriter.println(mx);
      printWriter.println(right);
      printWriter.close();
      throw new RuntimeException("Not correct work of solveLinearEquationUsingLQ");
    }
    return ans;
  }

  double sqr(double x) {
    return x * x;
  }

  @Override
  public PolynomialExponentRegion fit(DataSet learn, L2 loss) {
    Region base = greedyTDRegion.fit(learn, loss);
    features = base.getFeatures();
    mask = base.getMask();
    double baseMse = 0;
    for (int i = 0; i < learn.power(); i++)
      baseMse += sqr(base.value(learn.data().row(i)) - loss.target.get(i));
    System.out.println("\nBase_MSE = " + baseMse);

    int numberOfFeatures = features.length + 1;
    int matrixSize = (features.length + 1) * (features.length + 1);
    Mx mx = new VecBasedMx(matrixSize, matrixSize);
    Vec linear = new ArrayVec(matrixSize);
    int countIn = 0;
    double sum = 0;
    for (int i = 0; i < learn.data().rows(); i++) {
      Vec vec = learn.data().row(i);

      if (!base.contains(vec))
        continue;

      double data[] = new double[numberOfFeatures];
      data[0] = 1;
      for (int j = 0; j < features.length; j++)
        data[j + 1] = vec.get(features[j].findex);
      double f = loss.target.get(i);
      sum += f;
      countIn++;
      double weight = 1;// Math.exp(-getDistanseFromRegion(vec) * distCoeffiecent);
      //System.out.println(weight);
      for (int x = 0; x < numberOfFeatures; x++)
        for (int y = 0; y < numberOfFeatures; y++)
          linear.adjust(x + y * numberOfFeatures, -data[x] * data[y] * weight * f);
      for (int x = 0; x < numberOfFeatures; x++)
        for (int y = 0; y < numberOfFeatures; y++)
          for (int xx = 0; xx < numberOfFeatures; xx++)
            for (int yy = 0; yy < numberOfFeatures; yy++)
              mx.adjust(x + y * numberOfFeatures, xx + yy * numberOfFeatures, data[x] * data[y] * data[xx] * data[yy] * weight);
    }
    System.out.println("\n" + countIn + " " + sum);
    //for (int i = 0; i < matrixSize; i++)
    //mx.adjust(i, i, regulationCoeffiecent);
    for (int i = 0; i < matrixSize; i++)
      linear.set(i, -linear.get(i));

    //System.out.println(mx);
    //System.out.println(linear);
    //System.out.println(VecTools.inverseCholesky(mx));
    Vec result = solveLinearEquationUsingLQ(mx, linear);
    //result.set(0, sum / countIn);
    //for (int i = 1; i < result.dim(); i++)
    //result.set(i, 0);
    System.out.println(result);
    PolynomialExponentRegion ret = new PolynomialExponentRegion(features, mask, result.toArray(), distCoeffiecent);
    double mse = 0;
    for (int i = 0; i < learn.power(); i++)
      mse += sqr(ret.value(learn.data().row(i)) - loss.target.get(i));
    System.out.println("mse = " + mse);
    return ret;
  }

}
