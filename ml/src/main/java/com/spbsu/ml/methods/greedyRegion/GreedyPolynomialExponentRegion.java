package com.spbsu.ml.methods.greedyRegion;

import com.spbsu.commons.math.vectors.Mx;
import com.spbsu.commons.math.vectors.MxTools;
import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.commons.math.vectors.VecTools;
import com.spbsu.commons.math.vectors.impl.mx.VecBasedMx;
import com.spbsu.commons.math.vectors.impl.vectors.ArrayVec;
import com.spbsu.ml.BFGrid;
import com.spbsu.ml.data.set.VecDataSet;
import com.spbsu.ml.loss.L2;
import com.spbsu.ml.methods.VecOptimization;
import com.spbsu.ml.models.PolynomialExponentRegion;
import com.spbsu.ml.models.Region;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.PrintWriter;

/**
 * Created by towelenee on 20.02.14.
 */
public class GreedyPolynomialExponentRegion extends VecOptimization.Stub<L2> {
  private final GreedyTDRegion greedyTDRegion;
  private BFGrid.BinaryFeature[] features;
  private boolean[] mask;
  private final double distCoeffiecent, regulationCoeffiecent;

  public GreedyPolynomialExponentRegion(final BFGrid grid, final double distCoeffiecent, final double regulationCoeffiecent) {
    this.distCoeffiecent = distCoeffiecent;
    this.regulationCoeffiecent = regulationCoeffiecent;
    this.greedyTDRegion = new GreedyTDRegion<L2>(grid, 0.02, 0.5, 0);
  }

  double getDistanseFromRegion(final Vec x) {
    double distanse = 0;
    for (int i = 0; i < features.length; i++) {
      if (features[i].value(x) != mask[i])
        distanse += Math.pow(features[i].condition - x.get(features[i].findex), 2);
    }
    return distanse;
  }

  double sqr(final double x) {
    return x * x;
  }

  @Override
  public PolynomialExponentRegion fit(final VecDataSet learn, final L2 loss) {
    final Region base = greedyTDRegion.fit(learn, loss);
    features = base.features();
    mask = base.masks();
    double baseMse = 0;
    for (int i = 0; i < learn.length(); i++)
      baseMse += sqr(base.value(learn.data().row(i)) - loss.target.get(i));
    System.out.println("\nBase_MSE = " + baseMse);

    final int numberOfFeatures = features.length + 1;
    final int matrixSize = (features.length + 1) * (features.length + 1);
    final Mx mx = new VecBasedMx(matrixSize, matrixSize);
    final Vec linear = new ArrayVec(matrixSize);
    int countIn = 0;
    double sum = 0;
    for (int i = 0; i < learn.data().rows(); i++) {
      final Vec vec = learn.data().row(i);

      if (!base.contains(vec))
        continue;

      final double[] data = new double[numberOfFeatures];
      data[0] = 1;
      for (int j = 0; j < features.length; j++)
        data[j + 1] = vec.get(features[j].findex);
      final double f = loss.target.get(i);
      sum += f;
      countIn++;
      final double weight = 1;// Math.exp(-getDistanseFromRegion(vec) * distCoeffiecent);
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
    final Vec result = MxTools.solveSystemLq(mx, linear);
    //result.set(0, sum / countIn);
    //for (int i = 1; i < result.dim(); i++)
    //result.set(i, 0);
    System.out.println(result);
    final PolynomialExponentRegion ret = new PolynomialExponentRegion(features, mask, result.toArray(), distCoeffiecent);
    double mse = 0;
    for (int i = 0; i < learn.length(); i++)
      mse += sqr(ret.value(learn.data().row(i)) - loss.target.get(i));
    System.out.println("mse = " + mse);
    return ret;
  }

}
