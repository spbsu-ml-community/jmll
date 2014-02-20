package com.spbsu.ml.methods;

import com.spbsu.commons.math.vectors.Mx;
import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.commons.math.vectors.VecTools;
import com.spbsu.commons.math.vectors.impl.ArrayVec;
import com.spbsu.commons.math.vectors.impl.VecBasedMx;
import com.spbsu.ml.BFGrid;
import com.spbsu.ml.data.DataSet;
import com.spbsu.ml.loss.L2;
import com.spbsu.ml.models.PolynomialExponentRegion;
import com.spbsu.ml.models.Region;

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
        distanse += Math.pow(features[i].condition - x.get(features[i].bfIndex), 2);
    }
    return distanse;
  }
  public boolean validateSolution(Mx a, Vec right, Vec sol) {
    return VecTools.distance(VecTools.multiply(a, sol),right) < 1e-3 * right.dim();
  }
  public Vec solveLinearEquationUsingLQ(Mx mx, Vec right) {
    if (mx.rows() != mx.columns())
      throw new IllegalArgumentException("Matrix must be quadratic");
    if (right.dim() != mx.rows())
      throw new IllegalArgumentException("Vector must be the same size as Matrix");
    int n = mx.rows();
    Mx l = new VecBasedMx(n, n);
    Mx q = new VecBasedMx(n, n);
    VecTools.householderLQ(mx, l, q);
    Vec first = new ArrayVec(n);
    for (int i = 0; i < n; i++) {
      if (Math.abs(l.get(i,i)) > 1e-5) {
        double val = right.get(i);
        for (int j = 0; j < i; j++)
          val -= first.get(j) * l.get(i, j);
        first.set(i, val / l.get(i, i));
      } else
        first.set(i, 0);
    }
    Vec ans = VecTools.multiply(VecTools.transpose(q), first);
    if (!validateSolution(mx, right, ans))
      throw new RuntimeException("Not correct work of solveLinearEquationUsingLQ");
    return ans;
  }

  @Override
  public PolynomialExponentRegion fit(DataSet learn, L2 loss) {
    Region region = greedyTDRegion.fit(learn, loss);
    features = region.getFeatures();
    mask = region.getMask();
    int numberOfFeatures = features.length + 1;
    int matrixSize = (features.length + 1) * (features.length + 1);
    Mx mx = new VecBasedMx(matrixSize, matrixSize);
    Vec linear = new ArrayVec(matrixSize);
    for (int i = 0; i < learn.data().rows(); i++) {
      Vec vec = learn.data().row(i);
      double data[] = new double[numberOfFeatures];
      data[0] = 1;
      for (int j = 0; j < features.length; j++)
        data[j + 1] = vec.get(j);
      double f = loss.target.get(i);
      double weight = Math.exp(-getDistanseFromRegion(vec) * distCoeffiecent);
      for (int x = 0; x < numberOfFeatures; x++)
        for (int y = 0; y < numberOfFeatures; y++)
          linear.adjust(x + y * numberOfFeatures, -data[x] * data[y] * weight * f);
      for (int x = 0; x < numberOfFeatures; x++)
        for (int y = 0; y < numberOfFeatures; y++)
          for (int xx = 0; xx < numberOfFeatures; xx++)
            for (int yy = 0; yy < numberOfFeatures; yy++)
              mx.adjust(x + y * numberOfFeatures, xx + yy * numberOfFeatures, data[x] * data[y] * data[xx] * data[yy] * weight);
    }
    for (int i = 0; i < numberOfFeatures; i++)
      mx.adjust(i, i, regulationCoeffiecent);
    for (int i = 0; i < numberOfFeatures; i++)
      linear.set(i, -linear.get(i));

    Vec result = solveLinearEquationUsingLQ(mx, linear);
    return new PolynomialExponentRegion(features, mask, result.toArray(), distCoeffiecent);
  }

}
