package com.spbsu.ml.optimization;

import com.spbsu.commons.math.vectors.Mx;
import com.spbsu.commons.math.vectors.MxTools;
import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.commons.math.vectors.VecTools;
import com.spbsu.commons.math.vectors.impl.mx.VecBasedMx;
import com.spbsu.commons.math.vectors.impl.vectors.ArrayVec;


/**
 * User: qde
 * Date: 24.04.13
 * Time: 19:03
 * Description: positive-definite quadratic function
 */

public class PDQuadraticFunction extends FuncConvex.Stub {
  private final Mx mxA;
  private final Vec w;
  private final double w0;

  private final double m;
  private final double l;

  public PDQuadraticFunction(final Mx mxA, final Vec w, final double w0) {
    final double[] params = getConvAndLipConstants(mxA, w);

    this.mxA = mxA;
    this.w = w;
    this.w0 = w0;

    this.m = params[0];
    this.l = params[1];
  }

  // result[0] = m (convex param),
  // result[1] = l (lipshitz const);
  private static double[] getConvAndLipConstants(final Mx mxA, final Vec w) {
    final Mx q = new VecBasedMx(mxA.rows(), mxA.columns());
    final Mx sigma = new VecBasedMx(mxA.rows(), mxA.columns());
    MxTools.eigenDecomposition(mxA, q, sigma);

    double minEigenValue = sigma.get(0, 0);
    double maxEigenValue = sigma.get(0, 0);
    for (int i = 1; i < sigma.rows(); i++) {
      if (sigma.get(i, i) < minEigenValue)
        minEigenValue = sigma.get(i, i);
      if (sigma.get(i, i) > maxEigenValue)
        maxEigenValue = sigma.get(i, i);
    }
    return new double[]{minEigenValue, maxEigenValue};
  }

  @Override
  public int dim() {
    return w.dim();
  }

  @Override
  public double value(final Vec x) {
    return VecTools.multiply(MxTools.multiply(mxA, x), x) + VecTools.multiply(w, x) + w0;
  }

  public double getQuadrPartValue(final Vec x) {
    return VecTools.multiply(MxTools.multiply(mxA, x), x);
  }

  @Override
  public Vec gradient(final Vec x) {
    return VecTools.append(MxTools.multiply(mxA, x), w);
  }

  @Override
  public double getGradLipParam() {
    return l;
  }

  @Override
  public double getGlobalConvexParam() {
    return m;
  }

  public Vec getExactExtremum() {
    final Vec b = VecTools.copy(w);
    VecTools.scale(b, -1.0);
    final Mx l = MxTools.choleskyDecomposition(mxA);
    final Mx inverse = MxTools.inverseLTriangle(l);
    final Vec x = MxTools.multiply(MxTools.multiply(MxTools.transpose(inverse), inverse), b);

    //stupid cast (VecBasedMx -> ArrayVec) for solution out
    final Vec result = new ArrayVec(b.dim());
    VecTools.assign(result, x);
    return result;
  }
}
