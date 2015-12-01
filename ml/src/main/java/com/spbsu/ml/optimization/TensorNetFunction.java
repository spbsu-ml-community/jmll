package com.spbsu.ml.optimization;

import com.spbsu.commons.math.vectors.Mx;
import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.commons.math.vectors.impl.vectors.ArrayVec;
import org.jetbrains.annotations.NotNull;

/**
 * User: qdeee
 * Date: 30.08.13
 */
public class TensorNetFunction extends FuncConvex.Stub {
  private final Mx X;
  private final double c1;
  private final double c2;

  private final double L;
  private final double m;

  public TensorNetFunction(final Mx x, final double c1, final double c2) {
    X = x;
    this.c1 = c1;
    this.c2 = c2;

    double squareXNorm = 0;
    double squareXDetEstimate = 0.0; //assume that det=0  ;//1.0

    final int n = X.rows();
    for (int i = 0; i < n; i++) {
      double rowSumSquare = 0;
      for (int j = 0; j < n; j++) {
        final double squareElem = Math.pow(X.get(i, j), 2);
        rowSumSquare += squareElem;
      }
      squareXNorm += rowSumSquare;
      squareXDetEstimate *= rowSumSquare;
    }

    final double squareUBound = Math.sqrt(squareXNorm) / c2;
    final double squareVBound = Math.sqrt(squareXNorm) / c1;

    L = squareUBound + squareVBound + n * (c1*c1 + c2*c2) + Math.sqrt(Math.pow(squareUBound + n*c2*c2, 2) +
            Math.pow(squareVBound + n*c1*c1, 2) +
            4 * Math.pow(2 * squareXDetEstimate, 1.0 / n));

    m = n * (c1*c1 + c2*c2) - Math.sqrt(Math.pow(squareUBound + n*c2*c2, 2) +
            Math.pow(squareVBound + n*c1*c1, 2) +
            4 * Math.pow(2 * squareXDetEstimate, 1.0 / n));

//        m = getLocalConvexParam(new ArrayVec(2 * X.rows()));
  }

  @NotNull
  @Override
  public double getGlobalConvexParam() {
    return m;
  }

  @Override
  public double getGradLipParam() {
    return L;
  }


  @Override
  public int dim() {
    return X.rows();
  }

  @Override
  public double value(final Vec z) {
    final int n = z.dim() / 2;

    double value = 0.0;
    for (int i = 0; i < n; i++) {
      for (int j = 0; j < n; j++) {
        final double x = X.get(i, j);
        value += Math.pow(x - z.get(i) * z.get(j+n), 2);
        value += Math.pow(x - c1*z.get(i), 2);
        value += Math.pow(x - c2*z.get(j+n), 2);
      }
    }
    return value;
  }

  public Mx getX() {
    return X;
  }

  @Override
  public Vec gradient(final Vec z) {
    final int n = z.dim() / 2;

    final Vec grad = new ArrayVec(z.dim());
    for (int i = 0; i < n; i++) {
      double valU = 0.0;
      final double u_i = z.get(i);
      for (int j = 0; j < n; j++) {
        final double x = X.get(i, j);
        valU += z.get(n+j) * (z.get(n+j) * u_i - x) + c1 * (c1 * u_i - x);
      }
      grad.set(i, 2 * valU);
    }
    for (int j = 0; j < n; j++) {
      double valV = 0.0;
      final double v_j = z.get(n + j);
      for (int i = 0; i < n; i++) {
        final double x = X.get(i, j);
        valV += z.get(i) * (z.get(n + i) * v_j - x) + c2 * (c2 * v_j - x);
      }
      grad.set(n + j, 2 * valV);
    }
    return grad;
  }
}
