package com.spbsu.ml.optimization.impl;

import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.commons.math.vectors.VecTools;
import com.spbsu.commons.math.vectors.impl.vectors.ArrayVec;
import com.spbsu.commons.util.logging.Logger;
import com.spbsu.ml.optimization.FuncConvex;
import com.spbsu.ml.optimization.Optimize;

/**
 * User: qde
 * Date: 06.09.13
 * Time: 16:45
 */
public class Nesterov1 implements Optimize<FuncConvex> {
  private static final Logger LOG = Logger.create(Nesterov1.class);

  private final Vec x0;
  private final double eps;

  public Nesterov1(final Vec x0, final double eps) {
    this.x0 = x0;
    this.eps = eps;
  }

  @Override
  public Vec optimize(final FuncConvex func) {
    final int n = func.xdim();
    double alpha;
    final double L = func.getGradLipParam();
    final double m = func.getGlobalConvexParam();
    double gamma1 = L;
    double gamma2;

    Vec x1 = VecTools.copy(x0);
    final Vec x2 = new ArrayVec(n);
    Vec v1 = VecTools.copy(x0);
    final Vec v2 = new ArrayVec(n);
    final Vec y = new ArrayVec(n);
    Vec grad;

    int iter = 0;
    do {
//            compute alpha
      {
        final double D = m * m - 2 * gamma1 * m + 4 * gamma1 * L + gamma1 * gamma1;
        final double root1 = ((m - gamma1) - Math.sqrt(D)) / (2 * L);
        final double root2 = ((m - gamma1) + Math.sqrt(D)) / (2 * L);
        if (0 <= root2 && root2 <= 1)
          alpha = root2;
        else {
          if (0 <= root1 && root1 <= 1)
            alpha = root1;
          else
            throw new IllegalArgumentException("Roots are not in the interval, something was wrong on iter#" + iter);
        }
      }
      gamma2 = (1 - alpha) * gamma1 + alpha * m;

      for (int i = 0; i < n; i++) {
        y.set(i, (alpha * gamma1 * v1.get(i) + gamma2 * x1.get(i)) / (gamma1 + alpha * m));
      }

      grad = func.gradient().trans(y);
      for (int i = 0; i < n; i++) {
        x2.set(i, y.get(i) - (1.0 / L) * grad.get(i));
      }

      for (int i = 0; i < n; i++) {
        v2.set(i, (1.0 / gamma2) * ((1 - alpha) * gamma1 * v1.get(i) + alpha * m * y.get(i) - alpha * grad.get(i)));
      }

      x1 = VecTools.copy(x2);
      v1 = VecTools.copy(v2);
      gamma1 = gamma2;
      iter++;

    } while ((VecTools.norm(grad) / m) > eps);

    //LOG.message("N1 iterations = " + iter);
    return x2;
  }
}
