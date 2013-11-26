package com.spbsu.ml.optimization.impl;

import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.commons.math.vectors.VecTools;
import com.spbsu.commons.math.vectors.impl.ArrayVec;
import com.spbsu.commons.util.logging.Logger;
import com.spbsu.ml.optimization.ConvexFunction;
import com.spbsu.ml.optimization.ConvexOptimize;

/**
 * User: qde
 * Date: 06.09.13
 * Time: 16:45
 */
public class Nesterov1 implements ConvexOptimize {
  private static Logger LOG = Logger.create(Nesterov1.class);

  private Vec x0;

  public Nesterov1(Vec x0) {
    this.x0 = x0;
  }

  @Override
  public Vec optimize(ConvexFunction func, double eps) {
    final int n = func.xdim();
    double alpha;
    double L = func.getGradLipParam();
    double m = func.getGlobalConvexParam();
    double gamma1 = L;
    double gamma2;

    Vec x1 = VecTools.copy(x0);
    Vec x2 = new ArrayVec(n);
    Vec v1 = VecTools.copy(x0);
    Vec v2 = new ArrayVec(n);
    Vec y = new ArrayVec(n);
    Vec grad;

    int iter = 0;
    do {
//            compute alpha
      {
        double D = m * m - 2 * gamma1 * m + 4 * gamma1 * L + gamma1 * gamma1;
        double root1 = ((m - gamma1) - Math.sqrt(D)) / (2 * L);
        double root2 = ((m - gamma1) + Math.sqrt(D)) / (2 * L);
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

      grad = func.gradient().value(y);
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
