package com.spbsu.ml.optimization.impl;

import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.commons.math.vectors.VecTools;
import com.spbsu.commons.util.logging.Logger;
import com.spbsu.ml.optimization.ConvexFunction;
import com.spbsu.ml.optimization.ConvexOptimize;

import static com.spbsu.commons.math.vectors.VecTools.copy;

/**
 * User: qde
 * Date: 24.04.13
 * Time: 19:05
 */

public class Nesterov2 implements ConvexOptimize {
  private static Logger LOG = Logger.create(Nesterov2.class);
  private final Vec x0;
  private final double eps;

  public Nesterov2(Vec x0, double eps) {
    this.x0 = x0;
    this.eps = eps;
  }

  public Vec optimize(ConvexFunction func) {
    double m = func.getGlobalConvexParam();
    double lk = func.getGradLipParam();

    Vec y = copy(x0);
    Vec x1 = copy(x0);
    Vec x2 = copy(x0);
    Vec currentGrad;

    double a1 = 0.5;
    double a2, beta;
    final double q = m / lk;

    int iter = 0;

    double distance = 1;
    while (distance > eps) {

      //f'(y[k])
      currentGrad = func.gradient().trans(y);
      //x[k+1] = y[k] - 1/L * f'(y[k])
      for (int i = 0; i < x0.dim(); i++) {
        x2.set(i, y.get(i) - currentGrad.get(i) / lk);
      }

      //find 0<a[k+1]<1 : "a[k+1]^2 = (1 - a[k+1])*a[k]^2 + q*a[k+1]"
      double root1 = 0.5 * (q - a1*a1 - Math.sqrt(a1*a1 * (a1*a1 - 2*q + 4) + q*q));
      double root2 = 0.5 * (q - a1*a1 + Math.sqrt(a1*a1 * (a1*a1 - 2*q + 4) + q*q));

      if (root1 > 0 && root1 < 1)
        a2 = root1;
      else if (root2 > 0 && root2 < 1)
        a2 = root2;
      else
        throw new IllegalArgumentException("Roots are not in the interval, something was wrong at iter#" + iter);

      beta = a1 * (1 - a1) / (a1*a1 + a2);

      //y[k+1] = x[k+1] + beta * (x[k+1] - x[k])
      for (int i = 0; i < x0.dim(); i++) {
        y.set(i, x2.get(i) * (1 + beta) - beta * x1.get(i));
      }

      distance = VecTools.norm(currentGrad) / m;//VecTools.norm(func.gradient(x2)) / m;

      a1 = a2;
      x1 = copy(x2);
      iter++;
    }

    LOG.message("N2 iterations = " + iter);
    return x2;
  }
}
