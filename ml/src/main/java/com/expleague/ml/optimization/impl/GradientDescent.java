package com.expleague.ml.optimization.impl;

import com.expleague.commons.math.FuncC1;
import com.expleague.commons.math.vectors.Vec;
import com.expleague.commons.math.vectors.VecTools;
import com.expleague.commons.math.vectors.impl.vectors.ArrayVec;
import com.expleague.ml.func.RegularizerFunc;
import com.expleague.ml.optimization.FuncConvex;
import com.expleague.ml.optimization.Optimize;
import com.expleague.ml.optimization.PDQuadraticFunction;

import java.util.concurrent.TimeUnit;

/**
 * User: qde
 * Date: 25.04.13
 * Time: 23:41
 */
public class GradientDescent implements Optimize<FuncC1> {
  //  private static final Logger LOG = Logger.create(GradientDescent.class);
  private final Vec x0;
  private final double eps;

  public GradientDescent(final Vec x0, final double eps) {
    this.x0 = x0;
    this.eps = eps;
  }

  @Override
  public Vec optimize(final FuncC1 func, RegularizerFunc reg, Vec x0) {
    long time = System.nanoTime();
    final boolean isQuadraticFunc = func instanceof PDQuadraticFunction;

    final double constStep = 1.0 / VecTools.max(func.L(x0));

    Vec x1 = VecTools.copy(x0);
    final Vec x2 = new ArrayVec(x0.dim());
    Vec grad = func.gradient().trans(x0);

    int iter = 0;

    double distance = 1;
    while (distance > eps && iter < 1e5) {
      final double step = isQuadraticFunc? getStepSizeForQuadraticFunc((FuncConvex)func, grad) : constStep;
      for (int i = 0; i < x2.dim(); i++) {
        x2.set(i, x1.get(i) - grad.get(i) * step);
      }
      x1 = VecTools.copy(reg.project(x2));
      grad = func.gradient().trans(x1);
      distance = VecTools.norm(grad);
      if (iter % 1000 == 0) {
        final long newTime = System.nanoTime();
        //System.out.println("Gradient is " + grad.toString());
        System.out.println("Probability is " + func.value(x2) + " distance " + distance + " time=" + TimeUnit.NANOSECONDS.toMillis(newTime - time));
        time = newTime;
      }
      iter++;
    }
    System.out.println("Iterations made " + iter);
    System.out.println("Gradient distance is " + distance);
    System.out.println("Probability is " + func.value(x2));
    //        LOG.message("GDM iterations = " + iter + "\n\n");
    return x2;
  }

  @Override
  public Vec optimize(FuncC1 func) {
    return optimize(func, x0);
  }

  private double getStepSizeForQuadraticFunc(final FuncConvex func, final Vec grad) {
    return VecTools.multiply(grad, grad) / ((PDQuadraticFunction) func).getQuadrPartValue(grad);
  }
}
