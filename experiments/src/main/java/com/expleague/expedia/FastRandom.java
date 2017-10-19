package com.expleague.expedia;

import java.util.SplittableRandom;

public class FastRandom {
  private final State state = new State();
  private final SplittableRandom random = new SplittableRandom();

  public double nextStandardExponential() {
    return -Math.log(1.0 - nextDouble());
  }

  public double nextDouble() {
    return random.nextDouble();
  }

  public double nextGauss() {
    if (state.hasGauss == 1) {
      double tmp = state.gauss;
      state.gauss = 0;
      state.hasGauss = 0;
      return tmp;
    } else {
      double f, x1, x2, r2;

      do {
        x1 = 2.0 * nextDouble() - 1.0;
        x2 = 2.0 * nextDouble() - 1.0;
        r2 = x1 * x1 + x2 * x2;
      } while (r2 >= 1.0 || r2 == 0.0);

      /* Box-Muller transform */
      f = Math.sqrt(-2.0 * Math.log(r2) / r2);
      /* Keep for next call */
      state.gauss = f * x1;
      state.hasGauss = 1;
      return f * x2;
    }
  }

  public double nextStandardGamma(final double shape) {
    double b, c;
    double U, V, X, Y;

    if (shape == 1.0) {
      return nextStandardExponential();
    } else if (shape < 1.0) {
      for (; ; ) {
        U = nextDouble();
        V = nextStandardExponential();

        if (U <= 1.0 - shape) {
          X = Math.pow(U, 1. / shape);

          if (X <= V) {
            return X;
          }
        } else {
          Y = -Math.log((1 - U) / shape);
          X = Math.pow(1.0 - shape + shape * Y, 1. / shape);

          if (X <= (V + Y)) {
            return X;
          }
        }
      }
    } else {
      b = shape - 1. / 3.;
      c = 1. / Math.sqrt(9 * b);
      for (; ; ) {
        do {
          X = nextGauss();
          V = 1.0 + c * X;
        } while (V <= 0.0);

        V = V * V * V;
        U = nextDouble();
        if (U < 1.0 - 0.0331 * (X * X) * (X * X)) {
          return (b * V);
        }

        if (Math.log(U) < 0.5 * X * X + b * (1. - V + Math.log(V))) {
          return (b * V);
        }
      }
    }
  }

  public double[] nextDirichlet(final int[] params, double[] out) {
    if (out == null) {
      out = new double[params.length];
    }

    double total = 0;
    double gamma;

    for (int i = 0; i < params.length; ++i) {
      gamma = nextStandardGamma(params[i]);
      out[i] = gamma;
      total += gamma;
    }

    double invTotal = 1.0 / total;

    for (int i = 0; i < params.length; ++i) {
      out[i] *= invTotal;
    }

    return out;
  }

  public double nextGamma(final double shape, final double scale) {
    return scale * nextStandardGamma(shape);
  }

  private static class State {
    private int hasGauss = 0;
    private double gauss;
  }
}
