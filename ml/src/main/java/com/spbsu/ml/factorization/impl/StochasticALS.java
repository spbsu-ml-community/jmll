package com.spbsu.ml.factorization.impl;

import com.spbsu.commons.func.impl.WeakListenerHolderImpl;
import com.spbsu.commons.math.MathTools;
import com.spbsu.commons.math.vectors.Mx;
import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.commons.math.vectors.impl.vectors.ArrayVec;
import com.spbsu.commons.random.FastRandom;
import com.spbsu.commons.util.Pair;
import com.spbsu.ml.factorization.Factorization;

import java.util.logging.Level;
import java.util.logging.Logger;

import static com.spbsu.commons.math.vectors.VecTools.*;

/**
 * Experts League
 * Created by solar on 04.05.17.
 */

public class StochasticALS extends WeakListenerHolderImpl<Pair<Vec, Vec>> implements Factorization {
  private static final Logger log = Logger.getLogger(StochasticALS.class.getName());
  private final FastRandom rng;
  private final double gamma;

  public StochasticALS(FastRandom rng, double gamma) {
    this.rng = rng;
    this.gamma = gamma;
  }

  @Override
  public Pair<Vec, Vec> factorize(final Mx X) {
    final int m = X.rows();
    final int n = X.columns();
    if (m < n * 10)
      log.log(Level.WARNING, "This algorithm is intended to be used for matrices with rows >> columns");

    final Vec v = new ArrayVec(n);
    fillGaussian(v, rng);
    scale(v, 1./norm(v));
    final double gamma = this.gamma / (2 + X.rows());
    int k = 0;
    { // iterations
      double a;
      do {
        k++;
        final int i = rng.nextInt(m);
        final Vec row = X.row(i);
        final double u_hat = multiply(row, v);
        a = 0;
        for (int j = 0; j < n; j++) {
          final double val = gamma * (u_hat * (v.get(j) * u_hat - row.get(j))) / Math.log(1 + k);
          v.adjust(j, -val);
          if (a < Math.abs(val))
            a = Math.abs(val);
        }
        scale(v, 1 / norm(v));
      }
      while (a > 0.001 * gamma);
    }

    scale(v, 1 / norm(v));
    final Vec u = new ArrayVec(m);
    for (int i = 0; i < m; i++) {
      final Vec row = X.row(i);
      u.set(i, multiply(row, v));
    }
    return Pair.create(u, v);
  }
}
