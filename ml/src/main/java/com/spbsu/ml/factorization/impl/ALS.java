package com.spbsu.ml.factorization.impl;

import com.spbsu.commons.func.impl.WeakListenerHolderImpl;
import com.spbsu.commons.math.vectors.Mx;
import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.commons.math.vectors.VecTools;
import com.spbsu.commons.math.vectors.impl.vectors.ArrayVec;
import com.spbsu.commons.util.Pair;
import com.spbsu.ml.factorization.OuterFactorization;

/**
 * User: qdeee
 * Date: 09.09.13
 */
public class ALS extends WeakListenerHolderImpl<Pair<Vec, Vec>> implements OuterFactorization {
  private final int iterCount;
  private final double lambda;
  private Vec x0;

  public ALS(final int iterCount, final double lambda, final Vec x0) {
    this.iterCount = iterCount;
    this.lambda = lambda;
    this.x0 = x0;
  }

  public ALS(final int iterCount, final double lambda) {
    this(iterCount, lambda, null);
  }

  public ALS(final int iterCount) {
    this(iterCount, 0.0);
  }

  @Override
  public Pair<Vec, Vec> factorize(final Mx X) {
    final int m = X.rows();
    final int n = X.columns();
    if (x0 == null) {
      x0 = VecTools.fill(new ArrayVec(m + n), 1.0);
    }

    final Vec u = new ArrayVec(m);
    final Vec v = new ArrayVec(n);
    VecTools.assign(u, x0.sub(0, m));
    VecTools.assign(v, x0.sub(m, n));

    int iter = 0;
    while (iter++ < iterCount) {
      double squareNormV = Math.pow(VecTools.norm(v), 2);
      for (int i = 0; i < m; i++) {
        double sum = 0;
        for (int j = 0; j < n; j++) {
          sum += v.get(j) * X.get(i, j);
        }
        u.set(i, sum / (squareNormV + lambda));
      }
      double squareNormU = Math.pow(VecTools.norm(u), 2);
      for (int j = 0; j < n; j++) {
        double sum = 0;
        for (int i = 0; i < m; i++) {
          sum += u.get(i) * X.get(i, j);
        }
        v.set(j, sum / (squareNormU + lambda));
      }

      invoke(Pair.create(u, v));
    }

    return Pair.create(u, v);
  }
}
