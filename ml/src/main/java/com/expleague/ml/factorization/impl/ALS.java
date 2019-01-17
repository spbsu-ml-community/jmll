package com.expleague.ml.factorization.impl;

import com.expleague.commons.math.MathTools;
import com.expleague.commons.math.vectors.impl.vectors.ArrayVec;
import com.expleague.commons.func.impl.WeakListenerHolderImpl;
import com.expleague.commons.math.vectors.Mx;
import com.expleague.commons.math.vectors.Vec;
import com.expleague.commons.math.vectors.VecTools;
import com.expleague.commons.util.Pair;
import com.expleague.ml.factorization.Factorization;

/**
 * User: qdeee
 * Date: 09.09.13
 */
public class ALS extends WeakListenerHolderImpl<Pair<Vec, Vec>> implements Factorization {
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
      for (int i = 0; i < m; i++) {
        double sum = VecTools.multiply(v, X.row(i));
        u.set(i, sum / (1 + lambda));
      }
      double squareNormU = VecTools.sum2(u);
      for (int j = 0; j < n; j++) {
        double sum = VecTools.multiply(u, X.col(j));
        v.set(j, sum / (squareNormU + lambda));
      }
      VecTools.scale(v, 1/VecTools.norm(v));
      invoke(Pair.create(u, v));
    }

    return Pair.create(u, v);
  }
}
