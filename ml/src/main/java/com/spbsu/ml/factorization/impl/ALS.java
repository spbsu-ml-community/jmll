package com.spbsu.ml.factorization.impl;

import com.spbsu.commons.func.impl.WeakListenerHolderImpl;
import com.spbsu.commons.math.vectors.Mx;
import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.commons.math.vectors.VecTools;
import com.spbsu.commons.math.vectors.impl.vectors.ArrayVec;
import com.spbsu.commons.math.vectors.impl.vectors.ConcatVec;
import com.spbsu.commons.util.Pair;
import com.spbsu.ml.factorization.OuterFactorization;

/**
 * User: qdeee
 * Date: 09.09.13
 */
public class ALS extends WeakListenerHolderImpl<Vec> implements OuterFactorization {
    private final int iterCount;
    private Vec x0;

  public ALS(final int iterCount) {
    this.iterCount = iterCount;
    this.x0 = null;
  }
  public ALS(Vec x0, int iterCount) {
    this.x0 = x0;
    this.iterCount = iterCount;
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
        u.set(i, sum / squareNormV);
      }
      double squareNormU = Math.pow(VecTools.norm(u), 2);
      for (int j = 0; j < n; j++) {
        double sum = 0;
        for (int i = 0; i < m; i++) {
          sum += u.get(i) * X.get(i, j);
        }
        v.set(j, sum / squareNormU);
      }

      invoke(new ConcatVec(u, v));
    }

    return Pair.create(u, v);
  }
}
