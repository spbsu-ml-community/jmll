package com.spbsu.exp.multiclass.spoc.full.mx.bruteforce;

import com.spbsu.commons.math.vectors.Mx;
import com.spbsu.commons.math.vectors.VecTools;
import com.spbsu.commons.math.vectors.impl.mx.VecBasedMx;
import com.spbsu.ml.methods.multiclass.spoc.AbstractCodingMatrixLearning;

/**
 * User: qdeee
 * Date: 20.08.14
 */
public class FullMatrixLearning extends AbstractCodingMatrixLearning {
  public FullMatrixLearning(final int k) {
    super(k, 0, 0, 0, 0);
  }

  @Override
  protected Mx findMatrixB(final Mx S) {
    final Mx result = new VecBasedMx(k, (int) Math.pow(2, k - 1) - 1);
    for (int i = 0; i < k; i++) {
      for (int j = 0; j < result.columns(); j++) {
        final int bit = (j >>> i) & 1;
        result.set(i, j, 2 * bit - 1); //map (-1, 1) -> (0, 1)
      }
    }
    VecTools.fill(result.row(k - 1), 1);
    return result;
  }
}
