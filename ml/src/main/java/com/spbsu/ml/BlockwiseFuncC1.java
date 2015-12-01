package com.spbsu.ml;

import com.spbsu.commons.math.FuncC1;
import com.spbsu.commons.math.vectors.Mx;
import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.commons.math.vectors.impl.mx.VecBasedMx;

import static com.spbsu.commons.math.vectors.VecTools.copy;

/**
 * User: solar
 * Date: 21.12.2010
 * Time: 22:07:07
 */
public interface BlockwiseFuncC1 extends FuncC1 {
  void gradient(Vec pointBlock, Vec result, int index);

  double value(Vec pointBlock, int index);
  double transformResultValue(double value);

  int blockSize();

  abstract class Stub extends FuncC1.Stub implements BlockwiseFuncC1 {
    public final Mx gradient(final Mx x) {
      final Mx result = copy(x);
      for (int i = 0; i < x.rows(); i++) {
        gradient(x.row(i), result.row(i), i);
      }
      return result;
    }

    @Override
    public final Mx gradient(final Vec x) {
      return gradient(x instanceof Mx ? (Mx)x : new VecBasedMx(blockSize(), x));
    }

    protected double value(final Mx blocks) {
      double result = 0.0;
      for (int i = 0; i < blocks.rows(); i ++) {
        result += value(blocks.row(i), i);
      }
      return result;
    }

    @Override
    public final double value(final Vec x) {
      final double value = value(x instanceof Mx ? (Mx) x : new VecBasedMx(blockSize(), x));
      return transformResultValue(value);
    }
  }
}
