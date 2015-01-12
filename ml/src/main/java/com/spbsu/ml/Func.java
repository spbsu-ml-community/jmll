package com.spbsu.ml;

import com.spbsu.commons.math.vectors.Mx;
import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.commons.math.vectors.impl.vectors.ArrayVec;
import com.spbsu.commons.math.vectors.impl.mx.VecBasedMx;

/**
 * User: solar
 * Date: 21.12.2010
 * Time: 22:07:07
 */
public interface Func extends Trans {
  double value(Vec x);
  int dim();

  abstract class Stub extends Trans.Stub implements Func {
    public final int ydim() {
      return 1;
    }

    public final int xdim() {
      return dim();
    }

    public final Vec trans(final Vec x) {
      return new ArrayVec(new double[]{value(x)});
    }

    public Mx transAll(final Mx ds) {
      final Mx result = new VecBasedMx(1, new ArrayVec(ds.rows()));
      for (int i = 0; i < ds.rows(); i++) {
        result.set(i, value(ds.row(i)));
      }
      return result;
    }
  }
}
