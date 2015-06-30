package com.spbsu.ml;

import com.spbsu.commons.func.Computable;
import com.spbsu.commons.math.vectors.Mx;
import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.commons.math.vectors.VecTools;
import com.spbsu.commons.math.vectors.impl.vectors.ArrayVec;
import com.spbsu.commons.math.vectors.impl.mx.VecBasedMx;
import org.jetbrains.annotations.Nullable;

/**
 * User: solar
 * Date: 21.12.2010
 * Time: 22:07:07
 */
public interface Trans extends Computable<Vec,Vec> {
  int xdim();
  int ydim();
  @Nullable
  Trans gradient();

  Vec trans(Vec x);
  Vec transTo(Vec x, Vec to);

  Mx transAll(Mx x);

  abstract class Stub implements Trans {
    @Override
    public Trans gradient() {
      return null;
    }

    @Override
    public Vec compute(final Vec argument) {
      return trans(argument);
    }

    @Override
    public Vec transTo(final Vec argument, Vec to) {
      final Vec trans = trans(argument);
      VecTools.assign(to, trans);
      return to;
    }

    public Vec trans(final Vec arg) {
      final Vec result = new ArrayVec(ydim());
      return transTo(arg, result);
    }

    @Override
    public Mx transAll(final Mx ds) {
      final Mx result = new VecBasedMx(ydim(), new ArrayVec(ds.rows() * ydim()));
      for (int i = 0; i < ds.rows(); i++) {
        transTo(ds.row(i), result.row(i));
      }
      return result;
    }
  }
}
