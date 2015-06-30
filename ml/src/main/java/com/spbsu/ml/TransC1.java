package com.spbsu.ml;

import com.spbsu.commons.func.Computable;
import com.spbsu.commons.math.vectors.Mx;
import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.commons.math.vectors.VecTools;
import com.spbsu.commons.math.vectors.impl.mx.VecBasedMx;
import com.spbsu.commons.math.vectors.impl.vectors.ArrayVec;
import org.jetbrains.annotations.NotNull;
import org.jetbrains.annotations.Nullable;

/**
 * User: solar
 * Date: 21.12.2010
 * Time: 22:07:07
 */
public interface TransC1 extends Trans {
  @NotNull
  Trans gradient();
  Vec gradientTo(Vec x, Vec to);
  Vec gradientRowTo(Vec x, Vec to, int index);
  Vec gradient(Vec x);

  abstract class Stub extends Trans.Stub implements TransC1 {
    @Override @NotNull
    public Trans gradient() {
      return new Trans.Stub() {
        @Override
        public Vec transTo(Vec argument, Vec to) {
          return gradientTo(argument, to);
        }

        @Override
        public final int xdim() {
          return TransC1.Stub.this.xdim();
        }

        @Override
        public final int ydim() {
          return xdim();
        }
      };
    }

    public Vec gradientTo(Vec x, Vec to) {
      final int ydim = ydim();
      if (ydim == 0)
        return gradientRowTo(x, to, 0);
      for (int i = 0; i < ydim; i++)
        gradientRowTo(x, ((Mx) to).row(i), i);
      return to;
    }

    @Override
    public Vec gradient(Vec x) {
      final Vec to;
      if (ydim() == 1)
        to = new ArrayVec(xdim());
      else
        to = new VecBasedMx(xdim(), new ArrayVec(x.dim() * ydim()));
      return gradientTo(x, to);
    }
  }
}
