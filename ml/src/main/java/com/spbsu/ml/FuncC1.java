package com.spbsu.ml;

import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.commons.math.vectors.VecTools;
import com.spbsu.commons.math.vectors.impl.vectors.ArrayVec;
import org.jetbrains.annotations.NotNull;

/**
 * User: solar
 * Date: 21.12.2010
 * Time: 22:07:07
 */
public interface FuncC1 extends Func {
  Vec gradient(Vec x);
  Vec gradientTo(Vec x, Vec to);

  abstract class Stub extends Func.Stub implements FuncC1 {
    public Vec gradient(Vec x) {
      final Vec result = new ArrayVec(x.dim());
      return gradientTo(x, result);
    }

    public Vec gradientTo(Vec x, Vec to){
      final Vec trans = gradient(x);
      VecTools.assign(to, trans);
      return to;
    }

    @Override
    @NotNull
    public final Trans gradient() {
      return new Trans.Stub() {
        @Override
        public int xdim() {
          return dim();
        }

        @Override
        public int ydim() {
          return dim();
        }

        @Override
        public Vec transTo(final Vec x, final Vec to) {
          return FuncC1.Stub.this.gradientTo(x, to);
        }
      };
    }
  }
}
