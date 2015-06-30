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
public interface FuncC1 extends Func, TransC1 {
  abstract class Stub extends Func.Stub implements FuncC1 {
    public Vec gradientRowTo(Vec x, Vec to, int index) {
      return gradientTo(x, to);
    }

    public Vec gradientTo(Vec x, Vec to){
      final Vec trans = gradient(x);
      VecTools.assign(to, trans);
      return to;
    }

    public Vec gradient(Vec x) {
      final Vec result = new ArrayVec(x.dim());
      gradientTo(x, result);
      return result;
    }


    @Override
    @NotNull
    public Trans gradient() {
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
          return gradientTo(x, to);
        }
      };
    }
  }
}
