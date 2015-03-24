package com.spbsu.ml;

import com.spbsu.commons.math.vectors.Vec;
import org.jetbrains.annotations.NotNull;

/**
 * User: solar
 * Date: 21.12.2010
 * Time: 22:07:07
 */
public interface FuncC1 extends Func {
  Vec gradient(Vec x);

  abstract class Stub extends Func.Stub implements FuncC1 {
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
        public Vec trans(final Vec x) {
          return FuncC1.Stub.this.gradient(x);
        }
      };
    }
  }
}
