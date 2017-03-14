package com.spbsu.ml;

import com.spbsu.commons.math.Trans;
import com.spbsu.commons.math.vectors.Mx;
import com.spbsu.commons.math.vectors.MxTools;
import com.spbsu.commons.math.vectors.Vec;
import org.jetbrains.annotations.Nullable;
import sun.reflect.generics.reflectiveObjects.NotImplementedException;

/**
 * User: solar
 * Date: 21.12.2010
 * Time: 22:07:07
 */
public class CompositeTrans<F extends Trans, G extends Trans> extends Trans.Stub {
  public final F f;
  public final G g;

  public CompositeTrans(final F f, final G g) {
    this.f = f;
    this.g = g;
  }

  @Override
  public int xdim() {
    return g.xdim();
  }

  @Override
  public int ydim() {
    return f.ydim();
  }

  @Override
  public Vec trans(final Vec x) {
    return f.trans(g.trans(x));
  }

  @Nullable
  @Override
  public Trans gradient() {
    return new Stub() {
      @Override
      public int xdim() {
        return g.xdim();
      }

      @Override
      public int ydim() {
        return f.ydim() * f.xdim();
      }

      @Nullable
      @Override
      public Trans gradient() {
        throw new NotImplementedException();
      }

      @Override
      public Vec trans(final Vec x) {
        return MxTools.multiply((Mx) f.gradient().trans(g.trans(x)), (Mx) g.gradient().trans(x));
      }
    };
  }
}
