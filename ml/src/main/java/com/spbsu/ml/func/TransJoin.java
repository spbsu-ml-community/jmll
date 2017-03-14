package com.spbsu.ml.func;

import com.spbsu.commons.func.Evaluator;
import com.spbsu.commons.math.vectors.Mx;
import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.commons.math.vectors.VecTools;
import com.spbsu.commons.math.vectors.impl.mx.VecBasedMx;
import com.spbsu.commons.math.vectors.impl.vectors.ArrayVec;
import com.spbsu.commons.util.ArrayTools;
import com.spbsu.commons.math.Trans;
import org.jetbrains.annotations.Nullable;
import sun.reflect.generics.reflectiveObjects.NotImplementedException;

import java.util.Arrays;
import java.util.List;

/**
 * User: solar
 * Date: 21.12.2010
 * Time: 22:07:07
 */
public class TransJoin extends Trans.Stub {
  public final Trans[] dirs;
  private final int xdim;
  private final int ydim;

  public TransJoin(final Trans[] dirs) {
    this.dirs = dirs;
    xdim = dirs[ArrayTools.max(dirs, new Evaluator<Trans>() {
      @Override
      public double value(final Trans trans) {
        return trans.xdim();
      }
    })].xdim();
    ydim = dirs.length * dirs[ArrayTools.max(dirs, new Evaluator<Trans>() {
      @Override
      public double value(final Trans trans) {
        return trans.ydim();
      }
    })].ydim();
  }

  public TransJoin(final List<Trans> models) {
    this(models.toArray(new Trans[models.size()]));
  }

  @Override
  public int ydim() {
    return ydim;
  }

  @Override
  public int xdim() {
    return xdim;
  }

  @Nullable
  @Override
  public Trans gradient() {
    final Trans[] gradients = new Trans[ydim()];
    for (int i = 0; i < dirs.length; i++) {
      gradients[i] = dirs[i].gradient();
      if (gradients[i] == null)
        return null;
    }

    return new Stub() {
      @Override
      public int xdim() {
        return TransJoin.this.xdim();
      }

      @Override
      public int ydim() {
        return xdim() * TransJoin.this.ydim();
      }

      @Nullable
      @Override
      public Trans gradient() {
        throw new NotImplementedException();
      }

      @Override
      public Vec trans(final Vec x) {
        final Mx result = new VecBasedMx(xdim(), new ArrayVec(ydim()));
        for (int i = 0; i < dirs.length; i++) {
          VecTools.assign(result.row(i), gradients[i].trans(x));
        }
        return result;
      }
    };
  }

  @Override
  public Vec trans(final Vec x) {
    final Mx result = new VecBasedMx(ydim / dirs.length, new ArrayVec(ydim));
    for (int c = 0; c < dirs.length; c++) {
      VecTools.assign(result.row(c), dirs[c].trans(x));
    }
    return result;
  }

  @Override
  public boolean equals(final Object o) {
    if (this == o) return true;
    if (o == null || getClass() != o.getClass()) return false;

    final TransJoin transJoin = (TransJoin) o;

    if (xdim != transJoin.xdim) return false;
    if (ydim != transJoin.ydim) return false;
    if (!Arrays.equals(dirs, transJoin.dirs)) return false;

    return true;
  }

  @Override
  public int hashCode() {
    int result = Arrays.hashCode(dirs);
    result = 31 * result + xdim;
    result = 31 * result + ydim;
    return result;
  }
}
