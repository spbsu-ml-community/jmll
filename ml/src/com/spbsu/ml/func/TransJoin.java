package com.spbsu.ml.func;

import com.spbsu.commons.func.Evaluator;
import com.spbsu.commons.math.vectors.Mx;
import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.commons.math.vectors.VecTools;
import com.spbsu.commons.math.vectors.impl.ArrayVec;
import com.spbsu.commons.math.vectors.impl.VecBasedMx;
import com.spbsu.commons.util.ArrayTools;
import com.spbsu.ml.Trans;
import org.jetbrains.annotations.Nullable;
import sun.reflect.generics.reflectiveObjects.NotImplementedException;

import java.util.List;

/**
 * User: solar
 * Date: 21.12.2010
 * Time: 22:07:07
 */
public class TransJoin extends Trans.Stub {
  public final Trans[] dirs;
  private int xdim;
  private int ydim;

  public TransJoin(Trans[] dirs) {
    this.dirs = dirs;
    xdim = dirs[ArrayTools.max(dirs, new Evaluator<Trans>() {
      @Override
      public double value(Trans trans) {
        return trans.xdim();
      }
    })].xdim();
    ydim = dirs.length * dirs[ArrayTools.max(dirs, new Evaluator<Trans>() {
      @Override
      public double value(Trans trans) {
        return trans.ydim();
      }
    })].ydim();
  }

  public TransJoin(List<Trans> models) {
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
      public Vec trans(Vec x) {
        Mx result = new VecBasedMx(xdim(), new ArrayVec(ydim()));
        for (int i = 0; i < dirs.length; i++) {
          VecTools.assign(result.row(i), gradients[i].trans(x));
        }
        return result;
      }
    };
  }

  @Override
  public Vec trans(final Vec x) {
    Mx result = new VecBasedMx(ydim / dirs.length, new ArrayVec(ydim));
    for (int c = 0; c < dirs.length; c++) {
      VecTools.assign(result.row(c), dirs[c].trans(x));
    }
    return result;
  }
}
