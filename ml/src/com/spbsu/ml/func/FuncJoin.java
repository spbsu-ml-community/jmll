package com.spbsu.ml.func;

import com.spbsu.commons.func.Evaluator;
import com.spbsu.commons.math.vectors.Mx;
import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.commons.math.vectors.VecTools;
import com.spbsu.commons.math.vectors.impl.ArrayVec;
import com.spbsu.commons.math.vectors.impl.VecBasedMx;
import com.spbsu.commons.util.ArrayTools;
import com.spbsu.ml.Func;
import com.spbsu.ml.Trans;
import org.jetbrains.annotations.Nullable;
import sun.reflect.generics.reflectiveObjects.NotImplementedException;

import java.util.List;

/**
 * User: solar
 * Date: 21.12.2010
 * Time: 22:07:07
 */
public class FuncJoin extends Trans.Stub {
  public final Func[] dirs;

  public FuncJoin(Func[] dirs) {
    this.dirs = dirs;
  }

  public FuncJoin(List<Func> models) {
    this.dirs = models.toArray(new Func[models.size()]);
  }

  @Override
  public int ydim() {
    return dirs.length;
  }

  @Override
  public int xdim() {
    return dirs[ArrayTools.max(dirs, new Evaluator<Trans>() {
      @Override
      public double value(Trans trans) {
        return trans.xdim();
      }
    })].xdim();
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
        return FuncJoin.this.xdim();
      }

      @Override
      public int ydim() {
        return xdim() * FuncJoin.this.ydim();
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
    return new ArrayVec(ArrayTools.score(dirs, new Evaluator<Func>() {
      @Override
      public double value(Func func) {
        return func.value(x);
      }
    }));
  }
}
