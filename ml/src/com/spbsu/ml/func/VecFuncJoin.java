package com.spbsu.ml.func;

import com.spbsu.commons.func.Computable;
import com.spbsu.commons.func.Evaluator;
import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.commons.util.ArrayTools;
import com.spbsu.ml.Func;
import com.spbsu.ml.VecFunc;
import com.spbsu.ml.VecFuncStub;
import org.jetbrains.annotations.Nullable;

import java.util.List;

/**
 * User: solar
 * Date: 21.12.2010
 * Time: 22:07:07
 */
public class VecFuncJoin extends VecFuncStub {
  private final Func[] dirs;

  public VecFuncJoin(int count, Computable<Integer, Func> dirs) {
    this.dirs = new Func[count];
    for (int i = 0; i < count; i++) {
      this.dirs[i] = dirs.compute(i);
    }
  }

  public VecFuncJoin(Func[] dirs) {
    this.dirs = dirs;
  }

  public VecFuncJoin(List<Func> models) {
    this.dirs = models.toArray(new Func[models.size()]);
  }

  @Override
  public Func direction(int dir) {
    return dirs[dir];
  }

  @Override
  public double value(Vec x, int direction) {
    return dirs[direction].value(x);
  }

  @Override
  public @Nullable
  VecFunc gradient(int direction) {
    return dirs[direction].gradient();
  }

  @Override
  public int ydim() {
    return dirs.length;
  }

  @Override
  public int xdim() {
    return dirs[ArrayTools.max(dirs, new Evaluator<Func>() {
      @Override
      public double value(Func func) {
        return func.xdim();
      }
    })].xdim();
  }
}
