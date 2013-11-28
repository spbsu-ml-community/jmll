package com.spbsu.ml.func;

import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.ml.VecFunc;
import com.spbsu.ml.VecFuncStub;
import org.jetbrains.annotations.Nullable;

/**
 * User: solar
 * Date: 21.12.2010
 * Time: 22:07:07
 */
public abstract class VecTransform extends VecFuncStub {
  public abstract Vec vvalue(Vec x);

  @Override
  public double value(Vec x, int direction) {
    return vvalue(x).get(direction);
  }

  @Override
  public @Nullable
  VecFunc gradient(int direction) {
    return null;
  }

  @Override
  public int ydim() {
    return xdim();
  }
}
