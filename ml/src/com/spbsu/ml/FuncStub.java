package com.spbsu.ml;

import com.spbsu.commons.math.vectors.Mx;
import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.commons.math.vectors.impl.ArrayVec;
import com.spbsu.ml.func.VecTransform;
import org.jetbrains.annotations.Nullable;

/**
 * User: solar
 * Date: 21.12.2010
 * Time: 22:07:07
 */
public abstract class FuncStub implements Func {
  @Nullable
  @Override
  public VecFunc gradient() {
    return new VecTransform() {
      @Override
      public Vec vvalue(Vec x) {
        return FuncStub.this.gradient(x);
      }

      @Override
      public int xdim() {
        return FuncStub.this.xdim();
      }
    };
  }

  protected Vec gradient(Vec x) {
    return null;
  }

  public Vec value(Mx ds) {
    Vec result = new ArrayVec(ds.rows());
    for (int i = 0; i < ds.rows(); i++) {
      result.set(i, value(ds.row(i)));
    }
    return result;
  }
}
