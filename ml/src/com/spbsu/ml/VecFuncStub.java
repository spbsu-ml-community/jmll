package com.spbsu.ml;

import com.spbsu.commons.math.vectors.Mx;
import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.commons.math.vectors.VecTools;
import com.spbsu.commons.math.vectors.impl.ArrayVec;
import com.spbsu.commons.math.vectors.impl.VecBasedMx;
import org.jetbrains.annotations.Nullable;

/**
 * User: solar
 * Date: 21.11.13
 * Time: 10:30
 */
public abstract class VecFuncStub implements VecFunc {
  @Override
  public Func direction(final int direction) {
    return new FuncStub() {
      @Override
      public int xdim() {
        return xdim();
      }

      @Override
      public double value(Vec x) {
        return VecFuncStub.this.value(x, direction);
      }
    };
  }

  @Nullable
  @Override
  public VecFunc gradient(int direction) {
    return null;
  }

  public Mx gradient(Vec x) {
    final Mx result = new VecBasedMx(new VecBasedMx(ydim(), x.dim()));
    for (int i = 0; i < ydim(); i++) {
      final Vec dimGrad = gradient(i).vvalue(x);
      if (dimGrad == null)
        throw new RuntimeException("Gradient is not defined for function");
      VecTools.assign(result.row(i), dimGrad);
    }
    return result;
  }

  @Override
  public Vec vvalue(Vec x) {
    Vec result = new ArrayVec(ydim());
    for (int i = 0; i < ydim(); i++) {
      result.set(i, value(x, i));
    }
    return result;
  }

  @Override
  public Mx value(Mx ds) {
    Mx result = new VecBasedMx(ds.rows(), new ArrayVec(ds.rows() * ydim()));
    for (int i = 0; i < ds.rows(); i++) {
      final Vec row = ds.row(i);
      for (int c = 0; c < ydim(); c++) {
        result.set(c, i, value(row, c));
      }
    }
    return new VecBasedMx(ydim(), result);
  }

  public Func[] directions() {
    final Func[] result = new Func[ydim()];
    for (int i = 0; i < result.length; i++) {
      result[i] = direction(i);
    }
    return result;
  }
}
