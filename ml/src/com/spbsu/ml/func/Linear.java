package com.spbsu.ml.func;

import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.commons.math.vectors.VecTools;
import com.spbsu.commons.math.vectors.impl.ArrayVec;
import com.spbsu.ml.Func;
import com.spbsu.ml.Trans;
import org.jetbrains.annotations.Nullable;

/**
 * User: solar
 * Date: 26.11.12
 * Time: 15:56
 */
public class Linear extends Func.Stub {
  public final Vec weights;

  public Linear(double[] weights) {
    this(new ArrayVec(weights));
  }

  public Linear(Vec weights) {
    this.weights = weights;
  }

  public Linear(int size, double step) {
    this.weights = VecTools.fill(new ArrayVec(size), step);
  }

  @Override
  public int dim() {
    return weights.dim();
  }

  @Nullable
  @Override
  public Trans gradient() {
    return new Trans.Stub() {
      @Override
      public Vec trans(Vec x) {
        return weights;
      }

      @Override
      public int xdim() {
        return weights.dim();
      }

      public int ydim() {
        return weights.dim();
      }
    };
  }

  public double value(Vec point) {
    return VecTools.multiply(weights, point);
  }

  @Override
  public boolean equals(Object o) {
    if (this == o) return true;
    if (!(o instanceof Linear)) return false;

    Linear that = (Linear) o;
    return that.weights.equals(weights);
  }

  @Override
  public int hashCode() {
    return weights.hashCode();
  }
}
