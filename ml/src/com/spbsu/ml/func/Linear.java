package com.spbsu.ml.func;

import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.commons.math.vectors.VecTools;
import com.spbsu.commons.math.vectors.impl.ArrayVec;
import com.spbsu.ml.FuncStub;
import com.spbsu.ml.VecFunc;
import org.jetbrains.annotations.Nullable;

/**
 * User: solar
 * Date: 26.11.12
 * Time: 15:56
 */
public class Linear extends FuncStub {
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
  public int xdim() {
    return weights.dim();
  }

  @Nullable
  @Override
  public VecFunc gradient() {
    return new VecTransform() {
      @Override
      public Vec value(Vec x) {
        return weights;
      }

      @Override
      public int xdim() {
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
