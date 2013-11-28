package com.spbsu.ml.loss;

import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.commons.math.vectors.impl.ArrayVec;
import com.spbsu.ml.FuncStub;
import com.spbsu.ml.VecFunc;
import com.spbsu.ml.func.VecTransform;
import com.sun.javafx.beans.annotations.NonNull;

import static java.lang.Math.exp;

/**
 * We use value representation = \frac{e^x}{e^x + 1}.
 * User: solar
 * Date: 21.12.2010
 * Time: 22:37:55
 */
public class CELogit extends FuncStub {
  private final Vec target;

  public CELogit(Vec target) {
    this.target = target;
  }

  @NonNull
  @Override
  public VecFunc gradient() {
    return new VecTransform() {
      @Override
      public Vec vvalue(Vec x) {
        Vec result = new ArrayVec(x.dim());
        for (int i = 0; i < x.dim(); i++) {
          double b = target.get(i) > 0 ? 1 : -1;
          double a = exp(x.get(i) * b);
          result.set(i, 2 * a * b / (1 + a) / (1 + a));
        }
        return result;
      }

      @Override
      public int xdim() {
        return target.dim();
      }
    };
  }

  public int xdim() {
    return target.dim();
  }

  public double value(Vec point) {
    double result = 0;
    for (int i = 0; i < point.dim(); i++) {
      double expMX = exp(-point.get(i));
      double pX = 1. / (1. + expMX);
      if (target.get(i) > 0) // positive example
        result += 1 - pX;
      else // negative
        result += pX;
    }

    return result / point.dim();
  }
}
