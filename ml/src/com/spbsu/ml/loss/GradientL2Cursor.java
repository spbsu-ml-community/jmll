package com.spbsu.ml.loss;

import com.spbsu.commons.func.Computable;
import com.spbsu.commons.func.impl.WeakListenerHolderImpl;
import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.commons.math.vectors.impl.ArrayVec;
import com.spbsu.ml.CursorOracle;
import com.spbsu.ml.Oracle1;

/**
* User: solar
* Date: 18.11.13
* Time: 18:40
*/
public class GradientL2Cursor<O extends Oracle1> extends WeakListenerHolderImpl<CursorOracle.CursorMoved> implements CursorOracle<L2>, Oracle1 {
  Vec point;
  O initial;
  private final Computable<Vec, L2> factory;

  public GradientL2Cursor(O initial, Computable<Vec, L2> factory) {
    this.initial = initial;
    this.factory = factory;
    point = new ArrayVec(initial.dim());
  }

  public Vec cursor() {
    return point;
  }

  public Vec moveTo(Vec m) {
    return point = m;
  }

  public L2 local() {
    return factory.compute(initial.gradient(point));
  }

  public double value(Vec x) {
    return initial.value(x);
  }

  @Override
  public Vec gradient(Vec x) {
    return initial.gradient(x);
  }

  @Override
  public int dim() {
    return initial.dim();
  }
}
