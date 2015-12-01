package com.spbsu.ml.loss;

import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.commons.math.vectors.impl.vectors.ArrayVec;
import com.spbsu.commons.seq.IntSeq;
import com.spbsu.ml.BlockedTargetFunc;
import com.spbsu.commons.math.FuncC1;
import com.spbsu.ml.data.set.DataSet;
import com.spbsu.ml.func.generic.Log;

import static java.lang.Math.log;

/**
 * We use value representation = \frac{e^x}{e^x + 1}.
 * User: solar
 * Date: 21.12.2010
 * Time: 22:37:55
 */
public class LL extends FuncC1.Stub implements BlockedTargetFunc {
  protected final Vec target;
  private final DataSet<?> owner;

  public LL(final Vec target, final DataSet<?> owner) {
    this.target = target;
    this.owner = owner;
  }

  public LL(final IntSeq target, final DataSet<?> owner) {
    this.target = new ArrayVec(target.length());
    for (int i = 0; i < target.length(); i++) {
      this.target.set(i, target.at(i));
    }
    this.owner = owner;
  }

  @Override
  public Vec gradientTo(final Vec x, final Vec to) {
    for (int i = 0; i < x.dim(); i++) {
      final double pX = x.get(i);
      if (target.get(i) > 0) // positive example
        to.set(i, 1 / pX);
      else // negative
        to.set(i, -1/(1 - pX));
    }
    return to;
  }

  @Override
  public int dim() {
    return target.dim();
  }

  @Override
  public double value(final Vec point) {
    double result = 0;
    for (int i = 0; i < point.dim(); i++) {
      final double pX = point.get(i);
      if (target.get(i) > 0) // positive example
        result += log(pX);
      else // negative
        result += log(1 - pX);
    }

    return result;
  }

  public int label(final int idx) {
    return (int)target.get(idx);
  }

  public Vec labels() {
    return target;
  }

  @Override
  public DataSet<?> owner() {
    return owner;
  }

  @Override
  public Log block(int index) {
    return target.get(index) > 0 ? new Log(1., 0.) : new Log(-1., 1.);
  }

  @Override
  public int blocksCount() {
    return target.dim();
  }
}
