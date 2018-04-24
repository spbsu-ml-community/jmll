package com.expleague.ml.loss;

import com.expleague.commons.math.FuncC1;
import com.expleague.commons.math.vectors.Vec;
import com.expleague.commons.math.vectors.impl.vectors.ArrayVec;
import com.expleague.commons.seq.IntSeq;
import com.expleague.ml.BlockedTargetFunc;
import com.expleague.ml.data.set.DataSet;
import com.expleague.ml.func.generic.LogSoftMax;

public class CrossEntropy extends FuncC1.Stub implements BlockedTargetFunc {
  protected final Vec target;
  private final DataSet<?> owner;
  private final int nClasses;

  public CrossEntropy(final Vec target, final DataSet<?> owner, int nClasses) {
    this.target = target;
    this.owner = owner;
    this.nClasses = nClasses;
  }

  public CrossEntropy(final IntSeq target, final DataSet<?> owner, int nClasses) {
    this.target = new ArrayVec(target.length());
    for (int i = 0; i < target.length(); i++) {
      this.target.set(i, target.at(i));
    }
    this.owner = owner;
    this.nClasses = nClasses;
  }

  @Override
  public Vec gradient(final Vec x) {
    System.out.println("fuck");
    return null;
  }

  @Override
  public int dim() {
    return target.dim();
  }

  @Override
  public double value(final Vec logits) {
    double result = 0.;

    for (int i = 0; i < logits.dim() / nClasses; i++) {
      final int trueClass = label(i);
      result += LogSoftMax.staticValue(logits.sub(i * nClasses, nClasses), trueClass);
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
  public FuncC1 block(int index) {
    return new LogSoftMax(nClasses, label(index));
  }

  @Override
  public int blocksCount() {
    return target.dim();
  }
}
