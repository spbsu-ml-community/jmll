package com.spbsu.ml.loss;

import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.commons.math.vectors.impl.vectors.ArrayVec;
import com.spbsu.commons.seq.IntSeq;
import com.spbsu.commons.seq.IntSeqBuilder;
import com.spbsu.commons.util.ArrayTools;
import com.spbsu.ml.BlockedTargetFunc;
import com.spbsu.commons.math.FuncC1;
import com.spbsu.ml.data.set.DataSet;
import com.spbsu.ml.func.generic.Log;
import com.spbsu.ml.func.generic.ParallelFunc;
import com.spbsu.ml.func.generic.WSum;

import static java.lang.Math.log;

/**
 * We use value representation = \frac{e^x}{e^x + 1}.
 * User: solar
 * Date: 21.12.2010
 * Time: 22:37:55
 */
public class MLL extends FuncC1.Stub implements BlockedTargetFunc {
  protected final IntSeq target;
  private final DataSet<?> owner;
  private final int classesCount;

  public MLL(final Vec target, final DataSet<?> owner) {
    final IntSeqBuilder builder = new IntSeqBuilder();
    int lastClass = 0;
    for (int i = 0; i < target.length(); i++) {
      builder.add((int)target.get(i));
      lastClass = Math.max((int) target.get(i), lastClass);
    }
    this.target = builder.build();
    this.owner = owner;
    this.classesCount = lastClass + 1;
  }

  public MLL(final IntSeq target, final DataSet<?> owner) {
    this.classesCount = ArrayTools.max(target) + 1;
    this.target = target;
    this.owner = owner;
  }

  @Override
  public Vec gradientTo(final Vec x, final Vec to) {
    for (int i = 0; i < target.length(); i++) {
      final int index = i * classesCount + target.intAt(i);
      final double pX = x.get(index);
      to.set(index, 1 / pX);
    }
    return to;
  }

  @Override
  public int dim() {
    return target.length() * classesCount;
  }

  @Override
  public double value(final Vec point) {
    double result = 0;
    for (int i = 0; i < target.length(); i++) {
      result += log(point.get(i * classesCount + target.intAt(i)));
    }

    return result;
  }

  public int label(final int idx) {
    return (int)target.intAt(idx);
  }

  public IntSeq labels() {
    return target;
  }

  @Override
  public DataSet<?> owner() {
    return owner;
  }

  @Override
  public CompositeFunc block(int index) {
    final Vec w = new ArrayVec(classesCount);
    w.set(target.intAt(index), 1.);
    return new CompositeFunc(new WSum(w), new ParallelFunc(classesCount, new Log(1., 0.)));
  }

  @Override
  public int blocksCount() {
    return target.length();
  }

  public int classesCount() {
    return classesCount;
  }
}
