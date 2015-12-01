package com.spbsu.ml.loss.multiclass;

import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.commons.math.vectors.VecIterator;
import com.spbsu.commons.seq.IntSeq;
import com.spbsu.commons.math.Func;
import com.spbsu.ml.data.set.DataSet;

/**
 * User: qdeee
 * Date: 09.04.14
 */
public class MCMicroPrecision extends Func.Stub implements ClassicMulticlassLoss {
  protected final IntSeq target;
  private final DataSet<?> owner;

  public MCMicroPrecision(final IntSeq target, final DataSet<?> owner) {
    this.target = target;
    this.owner = owner;
  }

  public MCMicroPrecision(final Vec target, final DataSet<?> owner) {
    final int[] intTarget = new int[target.length()];
    final VecIterator iter = target.nonZeroes();
    while (iter.advance()) {
      intTarget[iter.index()] = (int) iter.value();
    }
    this.target = new IntSeq(intTarget);
    this.owner = owner;
  }

  @Override
  public double value(final Vec x) {
    int tp = 0;
    int fp = 0;
    for (int i = 0; i < x.dim(); i++) {
      final int expected = target.at(i);
      final int actual = (int) x.get(i);

      //skip unrecognized class
      if (actual == -1)
        continue;

      if (actual == expected)
        tp += 1;
      else
        fp += 1;
    }
    return tp / (tp + fp + 0.);
  }

  @Override
  public int dim() {
    return target.length();
  }

  @Override
  public DataSet<?> owner() {
    return owner;
  }

  @Override
  public IntSeq labels() {
    return target;
  }
}
