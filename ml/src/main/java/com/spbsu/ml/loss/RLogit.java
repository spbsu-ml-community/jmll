package com.spbsu.ml.loss;

import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.commons.seq.IntSeq;
import com.spbsu.commons.seq.IntSeqBuilder;
import com.spbsu.ml.data.set.DataSet;

/**
 * User: starlight
 * Date: 25.12.13
 */
public class RLogit extends LLLogit {
  private final IntSeq target;

  public RLogit(final Vec target, final DataSet<?> base) {
    super(target, base);
    final IntSeqBuilder builder = new IntSeqBuilder();
    for (int i = 0; i < target.length(); i++) {
      builder.add((int)target.get(i));
    }
    this.target = builder.build();
  }

  public RLogit(final IntSeq target, final DataSet<?> base) {
    super(target, base);
    this.target = target;
  }

  @Override
  public double value(final Vec point) {
    int truePositive = 0;
    int falseNegative = 0;

    for (int i = 0; i < point.dim(); i++) {
      if (point.get(i) > 0 && target.intAt(i) > 0) {
        truePositive++;
      } else if (point.get(i) <= 0 && target.intAt(i) > 0) {
        falseNegative++;
      }
    }

    return truePositive / (0. + truePositive + falseNegative);
  }
}
