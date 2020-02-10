package com.expleague.ml.loss;

import com.expleague.commons.math.vectors.Vec;
import com.expleague.commons.seq.IntSeq;
import com.expleague.commons.seq.IntSeqBuilder;
import com.expleague.ml.data.set.DataSet;

/**
 * User: starlight
 * Date: 25.12.13
 */
public class AccuracyLogit extends LLLogit {
  private final IntSeq target;

  public AccuracyLogit(final Vec target, final DataSet<?> base) {
    super(target, base);
    final IntSeqBuilder builder = new IntSeqBuilder();
    for (int i = 0; i < target.length(); i++) {
      builder.add((int)target.get(i));
    }
    this.target = builder.build();
  }

  public AccuracyLogit(final IntSeq target, final DataSet<?> base) {
    super(target, base);
    this.target = target;
  }

  @Override
  public double value(final Vec point) {
    int positive = 0;

    for (int i = 0; i < point.dim(); i++) {
      if (point.get(i) > 0 && target.intAt(i) > 0) {
        positive++;
      } else if (point.get(i) < 0 && target.intAt(i) <= 0) {
        positive++;
      }
    }

    return positive / (double)point.dim();
  }
}
