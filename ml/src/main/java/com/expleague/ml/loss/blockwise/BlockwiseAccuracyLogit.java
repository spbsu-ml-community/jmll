package com.expleague.ml.loss.blockwise;

import com.expleague.commons.math.vectors.Mx;
import com.expleague.commons.math.vectors.Vec;
import com.expleague.commons.seq.IntSeq;
import com.expleague.ml.data.set.DataSet;

import static java.lang.Math.exp;

public class BlockwiseAccuracyLogit extends BlockwiseMLLLogit {
  public BlockwiseAccuracyLogit(IntSeq target, DataSet<?> owner) {
    super(target, owner);
  }

  public BlockwiseAccuracyLogit(Vec target, DataSet<?> owner) {
    super(target, owner);
  }

  @Override
  public double value(Vec pointBlock, int blockId) {
    int result = blockSize();
    double score = 1;
    for (int c = 0; c < blockSize(); c++) {
      double current = exp(pointBlock.get(c));
      if (current > score) {
        result = c;
        score = current;
      }
    }
    final int pointClass = label(blockId);
    return pointClass == result ? 1 : 0;
  }

  @Override
  public double transformResultValue(double value) {
    return value / labels().length();
  }
}
