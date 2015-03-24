package com.spbsu.ml.loss.blockwise;

import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.commons.math.vectors.VecIterator;
import com.spbsu.commons.seq.IntSeq;
import com.spbsu.commons.util.ArrayTools;
import com.spbsu.ml.BlockwiseFuncC1;
import com.spbsu.ml.TargetFunc;
import com.spbsu.ml.data.set.DataSet;

import static java.lang.Math.exp;
import static java.lang.Math.log;

/**
 * We use value representation = \frac{e^x}{e^x + 1}.
 * User: solar
 * Date: 21.12.2010
 * Time: 22:37:55
 */
public class BlockwiseMLLLogit extends BlockwiseFuncC1.Stub implements TargetFunc {
  private final IntSeq target;
  private final DataSet<?> owner;
  private final int classesCount;

  public BlockwiseMLLLogit(final IntSeq target, final DataSet<?> owner) {
    this.target = target;
    this.owner = owner;
    classesCount = ArrayTools.max(target) + 1;
  }

  public BlockwiseMLLLogit(final Vec target, final DataSet<?> owner) {
    final int[] intTarget = new int[target.length()];
    final VecIterator iter = target.nonZeroes();
    while (iter.advance()) {
      intTarget[iter.index()] = (int) iter.value();
    }
    this.target = new IntSeq(intTarget);
    this.owner = owner;
    this.classesCount = ArrayTools.max(this.target) + 1;
  }

  @Override
  public void gradient(final Vec pointBlock, final Vec result, final int blockId) {
    final int blockSize = blockSize();
    double sum = 0.0;
    for (int c = 0; c < blockSize; c++) {
      sum += exp(pointBlock.get(c));
    }
    final int pointClass = target.at(blockId);
    for (int c = 0; c < blockSize; c++) {
      if (pointClass == c)
        result.set(c, -(1. + sum - exp(pointBlock.get(c))) / (1. + sum));
      else
        result.set(c, exp(pointBlock.get(c))/ (1. + sum));
    }
  }

  @Override
  public double value(final Vec pointBlock, final int blockId) {
    double result = 0.0;
    double sum = 0.0;
    for (int c = 0; c < blockSize(); c++) {
      sum += exp(pointBlock.get(c));
    }
    final int pointClass = target.at(blockId);
    if (pointClass != blockSize()) {
      result += log(exp(pointBlock.get(pointClass)) / (1. + sum));
    }
    else {
      result += log(1. / (1. + sum));
    }
    return result;
  }

  @Override
  public double transformResultValue(final double value) {
    return exp(value / target.length());
  }

  @Override
  public int blockSize() {
    return classesCount - 1;
  }

  @Override
  public int dim() {
    return target.length() * (classesCount - 1);
  }

  public int label(final int idx) {
    return target.at(idx);
  }

  public int classesCount() {
    return classesCount;
  }

  public IntSeq labels() {
    return target;
  }

  @Override
  public DataSet<?> owner() {
    return owner;
  }
}
