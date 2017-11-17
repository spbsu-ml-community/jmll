package com.expleague.ml.loss.blockwise;

import com.expleague.commons.math.vectors.Mx;
import com.expleague.commons.math.vectors.VecIterator;
import com.expleague.commons.math.vectors.VecTools;
import com.expleague.commons.seq.IntSeq;
import com.expleague.commons.math.vectors.Vec;
import com.expleague.commons.math.vectors.impl.mx.VecBasedMx;
import com.expleague.commons.util.ArrayTools;
import com.expleague.commons.util.ThreadTools;
import com.expleague.ml.BlockwiseFuncC1;
import com.expleague.ml.TargetFunc;
import com.expleague.ml.data.set.DataSet;

import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ForkJoinPool;

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
    VecTools.assign(result, pointBlock);
    VecTools.exp(result);
    final double sum = VecTools.sum(result);
    final int pointClass = target.at(blockId);
    for (int c = 0; c < blockSize; c++) {
      if (pointClass == c)
        result.set(c, -1 + result.get(c) / (1. + sum));
      else
        result.set(c, result.get(c)/ (1. + sum));
    }
  }

  public void gradient(final Vec result, final int blockId) {
    final int blockSize = blockSize();
    final double sum = VecTools.sum(result);
    final int pointClass = target.at(blockId);
    for (int c = 0; c < blockSize; c++) {
      if (pointClass == c)
        result.set(c, -1 + result.get(c) / (1. + sum));
      else
        result.set(c, result.get(c)/ (1. + sum));
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
  public Vec gradientTo(Vec x, Vec to) {
    VecTools.assign(to, x);
    VecTools.exp(to);
    final int blockSize = blockSize();
    final Mx result = new VecBasedMx(blockSize, to);
    final CountDownLatch latch = new CountDownLatch(ThreadTools.COMPUTE_UNITS - 1);
    for (int t = 0; t < ThreadTools.COMPUTE_UNITS - 1; t++) {
      final int finalT = t;
      ForkJoinPool.commonPool().execute(() -> {
        for (int i = finalT; i < result.rows(); i += ThreadTools.COMPUTE_UNITS - 1) {
          gradient(result.row(i), i);
        }
        latch.countDown();
      });
    }
    try {
      latch.await();
    } catch (InterruptedException e) {
      throw new RuntimeException(e);
    }
    return to;
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
