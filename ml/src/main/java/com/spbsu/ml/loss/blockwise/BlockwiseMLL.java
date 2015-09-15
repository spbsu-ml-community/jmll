package com.spbsu.ml.loss.blockwise;

import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.commons.math.vectors.VecIterator;
import com.spbsu.commons.math.vectors.impl.vectors.ArrayVec;
import com.spbsu.commons.seq.IntSeq;
import com.spbsu.commons.util.ArrayTools;
import com.spbsu.ml.BlockedTargetFunc;
import com.spbsu.ml.BlockwiseFuncC1;
import com.spbsu.ml.Func;
import com.spbsu.ml.FuncC1;
import com.spbsu.ml.data.set.DataSet;

import static java.lang.Math.exp;
import static java.lang.Math.log;

/**
 * Created by hrundelb on 12.09.2015.
 */
public class BlockwiseMLL extends BlockwiseFuncC1.Stub implements BlockedTargetFunc {
  private final IntSeq target;
  private final DataSet<?> owner;
  private final int classesCount;

  public BlockwiseMLL(final IntSeq target, final DataSet<?> owner) {
    this.owner = owner;
    this.target = target;
    this.classesCount = ArrayTools.max(target) + 1;
  }

  public BlockwiseMLL(final Vec target, final DataSet<?> owner) {
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
  public void gradient(Vec pointBlock, Vec result, int index) {
    final int blockSize = blockSize();
    double sum = 1.0;
    for (int c = 0; c < blockSize; c++) {
      sum += pointBlock.get(c);
    }
    final int pointClass = target.at(index);
    for (int c = 0; c < blockSize; c++) {
      if (pointClass == c)
        result.set(c, (sum - pointBlock.get(c)) / (pointBlock.get(c) * sum));
      else
        result.set(c, -1 / sum);
    }
  }

  @Override
  public double value(Vec pointBlock, int index) {
    double result = 0.0;
    double sum = 0.0;
    for (int c = 0; c < blockSize(); c++) {
      sum += pointBlock.get(c);
    }
    final int pointClass = target.at(index);
    if (pointClass != blockSize()) {
      result += log(pointBlock.get(pointClass) / (1. + sum));
    } else {
      result += log(1. / (1. + sum));
    }
    return result;
  }

  @Override
  public double transformResultValue(double value) {
    return exp(value / target.length());
  }

  @Override
  public int blockSize() {
    return classesCount - 1;
  }

  @Override
  public DataSet<?> owner() {
    return owner;
  }

  @Override
  public int dim() {
    return target.length() * (classesCount - 1);
  }


  @Override
  public Func block(final int index) {
    return new FuncC1.Stub() {
      @Override
      public Vec gradient(Vec x) {
        Vec result = new ArrayVec(this.dim());
        BlockwiseMLL.this.gradient(x, result, index);
        return result;
      }

      @Override
      public double value(Vec x) {
        return BlockwiseMLL.this.value(x, index);
      }

      @Override
      public int dim() {
        return blockSize();
      }
    };
  }

  @Override
  public int blocksCount() {
    return target.length();
  }
}
